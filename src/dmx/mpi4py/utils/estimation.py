"""MPI-enabled estimation helpers for `dmx.stats` models."""

# pylint: disable=duplicate-code

import sys
from typing import IO, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.random import RandomState

from dmx.mpi4py.stats import (
    seq_encode_mpi,
    seq_estimate_mpi,
    seq_initialize_mpi,
    seq_log_density_sum_mpi,
)
from dmx.mpi4py.utils import get_runtime_attr
from dmx.stats.pdist import (
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)

T = TypeVar("T")


# Keep `init_p` handling consistent with `dmx.utils.estimation`.
def _validate_init_p(init_p: float) -> None:
    """Validate the EM initialization proportion."""
    if not 0 < init_p <= 1:
        raise ValueError(
            f"Invalid init_p: {init_p}. It must be greater than 0 and less "
            "than or equal to 1."
        )


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def optimize_mpi(
    data: Optional[Sequence[T]],
    estimator: ParameterEstimator,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-9,
    init_estimator: Optional[ParameterEstimator] = None,
    init_p: float = 0.1,
    rng: RandomState = RandomState(),
    prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[List[Tuple[int, EncodedDataSequence]]] = None,
    enc_vdata: Optional[List[Tuple[int, EncodedDataSequence]]] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> Optional[SequenceEncodableProbabilityDistribution]:
    """Run EM estimation with MPI until convergence or `max_its`.

    This is collective over `MPI.COMM_WORLD`; every rank must call it in the
    same order. Raw `data` and `vdata` are required only on rank 0, but
    pre-encoded `enc_data` and `enc_vdata` must be supplied on every rank
    because their availability is checked with an all-reduce. Iterative model
    objects live on rank 0, while worker ranks generally receive and return
    `None` from the lower-level estimation helpers. Likelihood sums are reduced
    across all ranks and progress is written only by rank 0.

    Args:
        data (Optional[Sequence[T]]): Observed data compatible with `estimator`.
            Required on rank 0 unless encoded chunks are supplied on all ranks.
        estimator (ParameterEstimator): Estimator for the target model. Rank 0
            must receive a usable estimator.
        max_its (int): Maximum number of EM iterations to perform.
        delta (Optional[float]): Stop when the signed likelihood improvement is
            less than `delta`. If `None`, likelihood-decreasing updates are
            accepted.
        init_estimator (Optional[ParameterEstimator]): Estimator used for
            initialization. If `None`, use `estimator`.
        init_p (float): Initialization proportion in `(0, 1]`.
        rng (RandomState): RandomState used to set seeds for EM
            initialization. The default object is process-local and is advanced
            across calls.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution]):
            Optional previous model estimate, meaningful on rank 0 and
            consistent with `estimator`.
        enc_data (Optional[List[Tuple[int, EncodedDataSequence]]]): Optional
            rank-local encoded chunks. Formed from root `data` if omitted.
        enc_vdata (Optional[List[Tuple[int, EncodedDataSequence]]]): Optional
            rank-local encoded validation chunks.
        out (IO): Rank 0 stream for EM progress.
        print_iter (int): Print progress every `print_iter` iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        Optional[SequenceEncodableProbabilityDistribution]: Estimated model on
        rank 0 and `None` on worker ranks.

    Raises:
        ValueError: If rank 0 has no raw data and encoded data are not present
            on every rank, or if `init_p` is outside `(0, 1]`.
        RuntimeError: If encoded data expected on a rank are missing.
    """
    mpi = get_runtime_attr("mpi4py", "MPI")
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    # check if encoded data is already on each worker
    enc_data_exists = enc_data is not None
    enc_data_exists_all = comm.allreduce(enc_data_exists, op=mpi.LAND)
    if world_rank == 0:
        data_exception = data is None
    else:
        data_exception = None

    # enc_data_exists_all = comm.bcast(enc_data_exists_all, root=0)
    data_exception = comm.bcast(data_exception, root=0)

    if data_exception and not enc_data_exists_all:
        raise ValueError(
            "Optimization called with empty data on rank 0 and encoded data "
            "does not exist."
        )

    est = estimator if init_estimator is None else init_estimator

    if world_rank == 0:
        if prev_estimate is not None:
            data_encoder = prev_estimate.dist_to_encoder()
            mm = prev_estimate
            skip_init = True
        else:
            data_encoder = est.accumulator_factory().make().acc_to_encoder()
            mm = None
            skip_init = False
    else:
        data_encoder = None
        mm = None
        skip_init = None

    # has prev_estimate been passed to root
    skip_init = comm.bcast(skip_init, root=0)

    if not enc_data_exists_all:
        enc_data = seq_encode_mpi(
            data=data, encoder=data_encoder, num_chunks=num_chunks
        )
    if enc_data is None:
        raise RuntimeError("Encoded data is missing on this MPI rank.")
    encoded_data = enc_data

    if not skip_init:
        _validate_init_p(init_p)
        mm = seq_initialize_mpi(encoded_data, estimator=est, rng=rng, p=init_p)

    _, old_ll = seq_log_density_sum_mpi(enc_data=encoded_data, estimate=mm)

    # check if validation data is passed
    # check if encoded data is already on each worker
    enc_vdata_exists = enc_vdata is not None
    enc_vdata_exists_all = comm.allreduce(enc_vdata_exists, op=mpi.LAND)
    if world_rank == 0:
        vdata_exists = vdata is not None
    else:
        vdata_exists = None

    vdata_exists = comm.bcast(vdata_exists, root=0)

    if not enc_vdata_exists_all and vdata_exists:
        enc_vdata = seq_encode_mpi(vdata, encoder=data_encoder, num_chunks=num_chunks)
        enc_vdata_exists_all = True

    if enc_vdata_exists_all:
        if enc_vdata is None:
            raise RuntimeError("Encoded validation data is missing on this MPI rank.")
        _, old_vll = seq_log_density_sum_mpi(enc_vdata, mm)
        validation_data: Optional[Sequence[Tuple[int, EncodedDataSequence]]] = enc_vdata
    else:
        validation_data = None
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate_mpi(
            enc_data=encoded_data, estimator=est, prev_estimate=mm
        )
        _, ll = seq_log_density_sum_mpi(enc_data=encoded_data, estimate=mm_next)

        if validation_data is not None:
            _, vll = seq_log_density_sum_mpi(enc_data=validation_data, estimate=mm_next)
        else:
            vll = ll

        dll = ll - old_ll

        if (dll >= 0) or (delta is None):
            mm = mm_next

        # converged in delta tolerance
        if (delta is not None) and (dll < delta):
            if world_rank == 0:
                if enc_vdata_exists_all:
                    out.write(
                        f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                        f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}, "
                        f"ln[p_mat(Valid Data|Model)]={vll:e}\n"
                    )
                else:
                    out.write(
                        f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                        f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}\n"
                    )

            break

        if world_rank == 0:
            if (i + 1) % print_iter == 0:
                if enc_vdata_exists_all:
                    out.write(
                        f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                        f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}, "
                        f"ln[p_mat(Valid Data|Model)]={vll:e}\n"
                    )
                else:
                    out.write(
                        f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                        f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}\n"
                    )

        old_ll = ll

        if best_vll < vll:
            best_vll = vll
            best_model = mm

    return best_model


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def best_of_mpi(
    data: Optional[Sequence[T]],
    vdata: Optional[Sequence[T]],
    est: ParameterEstimator,
    trials: int,
    max_its: int,
    max_its_cnt: int,
    init_p: float,
    delta: float,
    rng: RandomState,
    init_estimator: Optional[ParameterEstimator] = None,
    enc_data: Optional[List[Tuple[int, EncodedDataSequence]]] = None,
    enc_vdata: Optional[Sequence[Tuple[int, EncodedDataSequence]]] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
) -> Optional[SequenceEncodableProbabilityDistribution]:
    """Run multiple MPI EM initializations and keep the best model.

    This is collective over `MPI.COMM_WORLD`; every rank must call it in the
    same order. Raw data are needed only on rank 0 unless every rank supplies
    its local encoded chunks. Each trial starts from a randomized
    initialization, runs at least one EM iteration, scores by validation
    likelihood when validation data are supplied, and then refines the best
    model through `optimize_mpi` for `max_its_cnt` additional iterations.
    Progress and the returned model are root-only.

    Args:
        data (Optional[Sequence[T]]): Data of type `T`. If `None` on rank 0,
            `enc_data` must be provided on every rank.
        vdata (Optional[Sequence[T]]): Optional validation set on rank 0.
        est (ParameterEstimator): Estimator for model to be estimated.
        trials (int): Number of randomized initial conditions to try; coerced
            to at least one.
        max_its (int): Maximum number of EM iterations per trial; coerced to at
            least one.
        max_its_cnt (int): Number of additional `optimize_mpi` iterations for
            the best trial model.
        init_p (float): Initialization proportion in `(0, 1]`.
        delta (float): Stop when signed likelihood improvement is less than
            `delta`.
        rng (RandomState): RandomState used to set trial seeds. It is advanced
            across calls.
        init_estimator (Optional[ParameterEstimator]): Optional estimator used
            for fitting.
        enc_data (Optional[List[Tuple[int, EncodedDataSequence]]]): Optional
            rank-local encoded chunks.
        enc_vdata (Optional[Sequence[Tuple[int, EncodedDataSequence]]]):
            Optional rank-local encoded validation chunks.
        out (IO): Rank 0 text output stream.
        print_iter (int): Print progress every `print_iter` iterations.

    Returns:
        Optional[SequenceEncodableProbabilityDistribution]: Best fitting model
        on rank 0 and `None` on worker ranks.

    Raises:
        ValueError: If rank 0 has no raw data and encoded data are not present
            on every rank, or if `init_p` is outside `(0, 1]`.
        RuntimeError: If encoded data expected on a rank are missing.
    """
    mpi = get_runtime_attr("mpi4py", "MPI")
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    # check if encoded data is already on each worker
    enc_data_exists = enc_data is not None
    enc_data_exists_all = comm.allreduce(enc_data_exists, op=mpi.LAND)
    if world_rank == 0:
        data_exception = data is None
    else:
        data_exception = None

    # enc_data_exists_all = comm.bcast(enc_data_exists_all, root=0)
    data_exception = comm.bcast(data_exception, root=0)

    if data_exception and not enc_data_exists_all:
        raise ValueError(
            "Optimization called with empty data on rank 0 and encoded data "
            "does not exist."
        )

    est = est if init_estimator is None else init_estimator

    if world_rank == 0:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        data_encoder = None

    # has prev_estimate been passed to root
    data_encoder = comm.bcast(data_encoder, root=0)

    if not enc_data_exists_all:
        enc_data = seq_encode_mpi(data=data, encoder=data_encoder)
    if enc_data is None:
        raise RuntimeError("Encoded data is missing on this MPI rank.")
    encoded_data = enc_data

    _validate_init_p(init_p)

    # check if validation data is passed
    # check if encoded data is already on each worker
    enc_vdata_exists = enc_vdata is not None
    enc_vdata_exists_all = comm.allreduce(enc_vdata_exists, op=mpi.LAND)
    if world_rank == 0:
        vdata_exists = vdata is not None
    else:
        vdata_exists = None

    vdata_exists = comm.bcast(vdata_exists, root=0)

    if not enc_vdata_exists_all and vdata_exists:
        enc_vdata = seq_encode_mpi(vdata, encoder=data_encoder)
        enc_vdata_exists_all = True
    validation_data: Optional[Sequence[Tuple[int, EncodedDataSequence]]]
    if enc_vdata_exists_all:
        if enc_vdata is None:
            raise RuntimeError("Encoded validation data is missing on this MPI rank.")
        validation_data = enc_vdata
    else:
        validation_data = None

    rv_ll = -np.inf
    rv_mm = None
    i_est = est if init_estimator is None else init_estimator

    max_its = max(max_its, 1)
    trials = max(trials, 1)

    for kk in range(trials):

        mm = seq_initialize_mpi(encoded_data, i_est, rng, init_p)
        _, old_ll = seq_log_density_sum_mpi(encoded_data, mm)

        for i in range(max_its):

            mm_next = seq_estimate_mpi(encoded_data, est, mm)
            _, ll = seq_log_density_sum_mpi(encoded_data, mm_next)
            dll = ll - old_ll

            if world_rank == 0:
                if (i + 1) % print_iter == 0:
                    out.write(f"Iteration {i + 1}. LL={ll:f}, delta LL={dll:e}\n")

            if (dll >= 0) or (delta is None):
                mm = mm_next

            if (delta is not None) and (dll < delta):
                break

            old_ll = ll

        score_data = validation_data if validation_data is not None else encoded_data
        _, vll = seq_log_density_sum_mpi(score_data, mm)
        if world_rank == 0:
            out.write(f"Trial {kk + 1}. VLL={vll:f}\n")

        if vll > rv_ll:
            rv_mm = mm
            rv_ll = vll

    # iterate further on best model
    rv_mm = optimize_mpi(
        data=None,
        enc_data=encoded_data,
        estimator=est,
        rng=rng,
        init_p=init_p,
        delta=delta,
        print_iter=print_iter,
        prev_estimate=rv_mm,
        max_its=max_its_cnt,
    )

    return rv_mm
