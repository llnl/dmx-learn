"""MPI-enabled estimation helpers for `dmx.stats` models."""

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
) -> SequenceEncodableProbabilityDistribution:
    """Run EM estimation with MPI until convergence or `max_its`.

    Args:
        data (Optional[List[T]]): Observed data compatible with `estimator`.
        estimator (ParameterEstimator): Estimator for the target model.
        max_its (int): Maximum number of EM iterations to perform.
        delta (Optional[float]): Stop when
            `|old_loglikelihood - new_loglikelihood| < delta`, or when
            `max_its` is reached.
        init_estimator (Optional[ParameterEstimator]): Estimator used for
            initialization. If `None`, use `estimator`.
        init_p (float): Initialization proportion in `(0, 1]`.
        rng (RandomState): RandomState used to set seed for initializing EM algorithm.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution]):
            Optional previous model estimate consistent with `estimator`.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            `List[Tuple[int, E]]`. Formed from `data` if `None`.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded
            validation set.
        out (IO): IO stream to write out iterations of EM algorithm.
        print_iter (int): Print progress every `print_iter` iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        SequenceEncodableProbabilityDistribution: Estimated model.
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

    if not skip_init:
        _validate_init_p(init_p)
        mm = seq_initialize_mpi(enc_data, estimator=est, rng=rng, p=init_p)

    _, old_ll = seq_log_density_sum_mpi(enc_data=enc_data, estimate=mm)

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
        _, old_vll = seq_log_density_sum_mpi(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate_mpi(enc_data=enc_data, estimator=est, prev_estimate=mm)
        _, ll = seq_log_density_sum_mpi(enc_data=enc_data, estimate=mm_next)

        if enc_vdata_exists_all:
            _, vll = seq_log_density_sum_mpi(enc_data=enc_vdata, estimate=mm_next)
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
) -> SequenceEncodableProbabilityDistribution:
    """Run multiple MPI EM initializations and keep the best model.

    Args:
        data (Optional[List[T]]): Data of type `T`. If `None`, `enc_data` must
            be provided.
        vdata (Optional[Sequence[T]]): Optional validation set.
        est (ParameterEstimator): ParameterEstimator for model to be estimated.
        trials (int): Number of randomized initial conditions to try.
        max_its (int): Maximum number of EM iterations per trial.
        init_p (float): Initialization proportion in `(0, 1]`.
        delta (float): Stop when
            `|old-log-likelihood - new-log-likelihood| < delta`.
        rng (RandomState): RandomState for setting seed.
        init_estimator (Optional[ParameterEstimator]): Optional estimator used
            for fitting.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data. If
            provided, `data` need not be provided.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded
            validation set.
        out (I0): Text output stream.
        print_iter (int): Print progress every `print_iter` iterations.

    Returns:
        SequenceEncodableProbabilityDistribution: Best fitting model.
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

    rv_ll = -np.inf
    rv_mm = None
    i_est = est if init_estimator is None else init_estimator

    max_its = max(max_its, 1)
    trials = max(trials, 1)

    for kk in range(trials):

        mm = seq_initialize_mpi(enc_data, i_est, rng, init_p)
        _, old_ll = seq_log_density_sum_mpi(enc_data, mm)

        for i in range(max_its):

            mm_next = seq_estimate_mpi(enc_data, est, mm)
            _, ll = seq_log_density_sum_mpi(enc_data, mm_next)
            dll = ll - old_ll

            if world_rank == 0:
                if (i + 1) % print_iter == 0:
                    out.write(f"Iteration {i + 1}. LL={ll:f}, delta LL={dll:e}\n")

            if (dll >= 0) or (delta is None):
                mm = mm_next

            if (delta is not None) and (dll < delta):
                break

            old_ll = ll

        _, vll = seq_log_density_sum_mpi(enc_vdata, mm)
        if world_rank == 0:
            out.write(f"Trial {kk + 1}. VLL={vll:f}\n")

        if vll > rv_ll:
            rv_mm = mm
            rv_ll = vll

    # iterate further on best model
    rv_mm = optimize_mpi(
        data=None,
        enc_data=enc_data,
        estimator=est,
        rng=rng,
        init_p=init_p,
        delta=delta,
        print_iter=print_iter,
        prev_estimate=rv_mm,
        max_its=max_its_cnt,
    )

    return rv_mm
