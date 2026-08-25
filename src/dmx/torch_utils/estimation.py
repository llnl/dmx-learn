"""Torch-backed estimation helpers for local and distributed EM fitting."""

import sys
from typing import IO, Any, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch as tn

from dmx.torch_stats import (
    seq_encode,
    seq_encode_mp,
    seq_estimate,
    seq_estimate_mp,
    seq_initialize,
    seq_initialize_mp,
    seq_log_density_sum,
    seq_log_density_sum_mp,
)
from dmx.torch_stats.pdist import (
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
)
from dmx.torch_utils.vector import (
    DeviceLike,
    float_dtype_for_device,
    resolve_device,
    set_default_float_dtype,
)

T = TypeVar("T")
EncodedChunks = List[Tuple[int, Any]]
MpEncodedChunks = Sequence[Tuple[int, Any]]


def _seq_log_density_sum_mp_checked(
    world_rank: int,
    enc_data: MpEncodedChunks,
    estimate: TorchProbabilityDistribution,
) -> Tuple[Optional[float], Optional[float]]:
    result = seq_log_density_sum_mp(
        world_rank=world_rank, enc_data=enc_data, estimate=estimate
    )
    if result is None:
        raise RuntimeError("Distributed log-density sum did not return a result tuple.")
    return result


def empirical_kl_divergence(
    dist1: TorchProbabilityDistribution,
    dist2: TorchProbabilityDistribution,
    enc_data: List[Tuple[int, TorchEncodedSequence]],
) -> Tuple[float, float, float]:
    """Compute the empirical KL-divergence between two densities.

    Compute the KL-divergence between `dist1` and `dist2` for an encoded
    sequence of data. Both distributions must use the same encodings, and the
    encoded tensors must already be on devices compatible with the two
    distributions. Only the first encoded chunk in `enc_data` is evaluated.

    Args:
        dist1 (TorchProbabilityDistribution): Distribution compatible with enc_data.
        dist2 (TorchProbabilityDistribution): Distribution compatible with enc_data.
        enc_data (List[Tuple[int, TorchEncodedSequence]]): List of tuples
            containing chunk size and `TorchEncodedSequence`.

    Returns:
        Tuple[float, float, float]: KL-divergence estimate, number of bad
        likelihood values for `dist1`, and number of bad likelihood values for
        `dist2`.
    """
    l1 = dist1.seq_log_density(enc_data[0][1])
    l2 = dist2.seq_log_density(enc_data[0][1])
    g1 = tn.bitwise_and(l1 != -tn.inf, ~tn.isnan(l1))
    g2 = tn.bitwise_and(l2 != -tn.inf, ~tn.isnan(l2))
    gg = tn.bitwise_and(g1, g2)

    max_l1 = tn.max(l1[gg])
    max_l2 = tn.max(l2[gg])

    p1 = tn.exp(l1[gg] - max_l1)
    p1 /= p1.sum()

    p2 = tn.exp(l2[gg] - max_l2)
    p2 /= p2.sum()

    r1 = (p1[gg] * (tn.log(p1[gg]) - tn.log(p2[gg]))).sum()
    r2 = (~g1).sum()
    r3 = (~g2).sum()

    return float(r1), float(r2), float(r3)


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def optimize(
    data: Optional[Sequence[T]],
    estimator: TorchParameterEstimator,
    seed: Optional[int] = None,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-9,
    init_estimator: Optional[TorchParameterEstimator] = None,
    init_p: float = 0.1,
    device: DeviceLike = None,
    prev_estimate: Optional[TorchProbabilityDistribution] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> TorchProbabilityDistribution:
    """Estimate `estimator` via EM until convergence or `max_its`.

    With `device=None`, the target device is auto-detected in CUDA, MPS, CPU
    order. The module-local default float dtype is set to `float64`, except MPS
    uses `float32`. Raw data are encoded on the target device; caller-supplied
    encoded data are used as-is. A previous estimate is moved to the target
    device in place before fitting.

    Args:
        data (Optional[Sequence[T]]): Observed data of type `T`. Must be
            compatible with the estimator.
        estimator (TorchParameterEstimator): Estimator used to specify the
            distribution for observed data.
        seed (Optional[int]): Seed for initialization. If `None`, a seed is
            drawn from NumPy's global random state.
        max_its (int): Maximum number of EM iterations to be performed.
            Default value is 10 iterations.
        delta (Optional[float]): Stop when the signed likelihood improvement
            `new_loglikelihood - old_loglikelihood` is less than `delta`. If
            `None`, likelihood-decreasing updates are accepted.
        init_estimator (Optional[TorchParameterEstimator]): Estimator used to
            initialize EM parameters. If `None`, `estimator` is used.
        init_p (float): Proportion of data points used in initialization.
            Values at or below zero become `0.1`; values above one are clamped
            to one.
        device (DeviceLike): Device used for tensor calculations. Strings are
            resolved to torch devices; `None` defaults to auto-detection.
        prev_estimate (Optional[TorchProbabilityDistribution]): Optional model
            estimate from prior fitting. Must be consistent with `estimator`.
        vdata (Optional[Sequence[T]]): Optional validation set.
        enc_data (Optional[List[Tuple[int, Any]]]): Optional encoded chunks.
            Formed from `data` when `None`.
        enc_vdata (Optional[List[Tuple[int, Any]]]): Optional encoded validation
            chunks.
        out (IO): IO stream to write EM progress.
        print_iter (int): Print likelihood progress every `print_iter` iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        TorchProbabilityDistribution: Best model by validation likelihood, or
        training likelihood when no validation data are supplied.

    Raises:
        ValueError: If both `data` and `enc_data` are `None`.
    """
    target_device = resolve_device(device)
    set_default_float_dtype(float_dtype_for_device(target_device))

    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    est = estimator if init_estimator is None else init_estimator

    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        prev_estimate.to(target_device)
        data_encoder = prev_estimate.dist_to_encoder()

    if enc_data is None:
        assert data is not None
        enc_data = seq_encode(
            data=data, encoder=data_encoder, num_chunks=num_chunks, device=target_device
        )

    if prev_estimate is None:
        if init_p <= 0.0:
            p = 0.10
        else:
            p = min(max(init_p, 0.0), 1.0)

        seed = seed if seed is not None else np.random.randint(2**31)
        mm = seq_initialize(
            enc_data=enc_data, estimator=est, seed=seed, p=p, device=target_device
        )
    else:
        mm = prev_estimate

    _, old_ll = seq_log_density_sum(enc_data=enc_data, estimate=mm)

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode(
            data=vdata,
            encoder=data_encoder,
            num_chunks=num_chunks,
            device=target_device,
        )

    if enc_vdata is not None:
        _, old_vll = seq_log_density_sum(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate(enc_data=enc_data, estimator=estimator, prev_estimate=mm)
        _, ll = seq_log_density_sum(enc_data=enc_data, estimate=mm_next)

        if enc_vdata is not None:
            _, vll = seq_log_density_sum(enc_vdata, mm_next)
        else:
            vll = ll

        dll = ll - old_ll

        if (dll >= 0) or (delta is None):
            mm = mm_next

        if (delta is not None) and (dll < delta):
            if enc_vdata is not None:
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

        if (i + 1) % print_iter == 0:
            if enc_vdata is not None:
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
# pylint: disable-next=too-many-positional-arguments,unused-argument
def optimize_mp(
    world_rank: int,
    world_size: int,
    data: Optional[Sequence[T]],
    estimator: TorchParameterEstimator,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-9,
    init_estimator: Optional[TorchParameterEstimator] = None,
    init_p: float = 0.1,
    seed: Optional[int] = None,
    prev_estimate: Optional[TorchProbabilityDistribution] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[MpEncodedChunks] = None,
    enc_vdata: Optional[MpEncodedChunks] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> TorchProbabilityDistribution:
    """Estimate `estimator` via distributed torch EM until convergence.

    This helper assumes an initialized `torch.distributed` process group. Every
    rank must call it in the same collective order, and `world_rank` and
    `world_size` must match that process group. Raw data are scattered from rank
    0 when encoded chunks are not supplied; pre-encoded chunks are expected to
    be local to each rank and already on compatible devices. Rank 0 makes
    update, stopping, and validation-selection decisions and broadcasts those
    decisions to the other ranks.

    Args:
        world_rank (int): Rank of this worker in the process group.
        world_size (int): Number of workers in the process group.
        data (Optional[Sequence[T]]): Observed data compatible with the
            estimator. Required on rank 0 unless local `enc_data` is supplied.
        estimator (TorchParameterEstimator): Estimator used to specify the
            distribution for observed data. It should be equivalent on all ranks.
        max_its (int): Maximum number of EM iterations to be performed.
            Default value is 10 iterations.
        delta (Optional[float]): Stop when rank 0 observes signed likelihood
            improvement below `delta`. If `None`, likelihood-decreasing updates
            are accepted.
        init_estimator (Optional[TorchParameterEstimator]): Estimator used to
            initialize EM parameters. If `None`, `estimator` is used.
        init_p (float): Proportion of data points used in initialization.
            Values at or below zero become `0.1`; values above one are clamped
            to one.
        seed (Optional[int]): Seed for initialization. If `None`, each rank
            draws from its NumPy global random state before the collective
            initialization call, so callers should synchronize this value when
            reproducibility across ranks matters.
        prev_estimate (Optional[TorchProbabilityDistribution]): Optional model
            estimate from prior fitting, expected to be available consistently
            on all ranks.
        vdata (Optional[Sequence[T]]): Optional validation set.
        enc_data (Optional[Sequence[Tuple[int, Any]]]): Optional rank-local
            encoded chunks. Formed from `data` when `None`.
        enc_vdata (Optional[Sequence[Tuple[int, Any]]]): Optional rank-local
            encoded validation chunks.
        out (IO): Rank 0 output stream for EM progress.
        print_iter (int): Print likelihood progress every `print_iter` iterations.
        num_chunks (int): Currently unused in the distributed encode path.

    Returns:
        TorchProbabilityDistribution: Best model by validation likelihood, or
        training likelihood when no validation data are supplied, on each rank.

    Raises:
        ValueError: If both `data` and `enc_data` are `None`.
        RuntimeError: If a distributed reduction or estimation step fails to
            return the expected result.
    """
    # Kept for API symmetry with `optimize`, even though the mp path does not
    # currently chunk encoded data in the same way.
    # pylint: disable=unused-argument
    # data on all nodes assumed for now. Can change this later.
    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    # estimator defined on all nodes
    est = estimator if init_estimator is None else init_estimator

    # create data encoder (prev_estimate lives on all nodes)
    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        data_encoder = prev_estimate.dist_to_encoder()

    # encode the data. Chunked to each worker.
    if enc_data is None:
        enc_data = seq_encode_mp(
            world_rank=world_rank,
            world_size=world_size,
            data=data,
            encoder=data_encoder,
        )

    if prev_estimate is None:
        p = 0.10 if init_p <= 0.0 else min(max(init_p, 0.0), 1.0)
        seed = np.random.randint(2**31) if seed is None else seed
        mm = seq_initialize_mp(
            world_rank=world_rank,
            world_size=world_size,
            enc_data=enc_data,
            estimator=est,
            seed=seed,
            p=p,
        )

    else:
        mm = prev_estimate

    # none on all except master
    _, old_ll_opt = _seq_log_density_sum_mp_checked(world_rank, enc_data, mm)
    old_ll = 0.0 if old_ll_opt is None else old_ll_opt

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode_mp(
            world_rank=world_rank,
            world_size=world_size,
            data=vdata,
            encoder=data_encoder,
        )

    if enc_vdata is not None:
        _, old_vll_opt = _seq_log_density_sum_mp_checked(world_rank, enc_vdata, mm)
        old_vll = 0.0 if old_vll_opt is None else old_vll_opt
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):
        # condition for stopping EM, updating model, and validation data
        break_cond = [False]
        update_model = [False]
        vflag = [False]

        maybe_mm_next = seq_estimate_mp(
            world_rank=world_rank,
            world_size=world_size,
            enc_data=enc_data,
            estimator=est,
            prev_estimate=mm,
        )
        if maybe_mm_next is None:
            raise RuntimeError("Distributed estimate did not produce a model.")
        mm_next = maybe_mm_next

        _, ll_opt = _seq_log_density_sum_mp_checked(world_rank, enc_data, mm_next)
        ll = 0.0 if ll_opt is None else ll_opt

        if enc_vdata is not None:
            _, vll_opt = _seq_log_density_sum_mp_checked(world_rank, enc_vdata, mm_next)
            vll = 0.0 if vll_opt is None else vll_opt
        else:
            vll = ll

        # check if model should be updated
        if world_rank == 0:
            dll = ll - old_ll
            if (dll >= 0) or (delta is None):
                update_model = [True]

        tn.distributed.broadcast_object_list(update_model, src=0)
        if update_model[0]:
            mm = mm_next

        # on master, compare the likelihood and write out states
        if world_rank == 0:
            if (delta is not None) and (dll < delta):
                if enc_vdata is not None:
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
                break_cond = [True]

            if (i + 1) % print_iter == 0:
                if enc_vdata is not None:
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

        # master broadcasts to workers if EM is done or continues
        tn.distributed.broadcast_object_list(break_cond, src=0)
        if break_cond[0]:
            break

        old_ll = ll

        # check validation set
        if world_rank == 0:
            if best_vll < vll:
                vflag = [True]

        tn.distributed.broadcast_object_list(vflag, src=0)
        if vflag[0]:
            best_vll = vll
            best_model = mm

    return best_model
