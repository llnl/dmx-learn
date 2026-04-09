import os
import sys
from typing import IO, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch as tn
import torch.distributed

from dmx.torch_stats import *
from dmx.torch_stats.pdist import (
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
)
from dmx.torch_utils.vector import (
    float_dtype_for_device,
    resolve_device,
    set_default_float_dtype,
)

T = TypeVar("T")
E0 = TypeVar("E0")


def empirical_kl_divergence(
    dist1: TorchProbabilityDistribution,
    dist2: TorchProbabilityDistribution,
    enc_data: List[Tuple[int, TorchEncodedSequence]],
) -> Tuple[float, float, float]:
    """Computes the empirical KL-divergence between two densities.

    Compute the KL-divergence between dist1 and dist2, for encoded sequence of data. Dists must both have the
    same encodings.

    Args:
        dist1 (TorchProbabilityDistribution): Distribution compatible with enc_data.
        dist2 (TorchProbabilityDistribution): Distribution compatible with enc_data.
        enc_data (List[Tuple[int, TorchEncodedSequence]]): List of Tuple containing chunk size and TorchEncodedSequence.

    Returns:
        Tuple of KL-div estimate, number of 'bad' likelihood values for dist1, 'bad' likelihood values for dist2.

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


def optimize(
    data: Optional[Sequence[T]],
    estimator: TorchParameterEstimator,
    seed: Optional[int] = None,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-9,
    init_estimator: Optional[TorchParameterEstimator] = None,
    init_p: float = 0.1,
    device: Optional[tn.device] = None,
    prev_estimate: Optional[TorchProbabilityDistribution] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[List[Tuple[int, E0]]] = None,
    enc_vdata: Optional[List[Tuple[int, E0]]] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> TorchProbabilityDistribution:
    """Estimation of 'estimator' via EM algorithm for max_its iterations or until
        new_loglikelihood - old_loglikelihood < delta.

    Args:
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator): ParameterEstimator used to specify to-be-estimated distribution for observed
            data.
        seed (Optional[int]): Seed for initializing.
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            |old_loglikelihood - new_loglikelihood| < delta or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        device (Optional[tn.device]): Set the device for tesnor calculations. Else autodection.
        tng (Generator): Set seed for initializing EM algorithm.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[TorchProbabilityDistribution]): Optional model estimate used from prior
            fitting. Must be consistent with estimator.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (IO): IO stream to write out iterations of EM algorithm.
        print_iter (int): Print iterations (i.e. log-likelihood difference) every print_iter-iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        TorchProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm is met.

    """
    device = resolve_device(device)
    set_default_float_dtype(float_dtype_for_device(device))

    if data is None and enc_data is None:
        raise Exception("Optimization called with empty data or enc_data.")

    est = estimator if init_estimator is None else init_estimator

    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        prev_estimate.to(device)
        data_encoder = prev_estimate.dist_to_encoder()

    if enc_data is None:
        enc_data = seq_encode(
            data=data, encoder=data_encoder, num_chunks=num_chunks, device=device
        )

    if prev_estimate is None:
        if init_p <= 0.0:
            p = 0.10
        else:
            p = min(max(init_p, 0.0), 1.0)

        seed = seed if seed is not None else np.random.randint(2**31)
        mm = seq_initialize(
            enc_data=enc_data, estimator=est, seed=seed, p=p, device=device
        )
    else:
        mm = prev_estimate

    _, old_ll = seq_log_density_sum(enc_data=enc_data, estimate=mm)

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode(
            data=vdata, encoder=data_encoder, num_chunks=num_chunks, device=device
        )

    if enc_vdata is not None:
        _, old_vll = seq_log_density_sum(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate(enc_data=enc_data, estimator=estimator, prev_estimate=mm)
        cnt, ll = seq_log_density_sum(enc_data=enc_data, estimate=mm_next)

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
                    "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, "
                    "ln[p_mat(Valid Data|Model)]=%e\n" % (i + 1, ll, dll, vll)
                )
            else:
                out.write(
                    "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n"
                    % (i + 1, ll, dll)
                )
            break

        if (i + 1) % print_iter == 0:
            if enc_vdata is not None:
                out.write(
                    "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, "
                    "ln[p_mat(Valid Data|Model)]=%e\n" % (i + 1, ll, dll, vll)
                )
            else:
                out.write(
                    "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n"
                    % (i + 1, ll, dll)
                )

        old_ll = ll

        if best_vll < vll:
            best_vll = vll
            best_model = mm

    return best_model


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
    enc_data: Optional[List[Tuple[int, E0]]] = None,
    enc_vdata: Optional[List[Tuple[int, E0]]] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> TorchProbabilityDistribution:
    """Estimation of 'estimator' via EM algorithm for max_its iterations or until
        new_loglikelihood - old_loglikelihood < delta.

    Args:
        world_rank (int): Rank of worker
        world_size (int): Total number of GPUs.
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator): ParameterEstimator used to specify to-be-estimated distribution for observed
            data.
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            |old_loglikelihood - new_loglikelihood| < delta or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        seed (Optional[int]): Set seed for initializing EM algorithm.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[TorchProbabilityDistribution]): Optional model estimate used from prior
            fitting. Must be consistent with estimator.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (IO): IO stream to write out iterations of EM algorithm.
        print_iter (int): Print iterations (i.e. log-likelihood difference) every print_iter-iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    # data on all nodes assumed for now. Can change this later.
    if data is None and enc_data is None:
        raise Exception("Optimization called with empty data or enc_data.")

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
    _, old_ll = seq_log_density_sum_mp(
        world_rank=world_rank, enc_data=enc_data, estimate=mm
    )

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode_mp(
            world_rank=world_rank,
            world_size=world_size,
            data=vdata,
            encoder=data_encoder,
        )

    if enc_vdata is not None:
        _, old_vll = seq_log_density_sum_mp(
            world_rank=world_rank, enc_data=enc_vdata, estimate=mm
        )
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):
        # condition for stopping EM, updating model, and validation data
        break_cond = [False]
        update_model = [False]
        vflag = [False]

        mm_next = seq_estimate_mp(
            world_rank=world_rank,
            world_size=world_size,
            enc_data=enc_data,
            estimator=est,
            prev_estimate=mm,
        )
        cnt, ll = seq_log_density_sum_mp(
            world_rank=world_rank, enc_data=enc_data, estimate=mm_next
        )

        if enc_vdata is not None:
            _, vll = seq_log_density_sum_mp(
                world_rank=world_rank, enc_data=enc_vdata, estimate=mm_next
            )
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
                        "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, "
                        "ln[p_mat(Valid Data|Model)]=%e\n" % (i + 1, ll, dll, vll)
                    )
                else:
                    out.write(
                        "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n"
                        % (i + 1, ll, dll)
                    )
                break_cond = [True]

            if (i + 1) % print_iter == 0:
                if enc_vdata is not None:
                    out.write(
                        "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, "
                        "ln[p_mat(Valid Data|Model)]=%e\n" % (i + 1, ll, dll, vll)
                    )
                else:
                    out.write(
                        "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n"
                        % (i + 1, ll, dll)
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
