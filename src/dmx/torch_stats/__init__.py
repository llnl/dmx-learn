__all__ = [
    "ExponentialDistribution",
    "ExponentialEstimator",
    "GammaDistribution",
    "GammaEstimator",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianEstimator",
    "GaussianDistribution",
    "GaussianEstimator",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalEstimator",
    "PoissonDistribution",
    "PoissonEstimator",
    "BinomialDistribution",
    "BinomialEstimator",
    "GeometricDistribution",
    "GeometricEstimator",
    "IntegerBernoulliSetDistribution",
    "IntegerBernoulliSetEstimator",
    "IntegerMultinomialDistribution",
    "IntegerMultinomialEstimator",
    "JointMixtureDistribution",
    "JointMixtureEstimator",
    "NullDistribution",
    "NullEstimator",
    "CompositeDistribution",
    "ConditionalDistribution",
    "ConditionalDistributionEstimator",
    "MixtureDistribution",
    "MixtureEstimator",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    "SequenceDistribution",
    "SequenceEstimator",
    "HeterogeneousMixtureDistribution",
    "HeterogeneousMixtureEstimator",
    "HiddenMarkovModelDistribution",
    "HiddenMarkovEstimator",
    "IntegerPLSIDistribution",
    "IntegerPLSIEstimator",
    "seq_encode",
    "seq_log_density",
    "seq_log_density_sum",
    "seq_estimate",
    "seq_initialize",
    "seq_encode_mp",
    "seq_log_density_sum_mp",
    "seq_estimate_mp",
    "seq_initialize_mp",
    "TorchProbabilityDistribution",
    "TorchEncodedSequence",
    "TorchSequenceEncoder",
    "TorchParameterEstimator",
    "TorchDevice"
]

import os
from typing import Optional, TypeVar, List, Tuple, Any, Union, Sequence
import numpy as np
import torch as tn
import torch.distributed

# abstract classes
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchEncodedSequence, TorchSequenceEncoder, TorchDevice

# Cont dists
from dmx.torch_stats.exponential import ExponentialDistribution, ExponentialEstimator
from dmx.torch_stats.gamma import GammaDistribution, GammaEstimator
from dmx.torch_stats.dmvn import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from dmx.torch_stats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.torch_stats.mvn import MultivariateGaussianDistribution, MultivariateGaussianEstimator
from dmx.torch_stats.jmixture import JointMixtureDistribution, JointMixtureEstimator

# Discrete
from dmx.torch_stats.intrange import IntegerCategoricalDistribution, IntegerCategoricalEstimator
from dmx.torch_stats.poisson import PoissonDistribution, PoissonEstimator
from dmx.torch_stats.binomial import BinomialDistribution, BinomialEstimator
from dmx.torch_stats.geometric import GeometricDistribution, GeometricEstimator
from dmx.torch_stats.intsetdist import IntegerBernoulliSetDistribution, IntegerBernoulliSetEstimator
from dmx.torch_stats.intmultinomial import IntegerMultinomialDistribution, IntegerMultinomialEstimator

# Msc
from dmx.torch_stats.null_dist import NullDistribution, NullEstimator

# Combinators
from dmx.torch_stats.composite import CompositeDistribution, CompositeEstimator
from dmx.torch_stats.conditional import ConditionalDistribution, ConditionalDistributionEstimator
from dmx.torch_stats.mixture import MixtureDistribution, MixtureEstimator
from dmx.torch_stats.sequence import SequenceDistribution, SequenceEstimator
from dmx.torch_stats.heterogenous_mixture import HeterogeneousMixtureDistribution, HeterogeneousMixtureEstimator
from dmx.torch_stats.int_plsi import IntegerPLSIDistribution, IntegerPLSIEstimator
from dmx.torch_stats.hmm import HiddenMarkovModelDistribution, HiddenMarkovEstimator

import dmx.torch_utils.vector as vec


T = TypeVar('T')

# need to figure out chunking with sequence encoder. Should be returning a DataLoader.
# may need to add this to each class.
def seq_encode_mp(
        world_rank: int,
        world_size: int,
        data: Optional[Sequence[T]],
        encoder: Optional[TorchSequenceEncoder] = None,
        estimator: Optional[TorchParameterEstimator] = None,
        model: Optional[TorchProbabilityDistribution] = None,
        num_chunks: int = 1,
        chunk_size: Optional[int] = None) -> Sequence[Tuple[int, Any]]:

    # set device for GPU if nccl
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # create TorchSequenceEncoder on CPU. Broadcast this object to all workers
    # chunks of data are sent to each worker.
    if world_rank == 0:
        if data is None:
            raise Exception(f'Data cannot be None on device id {world_rank}.')
        if encoder is None:
            if model is not None:
                local_encoder = [model.dist_to_encoder()]
            else:
                local_encoder = [estimator.accumulator_factory().make().acc_to_encoder()]
        else:
            local_encoder = [encoder]

        data_scatter = [[data[i] for i in range(r, len(data), world_size)] for r in range(world_size)]
        # data_scatter = [(len(xx), xx) for xx in data_scatter]
        data_scatter = [xx for xx in data_scatter]

    else:

        local_encoder: List[Optional[TorchSequenceEncoder]] = [None]
        data_scatter = [None]*world_size

    data_loc: List[Optional[Any]] = [None]

    tn.distributed.broadcast_object_list(local_encoder, src=0)
    tn.distributed.scatter_object_list(data_loc, data_scatter, src=0)

    # sequence encode the data on the worker (i.e. the seq encoded data now lives on GPU device)
    # enc_local = []
    # for sz, xx in data_loc:
    #     enc_local.append((sz, local_encoder[0].seq_encode(xx, device=device if device is not None else None)))
    #
    # return enc_local

    return seq_encode(data_loc[0], encoder=local_encoder[0], num_chunks=num_chunks, chunk_size=chunk_size, device=device)


def seq_initialize_mp(
        world_rank: int,
        world_size: int,
        enc_data: Sequence[Tuple[int, Any]],
        estimator: TorchParameterEstimator,
        seed: int,
        p: float = 0.1) -> TorchProbabilityDistribution:

    # set device if nccl backend is detected for GPU use
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # create seeds on master GPU, broadcast the tensors to each worker
    if world_rank == 0:
        tng = tn.Generator(device=device).manual_seed(seed)
        seeds = vec.int_vec((world_size, 2), device=device)
        seeds += tn.randint(0, 2 ** 31, (world_size, 2), generator=tng, device=device, dtype=seeds.dtype)
    else:
        seeds = vec.int_vec((world_size, 2), device=device)

    tn.distributed.broadcast(seeds, src=0)

    # make local accumulator object with tensors on device (estimator was defined on each node)
    local_acc = estimator.accumulator_factory().make(device=device)

    # perform updates on each GPU device
    nobs = 0.0
    local_seeds = seeds[world_rank, :]
    tng_local = tn.Generator(device=device).manual_seed(int(local_seeds[0]))
    tng_w = tn.Generator(device=device).manual_seed(int(local_seeds[1]))

    for sz, enc_x in enc_data:
        w = vec.zeros(sz, device=device)
        u = tn.rand(sz, generator=tng_w, device=device, dtype=w.dtype)
        w[u <= p] += 1.0

        local_acc.seq_initialize(enc_x, w, tng_local)
        nobs += sz

    suff_stats = [None]*world_size
    nobs_list = [None]*world_size

    # gather suff stats, all are on cpu devices (i.e. to tensors)
    tn.distributed.gather_object(local_acc.value(), suff_stats if world_rank == 0 else None, dst=0)
    tn.distributed.gather_object(nobs, nobs_list if world_rank == 0 else None, dst=0)

    # aggregate sufficient statistics on master (to tensor operations).
    if world_rank == 0:

        for ss in suff_stats[1:]:
            local_acc.combine(ss)
        nobs = sum(nobs_list)

        stats_dict = dict()
        local_acc.key_merge(stats_dict)
        local_acc.key_replace(stats_dict)

        agg_ss = [local_acc.value()]
    else:
        agg_ss = [None]

    # broadcast aggregated suff stat to each worker
    tn.distributed.broadcast_object_list(agg_ss, src=0)

    # next model is returned to each device (parameters of model are tensors on worker GPU)
    return estimator.estimate(None, agg_ss[0], device=device)

def seq_estimate_mp(
        world_rank: int,
        world_size: int,
        enc_data: Sequence[Tuple[int, Any]],
        estimator: TorchParameterEstimator,
        prev_estimate: TorchProbabilityDistribution) -> Optional[TorchProbabilityDistribution]:

    # set the device for nccl backend GPUs
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # estimator defined on each device (no tensors), creates accumulator with tensors on device
    local_acc = estimator.accumulator_factory().make(device=device)

    # accumulate sufficient statistics with tensor calcs, result on local node cpu.
    nobs = vec.zeros(1, device=device)

    for sz, enc_x in enc_data:
        w = vec.ones(sz, device=device)
        local_acc.seq_update(enc_x, w, prev_estimate)
        nobs += sz

    # gather sufficient statistics on master (no tensors in suff stats)
    suff_stats = [None]*world_size
    nobs_list = [None]*world_size
    tn.distributed.gather_object(local_acc.value(), suff_stats if world_rank == 0 else None, dst=0)
    tn.distributed.gather_object(nobs, nobs_list if world_rank == 0 else None, dst=0)

    # accumulate sufficient statistics on master
    if world_rank == 0:

        for ss in suff_stats[1:]:
            local_acc.combine(ss)
        nobs = sum(nobs_list)

        stats_dict = dict()
        local_acc.key_merge(stats_dict)
        local_acc.key_replace(stats_dict)

        agg_ss = [local_acc.value()]
    else:
        agg_ss = [None]

    # broadcast aggregated suff stat to each worker
    tn.distributed.broadcast_object_list(agg_ss, src=0)

    # next model is returned to each device (parameters of model are tensors on worker GPU)
    return estimator.estimate(None, agg_ss[0], device=device)

def seq_log_density_sum_mp(world_rank: int, enc_data: Sequence[Tuple[int, Any]], estimate: TorchProbabilityDistribution) \
        -> Optional[Tuple[Optional[float], Optional[float]]]:

    # estimate passed is already defined on each device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # perform local likelihood evaluations on each device (tensor operations)
    nobs_sum = vec.zeros(1, device=device)
    ll_sum = vec.zeros(1, device=device)

    for sz, enc_x in enc_data:
        nobs_sum += sz
        ll_sum += estimate.seq_log_density(enc_x).sum()

    # reduction of all likelihoods to master
    tn.distributed.reduce(ll_sum, dst=0, op=tn.distributed.ReduceOp.SUM)
    tn.distributed.reduce(nobs_sum, dst=0, op=tn.distributed.ReduceOp.SUM)

    # only return likelihood and nobs_sum on CPU for master (optimize only needs 1 check)
    if world_rank == 0:
        return float(nobs_sum), float(ll_sum)
    else:
        return None, None

def seq_encode(data: Sequence[T],
               encoder: Optional[TorchSequenceEncoder] = None,
               estimator: Optional[TorchParameterEstimator] = None,
               model: Optional[TorchProbabilityDistribution] = None,
               device: Optional[TorchDevice] = None,
               num_chunks: int = 1, chunk_size: Optional[int] = None) -> List[Tuple[int, Any]]:

    if encoder is None:
        if model is not None:
            encoder = model.dist_to_encoder()
        elif estimator is not None:
            encoder = estimator.accumulator_factory().make().acc_to_encoder()
        else:
            raise Exception('At least one arg: encoder, estimator, or dist must be passed.')

    device = model.model_device() if model is not None else device

    sz = len(data)
    if chunk_size is not None:
        num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
    else:
        num_chunks_loc = num_chunks

    rv = []
    for i in range(num_chunks_loc):
        data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
        enc_data = encoder.seq_encode(data_loc, device=device)
        rv.append((len(data_loc), enc_data))

    return rv


def seq_log_density_sum(enc_data: List[Tuple[int, T]], estimate: TorchProbabilityDistribution) -> Tuple[float, float]:
    return sum([u[0] for u in enc_data]), float(sum(estimate.seq_log_density(u[1]).sum() for u in enc_data))


def seq_log_density(enc_data: List[Tuple[int, T]],
                    estimate: Union[Sequence[TorchProbabilityDistribution], TorchProbabilityDistribution]) \
        -> List[tn.Tensor]:

    is_list = issubclass(type(estimate), Sequence)

    if is_list:
        return [
            tn.concatenate([ee.seq_log_density(u[1]) for ee in estimate], 0)
            for u in enc_data
        ]
    else:
        return [estimate.seq_log_density(u[1]) for u in enc_data]


def seq_estimate(enc_data: List[Tuple[int, T]], estimator: TorchParameterEstimator, prev_estimate: TorchProbabilityDistribution) -> TorchProbabilityDistribution:

    device = prev_estimate.model_device()
    accumulator = estimator.accumulator_factory().make(device=device)
    nobs = 0.0

    for sz, x in enc_data:
        nobs += sz
        accumulator.seq_update(x, vec.ones(sz, device=device), prev_estimate)

    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)

    return estimator.estimate(None, accumulator.value(), device=device)


def seq_initialize(enc_data: List[Tuple[int, T]],
                   estimator: TorchParameterEstimator,
                   seed: int,
                   p: float = 0.1,
                   device: Optional[TorchDevice] = None) -> TorchProbabilityDistribution:

    accumulator = estimator.accumulator_factory().make(device=device)
    nobs = 0.0
    tng = tn.Generator(device=device).manual_seed(seed)

    for sz, enc_x in enc_data:
        w = vec.zeros(sz, device=device)
        u = tn.rand(sz, generator=tng, dtype=w.dtype, device=device)
        w[u <= p] += 1.0

        accumulator.seq_initialize(enc_x, w, tng)
        nobs += sz

    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)

    return estimator.estimate(nobs, accumulator.value(), device=device)




# def seq_encode_mp(data: Sequence[T], encoder: Optional[TorchSequenceEncoder] = None,
#                   estimator: Optional[TorchParameterEstimator] = None,
#                   model: Optional[TorchProbabilityDistribution] = None) -> Sequence[Tuple[int, Any]]:
#
#     if WORLD_RANK == 0:
#         if encoder is None:
#             if model is not None:
#                 local_encoder = [model.dist_to_encoder()]
#             else:
#                 local_encoder = [estimator.accumulator_factory().make().acc_to_encoder()]
#         else:
#             local_encoder = [encoder]
#
#         data_scatter = [[data[i] for i in range(r, len(data), WORLD_SIZE)] for r in range(WORLD_SIZE)]
#         data_scatter = [(len(xx), xx) for xx in data_scatter]
#
#     else:
#         local_encoder: List[Optional[TorchSequenceEncoder]] = [None]
#         data_scatter = [None]*WORLD_SIZE
#
#     data_loc = [None]
#
#     tn.distributed.broadcast_object_list(local_encoder, src=0)
#     tn.distributed.scatter_object_list(data_loc, data_scatter, src=0)
#
#     enc_local = []
#     for sz, xx in data_loc:
#         enc_local.append((sz, local_encoder[0].seq_encode(xx)))
#
#     return enc_local
#
#
# def seq_initialize_mp(enc_data: List[Tuple[int, T]],
#                    estimator: TorchParameterEstimator,
#                    tng: tn.Generator,
#                    p: float = 0.1) -> Optional[TorchProbabilityDistribution]:
#
#     if WORLD_RANK == 0:
#         local_fac = [estimator.accumulator_factory()]
#         seeds = tn.randint(0, 2 ** 31, (WORLD_SIZE, 2), generator=tng, dtype=tn.int)
#     else:
#         local_fac = [None]
#         seeds = tn.zeros((WORLD_SIZE, 2), dtype=tn.int)
#
#     tn.distributed.broadcast(seeds, src=0)
#     tn.distributed.broadcast_object_list(local_fac, src=0)
#
#     local_acc = local_fac[0].make()
#
#     nobs = 0.0
#     local_seeds = seeds[WORLD_RANK, :]
#     tng_local = tn.Generator().manual_seed(int(local_seeds[0]))
#     tng_w = tn.Generator().manual_seed(int(local_seeds[1]))
#
#     for sz, enc_x in enc_data:
#         u = tn.rand(sz, generator=tng_w, dtype=tn.float64)
#         w = tn.zeros(sz, dtype=tn.float64)
#         w[u <= p] += 1.0
#
#         local_acc.seq_initialize(enc_x, w, tng_local)
#         nobs += sz
#
#     suff_stats = [None]*WORLD_SIZE
#     nobs_list = [None]*WORLD_SIZE
#     tn.distributed.gather_object(local_acc.value(), suff_stats if WORLD_RANK == 0 else None, dst=0)
#     tn.distributed.gather_object(nobs, nobs_list if WORLD_RANK == 0 else None, dst=0)
#
#     if WORLD_RANK == 0:
#
#         for ss in suff_stats[1:]:
#             local_acc.combine(ss)
#         nobs = sum(nobs_list)
#
#         stats_dict = dict()
#         local_acc.key_merge(stats_dict)
#         local_acc.key_replace(stats_dict)
#
#         return estimator.estimate(None, local_acc.value())
#
#     else:
#         return None
#
#
# def seq_estimate_mp(enc_data: List[Tuple[int, T]],
#                    estimator: TorchParameterEstimator,
#                    prev_estimate: TorchProbabilityDistribution) -> Optional[TorchProbabilityDistribution]:
#
#     if WORLD_RANK == 0:
#         local_fac = [estimator.accumulator_factory()]
#         bcast_model = [prev_estimate]
#
#     else:
#         local_fac = [None]
#         bcast_model: List[Optional[TorchProbabilityDistribution]] = [None]
#
#     tn.distributed.broadcast_object_list(bcast_model, src=0)
#     tn.distributed.broadcast_object_list(local_fac, src=0)
#
#     local_acc = local_fac[0].make()
#     nobs = 0.0
#
#     for sz, enc_x in enc_data:
#         w = tn.ones(sz, dtype=tn.float64)
#         local_acc.seq_update(enc_x, w, bcast_model[0])
#         nobs += sz
#
#     suff_stats = [None]*WORLD_SIZE
#     nobs_list = [None]*WORLD_SIZE
#     tn.distributed.gather_object(local_acc.value(), suff_stats if WORLD_RANK == 0 else None, dst=0)
#     tn.distributed.gather_object(nobs, nobs_list if WORLD_RANK == 0 else None, dst=0)
#
#     if WORLD_RANK == 0:
#         for ss in suff_stats:
#             local_acc.combine(ss)
#         nobs = sum(nobs_list)
#         stats_dict = dict()
#         local_acc.key_merge(stats_dict)
#         local_acc.key_replace(stats_dict)
#
#         return estimator.estimate(None, local_acc.value())
#
#     else:
#         return None
#
#
# def seq_log_density_sum_mp(enc_data: List[Tuple[int, T]], estimate: TorchProbabilityDistribution) \
#         -> Optional[Tuple[Optional[float], Optional[float]]]:
#     if WORLD_RANK == 0:
#         bcast_model = [estimate]
#     else:
#         bcast_model: List[Optional[TorchProbabilityDistribution]] = [None]
#
#     tn.distributed.broadcast_object_list(bcast_model, src=0)
#
#     nobs_sum = 0.0
#     ll_sum = 0.0
#     for sz, enc_x in enc_data:
#         nobs_sum += sz
#         ll_sum += bcast_model[0].seq_log_density(enc_x).sum()
#
#     nobs_list = [None]*WORLD_SIZE
#     ll_list = [None]*WORLD_SIZE
#
#     tn.distributed.gather_object(float(nobs_sum), nobs_list if WORLD_RANK == 0 else None, dst=0)
#     tn.distributed.gather_object(float(ll_sum), ll_list if WORLD_RANK == 0 else None, dst=0)
#
#     if WORLD_RANK == 0:
#         return sum(nobs_list), sum(ll_list)
#     else:
#         return None, None
#
#
# def seq_encode(data: Sequence[T],
#                encoder: Optional[TorchSequenceEncoder] = None,
#                estimator: Optional[TorchParameterEstimator] = None,
#                model: Optional[TorchProbabilityDistribution] = None,
#                num_chunks: int = 1, chunk_size: Optional[int] = None) -> List[Tuple[int, Any]]:
#
#     if encoder is None:
#         if model is not None:
#             encoder = model.dist_to_encoder()
#         elif estimator is not None:
#             encoder = estimator.accumulator_factory().make().acc_to_encoder()
#         else:
#             raise Exception('At least one arg: encoder, estimator, or dist must be passed.')
#
#     sz = len(data)
#     if chunk_size is not None:
#         num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
#     else:
#         num_chunks_loc = num_chunks
#
#     rv = []
#     for i in range(num_chunks_loc):
#         data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
#         enc_data = encoder.seq_encode(data_loc)
#         rv.append((len(data_loc), enc_data))
#
#     return rv
#
#
# def seq_log_density_sum(enc_data: List[Tuple[int, T]], estimate: TorchProbabilityDistribution) -> Tuple[float, float]:
#     return sum([u[0] for u in enc_data]), float(sum(estimate.seq_log_density(u[1]).sum()for u in enc_data))
#
#
# def seq_log_density(enc_data: List[Tuple[int, T]],
#                     estimate: Union[Sequence[TorchProbabilityDistribution], TorchProbabilityDistribution]) \
#         -> List[tn.Tensor]:
#
#     is_list = issubclass(type(estimate), Sequence)
#
#     if is_list:
#         return [
#             tn.asarray([ee.seq_log_density(u[1]) for ee in estimate])
#             for u in enc_data
#         ]
#     else:
#         return [estimate.seq_log_density(u[1]) for u in enc_data]
#
#
# def seq_estimate(enc_data: List[Tuple[int, T]], estimator: TorchParameterEstimator, prev_estimate: TorchProbabilityDistribution) -> TorchProbabilityDistribution:
#     accumulator = estimator.accumulator_factory().make()
#     nobs = 0.0
#
#     for sz, x in enc_data:
#         nobs += sz
#         accumulator.seq_update(x, tn.ones(sz, dtype=tn.float64), prev_estimate)
#
#     stats_dict = dict()
#     accumulator.key_merge(stats_dict)
#     accumulator.key_replace(stats_dict)
#
#     return estimator.estimate(None, accumulator.value())
#
#
# def seq_initialize(enc_data: List[Tuple[int, T]], estimator: TorchParameterEstimator, tng: tn.Generator,
#                    p: float = 0.1) -> TorchProbabilityDistribution:
#
#     accumulator = estimator.accumulator_factory().make()
#     nobs = 0.0
#     tng_w = tn.Generator().manual_seed(int(tn.randint(0, 2 ** 31, (1,), generator=tng)[0]))
#
#     for sz, enc_x in enc_data:
#         u = tn.rand(sz, generator=tng_w, dtype=tn.float64)
#         w = tn.zeros(sz, dtype=tn.float64)
#         w[u <= p] += 1.0
#
#         accumulator.seq_initialize(enc_x, w, tng)
#         nobs += sz
#
#     stats_dict = dict()
#     accumulator.key_merge(stats_dict)
#     accumulator.key_replace(stats_dict)
#
#     return estimator.estimate(nobs, accumulator.value())
