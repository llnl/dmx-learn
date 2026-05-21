# pylint: disable=too-many-positional-arguments,duplicate-code

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
    "CompositeEstimator",
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
    "TorchDevice",
]

_TORCH_IMPORT_ERROR = ""

# Check if torch is available
try:
    import torch as tn
    import torch.distributed

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = """
PyTorch is required to use dmx.torch_stats but is not installed.

To install PyTorch with dmx-learn:
    poetry install --with torch

Or install PyTorch separately:
    pip install torch

For more information, see the README.md file.
"""

# If torch is not available, raise a helpful error when trying to import
if not TORCH_AVAILABLE:

    def _raise_torch_error(*args, **kwargs):
        raise ImportError(_TORCH_IMPORT_ERROR)

    # Create placeholder for all exports
    for name in __all__:
        globals()[name] = _raise_torch_error

else:
    # Torch is available, proceed with normal imports
    from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union

    import numpy as np

    import dmx.torch_utils.vector as vec
    from dmx.torch_stats.binomial import BinomialDistribution, BinomialEstimator

    # Combinators
    from dmx.torch_stats.composite import CompositeDistribution, CompositeEstimator
    from dmx.torch_stats.conditional import (
        ConditionalDistribution,
        ConditionalDistributionEstimator,
    )
    from dmx.torch_stats.dmvn import (
        DiagonalGaussianDistribution,
        DiagonalGaussianEstimator,
    )

    # Cont dists
    from dmx.torch_stats.exponential import (
        ExponentialDistribution,
        ExponentialEstimator,
    )
    from dmx.torch_stats.gamma import GammaDistribution, GammaEstimator
    from dmx.torch_stats.gaussian import GaussianDistribution, GaussianEstimator
    from dmx.torch_stats.geometric import GeometricDistribution, GeometricEstimator
    from dmx.torch_stats.heterogenous_mixture import (
        HeterogeneousMixtureDistribution,
        HeterogeneousMixtureEstimator,
    )
    from dmx.torch_stats.hmm import HiddenMarkovEstimator, HiddenMarkovModelDistribution
    from dmx.torch_stats.int_plsi import IntegerPLSIDistribution, IntegerPLSIEstimator
    from dmx.torch_stats.intmultinomial import (
        IntegerMultinomialDistribution,
        IntegerMultinomialEstimator,
    )

    # Discrete
    from dmx.torch_stats.intrange import (
        IntegerCategoricalDistribution,
        IntegerCategoricalEstimator,
    )
    from dmx.torch_stats.intsetdist import (
        IntegerBernoulliSetDistribution,
        IntegerBernoulliSetEstimator,
    )
    from dmx.torch_stats.jmixture import JointMixtureDistribution, JointMixtureEstimator
    from dmx.torch_stats.mixture import MixtureDistribution, MixtureEstimator
    from dmx.torch_stats.mvn import (
        MultivariateGaussianDistribution,
        MultivariateGaussianEstimator,
    )

    # Msc
    from dmx.torch_stats.null_dist import NullDistribution, NullEstimator

    # abstract classes
    from dmx.torch_stats.pdist import (
        TorchDevice,
        TorchEncodedSequence,
        TorchParameterEstimator,
        TorchProbabilityDistribution,
        TorchSequenceEncoder,
    )
    from dmx.torch_stats.poisson import PoissonDistribution, PoissonEstimator
    from dmx.torch_stats.sequence import SequenceDistribution, SequenceEstimator

    T = TypeVar("T")

    # Need to figure out chunking with sequence encoder.
    # Should be returning a DataLoader.
    # may need to add this to each class.
    def seq_encode_mp(
        world_rank: int,
        world_size: int,
        data: Optional[Sequence[T]],
        encoder: Optional[TorchSequenceEncoder] = None,
        estimator: Optional[TorchParameterEstimator] = None,
        model: Optional[TorchProbabilityDistribution] = None,
        num_chunks: int = 1,
        chunk_size: Optional[int] = None,
    ) -> Sequence[Tuple[int, Any]]:

        # set device for GPU if nccl
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create TorchSequenceEncoder on CPU. Broadcast this object to all workers
        # chunks of data are sent to each worker.
        if world_rank == 0:
            if data is None:
                raise ValueError(f"Data cannot be None on device id {world_rank}.")
            if encoder is None:
                if model is not None:
                    local_encoder = [model.dist_to_encoder()]
                else:
                    local_encoder = [
                        estimator.accumulator_factory().make().acc_to_encoder()
                    ]
            else:
                local_encoder = [encoder]

            data_scatter = [
                [data[i] for i in range(r, len(data), world_size)]
                for r in range(world_size)
            ]
            # data_scatter = [(len(xx), xx) for xx in data_scatter]

        else:

            local_encoder: List[Optional[TorchSequenceEncoder]] = [None]
            data_scatter = [None] * world_size

        data_loc: List[Optional[Any]] = [None]

        tn.distributed.broadcast_object_list(local_encoder, src=0)
        tn.distributed.scatter_object_list(data_loc, data_scatter, src=0)

        # Sequence-encode the data on the worker so the encoded data now
        # lives on the GPU device.
        # enc_local = []
        # for sz, xx in data_loc:
        #     enc_local.append(
        #         (sz, local_encoder[0].seq_encode(xx, device=device))
        #     )
        #
        # return enc_local

        return seq_encode(
            data_loc[0],
            encoder=local_encoder[0],
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            device=device,
        )

    def seq_initialize_mp(
        world_rank: int,
        world_size: int,
        enc_data: Sequence[Tuple[int, Any]],
        estimator: TorchParameterEstimator,
        seed: int,
        p: float = 0.1,
    ) -> TorchProbabilityDistribution:

        # set device if nccl backend is detected for GPU use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create seeds on master GPU, broadcast the tensors to each worker
        if world_rank == 0:
            tng = tn.Generator(device=device).manual_seed(seed)
            seeds = vec.int_vec((world_size, 2), device=device)
            seeds += tn.randint(
                0,
                2**31,
                (world_size, 2),
                generator=tng,
                device=device,
                dtype=seeds.dtype,
            )
        else:
            seeds = vec.int_vec((world_size, 2), device=device)

        tn.distributed.broadcast(seeds, src=0)

        # Make a local accumulator with tensors on device.
        # The estimator was defined on each node.
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

        suff_stats = [None] * world_size
        nobs_list = [None] * world_size

        # gather suff stats, all are on cpu devices (i.e. to tensors)
        tn.distributed.gather_object(
            local_acc.value(), suff_stats if world_rank == 0 else None, dst=0
        )
        tn.distributed.gather_object(
            nobs, nobs_list if world_rank == 0 else None, dst=0
        )

        # aggregate sufficient statistics on master (to tensor operations).
        if world_rank == 0:

            for ss in suff_stats[1:]:
                local_acc.combine(ss)
            nobs = sum(nobs_list)

            stats_dict = {}
            local_acc.key_merge(stats_dict)
            local_acc.key_replace(stats_dict)

            agg_ss = [local_acc.value()]
        else:
            agg_ss = [None]

        # broadcast aggregated suff stat to each worker
        tn.distributed.broadcast_object_list(agg_ss, src=0)

        # Return the next model to each device with parameters on worker GPU.
        return estimator.estimate(None, agg_ss[0], device=device)

    def seq_estimate_mp(
        world_rank: int,
        world_size: int,
        enc_data: Sequence[Tuple[int, Any]],
        estimator: TorchParameterEstimator,
        prev_estimate: TorchProbabilityDistribution,
    ) -> Optional[TorchProbabilityDistribution]:

        # set the device for nccl backend GPUs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # The estimator is defined on each device and creates an accumulator
        # with tensors on device.
        local_acc = estimator.accumulator_factory().make(device=device)

        # accumulate sufficient statistics with tensor calcs, result on local node cpu.
        nobs = vec.zeros(1, device=device)

        for sz, enc_x in enc_data:
            w = vec.ones(sz, device=device)
            local_acc.seq_update(enc_x, w, prev_estimate)
            nobs += sz

        # gather sufficient statistics on master (no tensors in suff stats)
        suff_stats = [None] * world_size
        nobs_list = [None] * world_size
        tn.distributed.gather_object(
            local_acc.value(), suff_stats if world_rank == 0 else None, dst=0
        )
        tn.distributed.gather_object(
            nobs, nobs_list if world_rank == 0 else None, dst=0
        )

        # accumulate sufficient statistics on master
        if world_rank == 0:

            for ss in suff_stats[1:]:
                local_acc.combine(ss)
            nobs = sum(nobs_list)

            stats_dict = {}
            local_acc.key_merge(stats_dict)
            local_acc.key_replace(stats_dict)

            agg_ss = [local_acc.value()]
        else:
            agg_ss = [None]

        # broadcast aggregated suff stat to each worker
        tn.distributed.broadcast_object_list(agg_ss, src=0)

        # Return the next model to each device with parameters on worker GPU.
        return estimator.estimate(None, agg_ss[0], device=device)

    def seq_log_density_sum_mp(
        world_rank: int,
        enc_data: Sequence[Tuple[int, Any]],
        estimate: TorchProbabilityDistribution,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:

        # estimate passed is already defined on each device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # perform local likelihood evaluations on each device (tensor operations)
        nobs_sum = vec.zeros(1, device=device)
        ll_sum = vec.zeros(1, device=device)

        for sz, enc_x in enc_data:
            nobs_sum += sz
            ll_sum += estimate.seq_log_density(enc_x).sum()

        # reduction of all likelihoods to master
        tn.distributed.reduce(ll_sum, dst=0, op=tn.distributed.ReduceOp.SUM)
        tn.distributed.reduce(nobs_sum, dst=0, op=tn.distributed.ReduceOp.SUM)

        # Only return likelihood and nobs_sum on CPU for master.
        # Optimize only needs one check.
        if world_rank == 0:
            return float(nobs_sum), float(ll_sum)

        return None, None

    def seq_encode(
        data: Sequence[T],
        encoder: Optional[TorchSequenceEncoder] = None,
        estimator: Optional[TorchParameterEstimator] = None,
        model: Optional[TorchProbabilityDistribution] = None,
        device: Optional[TorchDevice] = None,
        num_chunks: int = 1,
        chunk_size: Optional[int] = None,
    ) -> List[Tuple[int, Any]]:

        if encoder is None:
            if model is not None:
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
            else:
                raise ValueError(
                    "At least one arg: encoder, estimator, or dist must be passed."
                )

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

    def seq_log_density_sum(
        enc_data: List[Tuple[int, T]], estimate: TorchProbabilityDistribution
    ) -> Tuple[float, float]:
        return sum(u[0] for u in enc_data), float(
            sum(estimate.seq_log_density(u[1]).sum() for u in enc_data)
        )

    def seq_log_density(
        enc_data: List[Tuple[int, T]],
        estimate: Union[
            Sequence[TorchProbabilityDistribution], TorchProbabilityDistribution
        ],
    ) -> List[tn.Tensor]:

        is_list = issubclass(type(estimate), Sequence)

        if is_list:
            return [
                tn.concatenate([ee.seq_log_density(u[1]) for ee in estimate], 0)
                for u in enc_data
            ]

        return [estimate.seq_log_density(u[1]) for u in enc_data]

    def seq_estimate(
        enc_data: List[Tuple[int, T]],
        estimator: TorchParameterEstimator,
        prev_estimate: TorchProbabilityDistribution,
    ) -> TorchProbabilityDistribution:

        device = prev_estimate.model_device()
        accumulator = estimator.accumulator_factory().make(device=device)
        nobs = 0.0

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, vec.ones(sz, device=device), prev_estimate)

        stats_dict = {}
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(None, accumulator.value(), device=device)

    def seq_initialize(
        enc_data: List[Tuple[int, T]],
        estimator: TorchParameterEstimator,
        seed: int,
        p: float = 0.1,
        device: Optional[TorchDevice] = None,
    ) -> TorchProbabilityDistribution:

        accumulator = estimator.accumulator_factory().make(device=device)
        nobs = 0.0
        tng = tn.Generator(device=device).manual_seed(seed)

        for sz, enc_x in enc_data:
            w = vec.zeros(sz, device=device)
            u = tn.rand(sz, generator=tng, dtype=w.dtype, device=device)
            w[u <= p] += 1.0

            accumulator.seq_initialize(enc_x, w, tng)
            nobs += sz

        stats_dict = {}
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value(), device=device)
