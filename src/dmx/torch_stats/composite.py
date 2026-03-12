"""Create, estimate, and sample from a Composite distribution.

Defines the CompositeDistribution, CompositeSampler, CompositeAccumulatorFactory, CompositeAccumulator,
CompositeEstimator, and the CompositeDataEncoder classes for use with pysparkplug.

Data type: (Tuple[T_0, ... T_{n-1}]): The CompositeDistribution of size 'n' is a joint distribution for
independent observations of 'n'-tupled data. Each component 'k' of the CompositeDistribution has data type T_k that
must be compatible with data type T_k.

"""
import torch as tn
from dmx.torch_stats.null_dist import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence

from dmx.arithmetic import maxrandint
from torch import Generator
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict

E = TypeVar('E')
SS = TypeVar('SS')


class CompositeDistribution(TorchProbabilityDistribution):
    """CompositeDistribution for modeling independent distributions of from (Dist_0,Dist_1,...,Dist_{n-1}).

    Notes:
        Data type must be (T_0, T_1, ..., T_{n-1}), where data type T_k is consistent with distribution Dist_k. The
        density for a single observation tuple x = (x_0,x_1,...,x_{n-1}) is given by,

        p_mat(x) = p_mat(x_0 | Dist_0)*p_mat(x_1 | Dist_1)*...*p_mat(x_{n-1} | Dist_{n-1}).


    Attributes:
        dists: (Sequence[TorchProbabilityDistribution]): Distributions given by Dist_k above.
        counts (int): Number of components (i.e. len(dists)).

    """

    def __init__(self,
                 dists: Sequence[TorchProbabilityDistribution],
                 device: Optional[TorchDevice] = None) -> None:
        """CompositeDistribution object.

        Args:
            dists (Sequence[TorchProbabilityDistribution]): Distributions given by Dist_k above.
            device (Optional[str]): Set the device type for object.

        """
        super().__init__(device)
        self.dists = dists
        self.count = len(dists)

    def to(self, device: TorchDevice) -> None:
        self._device = device
        for comp in self.dists:
            comp.to(device)

    def __repr__(self) -> str:
        s0 = ','.join(map(str, self.dists))
        return 'CompositeDistribution((%s))' % s0

    def density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates density of CompositeDistribution for single observation tuple x.

        Notes:
            p_mat(x) = p_mat(x_0 | dist_0)*p_mat(x_1 | dist_1)*...*p_mat(x_{n-1} | dist_{n-1}),

            where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            float: Density as float.

        """
        rv = 0.0

        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])

        return rv

    def log_density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates log-density of CompositeDistribution for single observation tuple x.

        Notes:
            log(p_mat(x)) = log(p_mat(x_0 | dist_0)) + log(p_mat(x_1 | dist_1)) + ... + log(p_mat(x_{n-1} | dist_{n-1})),

            where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            float: Log-density as float.

        """
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def seq_log_density(self, x: 'CompositeTorchEncodedSequence') -> tn.Tensor:
        if not isinstance(x, CompositeTorchEncodedSequence):
            raise Exception('Requires CompositeTorchEncodedSequence for `seq_` calls.')

        rv = self.dists[0].seq_log_density(x.data[0])

        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x.data[i])

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'CompositeSampler':
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'CompositeEstimator':
        return CompositeEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists])

    def dist_to_encoder(self) -> 'CompositeDataEncoder':
        encoders = tuple([d.dist_to_encoder() for d in self.dists])

        return CompositeDataEncoder(encoders=encoders)


class CompositeSampler(DistributionSampler):
    """CompositeSampler used to generate samples from CompositeDistribution.

    Attributes:
        dist (CompositeDistribution): CompositeDistribution to draw samples from.
        rng (RandomState): RandomState with seed set if provided.
        dist_samplers (List[DistributionSamplers]): List of DistributionSamplers for each component
            (len=len(dists)).
    """

    def __init__(self, dist: 'CompositeDistribution', seed: Optional[int] = None) -> None:
        """CompositeSampler object.

        Args:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists]

    def sample(self, size: Optional[int] = None) -> Union[List[Tuple[Any, ...]], Tuple[Any, ...]]:
        """Generate independent samples from a CompositeDistribution.

        If size is None, draw one sample and return as Tuple of length = len(dists). If size > 0,
        draw size samples and return a list of length size containing tuples of len(dists).

        Args:
            size (Optional[int]): If None, draw 1 sample. Else, draw size number of iid samples.

        Returns:
            A tuple of length = len(dists) or a list of length size containing tuples of length = len(dists).

        """
        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])

        else:
            return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(TorchStatisticAccumulator):
    """CompositeAccumulator object used for aggregating suffcient statistics of each component of the
        CompositeDistribution.

    Attributes:
        accumulators (List[TorchStatisticAccumulator]): List of TorchStatisticAccumulator
            objects for accumulating sufficient statsitics for each component of the CompositeDistribution.
        count (int): Length of accumulators.
        keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
        _init_tng (bool): Is True if _acc_tng has been set by a single function call to initialize.
        _acc_tng (List[Generator]): List of Generator objects generated from seeds set by tng in initialize.

    """

    def __init__(self, accumulators: Sequence[TorchStatisticAccumulator], keys: Optional[str] = None, 
                 device: Optional[TorchDevice] = None) -> None:
        """CompositeAccumulator object.

        Args:
            accumulators (List[TorchStatisticAccumulator]):
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
            device (Optional[str]): Set the device type for object.

        """
        super().__init__(device)
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.key = keys

    def seq_initialize(self, x: 'CompositeTorchEncodedSequence', weights: tn.Tensor, tng: Generator) -> None:

        for i in range(0, self.count):
            self.accumulators[i].seq_initialize(x.data[i], weights, tng)

    def seq_update(self, x: 'CompositeTorchEncodedSequence', weights: tn.Tensor,
                   estimate: Optional['CompositeDistribution']) -> None:
        for i in range(self.count):
            self.accumulators[i].seq_update(x.data[i], weights, estimate.dists[i] if estimate is not None else None)

    def combine(self, suff_stat: SS) -> 'CompositeAccumulator':
        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])

        return self

    def value(self) -> Tuple[Any, ...]:
        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x: SS) -> 'CompositeAccumulator':
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'CompositeDataEncoder':
        encoders = tuple([acc.acc_to_encoder() for acc in self.accumulators])

        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """CompositeAccumulatorFactory used for lightweight creation of CompositeAccumulator.

    Attributes:
        factories (List[TorchStatisticAccumulatorFactory]): List of TorchStatisticAccumulatorFactory objects for
            each component.
        keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.

    """

    def __init__(self, factories: Sequence[TorchStatisticAccumulatorFactory], keys: Optional[str] = None) -> None:
        """CompositeAccumulatorFactory object.

        Attributes:
            factories (List[TorchStatisticAccumulatorFactory]): List of TorchStatisticAccumulatorFactory objects for 
                each component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.
            
        """
        self.factories = factories
        self.keys = keys

    def make(self, device: Optional[TorchDevice] = None) -> 'CompositeAccumulator':
        return CompositeAccumulator([u.make() for u in self.factories], keys=self.keys, device=device)


class CompositeEstimator(TorchParameterEstimator):
    """CompositeEstimator object used to estimate CompositeDistribution from sufficient statistics of each
        component.

    Attributes:
        estimators (List[TorchParameterEstimator]): List of TorchParameterEstimator objects for each component of
            CompositeEstimator.
        keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
        count (int): Number of components in CompositeEstimator.

    """

    def __init__(self, estimators: Sequence[TorchParameterEstimator], keys: Optional[str] = None) -> None:
        """CompositeEstimator object.

        Args:
            estimators (List[TorchParameterEstimator]): List of TorchParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.

        """
        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys

    def accumulator_factory(self) -> 'CompositeAccumulatorFactory':
        """Creates CompositeAccumulatorFactory from each TorchParameterEstimator in estimators.

        Returns:
            CompositeAccumulatorFactory.

        """
        return CompositeAccumulatorFactory([u.accumulator_factory() for u in self.estimators], self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: SS, device: Optional[TorchDevice]= None) -> 'CompositeDistribution':
        """Estimate a CompositeDistribution from an aggregated sufficient statistics Tuple for a given number of
            observations (nobs).

        Args:
            nobs (Optional[float]): Weighted number of observations used to form suff_stat.
            suff_stat (SS): Tuple of sufficient statistics for each TorchParameterEstimator of estimators.
            device (Optional[TorchDevice]): Device to declare new estimate on.

        Returns:
            CompositeDistribution estimated from argument aggregated sufficient statistics (suff_stat), from a given
                number of observation (nobs).

        """
        return CompositeDistribution(tuple([est.estimate(nobs, ss, device=device) for est, ss in zip(self.estimators, suff_stat)]),
                                     device=device)


class CompositeDataEncoder(TorchSequenceEncoder):
    """CompositeDataEncoder used for encoding data.

    Attributes:
        encoders (Sequence[TorchSequenceEncoder]): TorchSequenceEncoders for each component of the
            CompositeDistribution.

    """

    def __init__(self, encoders: Sequence[TorchSequenceEncoder]) -> None:
        """CompositeDataEncoder object.

        Args:
            encoders (Sequence[TorchSequenceEncoder]): TorchSequenceEncoders for each component of the
                CompositeDistribution.

        """
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeDataEncoder):
            return False

        else:

            for i, encoder in enumerate(self.encoders):
                if not encoder == other.encoders[i]:
                    return False

        return True

    def __str__(self) -> str:

        s = 'CompositeDataEncoder(['

        for d in self.encoders[:-1]:
            s += str(d) + ','

        s += str(self.encoders[-1]) + '])'

        return s

    def seq_encode(self, x: Sequence[Tuple[Any, ...]], device: Optional[TorchDevice] = None) -> 'CompositeTorchEncodedSequence':
        """Encode Sequence of tuples of data for use with vectorized "seq_" functions.

        The input x must be a Sequence of Tuples of length equal to the length of encoders. Each component tuple
        observation of x, say x[i], must be component-wise compatible with encoders.

        Args:
            x (Sequence[Tuple[Any, ...]]): Sequence of tuples of length equal to len(encoders).
            device (Optional[TorchDevice]): Set device for tensors.

        Returns:
            CompositeTorchEncodedSequence

        """
        enc_data = []

        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x], device=device))

        return CompositeTorchEncodedSequence(data=tuple(enc_data), device=device)


class CompositeTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[TorchEncodedSequence, ...], device: Optional[TorchDevice] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'CompositeTorchEncodedSequence(device={repr(self.device)})'

