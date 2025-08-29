"""Create, estimate, and sample from an IgnoredDistribution.

Defines the IgnoredDistribution, IgnoredSampler, IgnoredAccumulatorFactory, IgnoredAccumulator, IgnoredEstimator,
and the IgnoredDataEncoder classes for use with dmx-learn.

Ignored distribution is simply a distribution that is ignored in estimation and treated as fixed.

"""
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory, EncodedDataSequence
from numpy.random import RandomState
import numpy as np
from dmx.stats.null_dist import NullDistribution, NullDataEncoder, NullSampler
from typing import Dict, Any, Sequence, TypeVar, Optional, Union

T = TypeVar('T')
E = TypeVar('E')


class IgnoredDistribution(SequenceEncodableProbabilityDistribution):
    """IgnoredDistribution object for using IgnoredDistributions in estimation.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution to be ignored.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Keys for distribution (just a place holder).

    """

    def __init__(self, dist: Optional[SequenceEncodableProbabilityDistribution], name: Optional[str] = None, keys: Optional[str] = None):
        """IgnoredDistribution object.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to be ignored.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Keys for distribution (just a place holder).

        """
        self.dist = dist if dist is not None else NullDistribution()
        self.name = name
        self.keys = keys

    def __str__(self) -> str:

        return 'IgnoredDistribution(%s, name=%s, keys=%s)' % (repr(self.dist), repr(self.name), repr(self.keys))

    def density(self, x: T) -> float:
        """Evaluate the density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            float: Density of attribute 'dist' at x

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T):
        """Evaluate the log-density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            float: log-density of attribute 'dist' at x.

        """
        return self.dist.log_density(x)

    def seq_log_density(self, x: EncodedDataSequence) -> np.ndarray:

        if isinstance(x, IgnoredEncodedDataSequence):
            rv = self.dist.seq_log_density(x.data)
        elif not isinstance(x, IgnoredEncodedDataSequence) and isinstance(x, EncodedDataSequence):
            rv = self.dist.seq_log_density(x)
        else:
            raise Exception("Wrong EncodedDataSequence passed to seq_log_density().")
        
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IgnoredSampler':
        return IgnoredSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IgnoredEstimator':
        return IgnoredEstimator(dist=self.dist, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'IgnoredDataEncoder':
        return IgnoredDataEncoder(encoder=self.dist.dist_to_encoder())


class IgnoredSampler(DistributionSampler):
    """IgnoredSampler object for generating samples from Ignored distribution.

    Attributes:
        dist_sampler (DistributionSampler): DistributionSampler for ignored distribution.
        null_sampler (bool): True if IgnoredDistribution is the NullDistribution.

    """

    def __init__(self, dist: IgnoredDistribution, seed: Optional[int] = None) -> None:
        """IgnoredSampler object.

        Attributes:
            dist (IgnoredDistribution): DistributionSampler for ignored distribution.
            seed (Optional[int]): Set seed for generating random samples.

        """
        self.dist_sampler = dist.dist.sampler(seed)
        self.null_sampler = isinstance(self.dist_sampler, NullSampler)

    def sample(self, size: Optional[int] = None):
        if self.null_sampler:
            if size is None:
                return None
            else:
                return [None]*size
        else:
            return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(SequenceEncodableStatisticAccumulator):
    """IgnoredAccumulator object for aggregating sufficient statistics.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the ignored distribution.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Name for param dists (place holder only).

    """

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder(), name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """IgnoredAccumulator object.

        Args:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for the ignored distribution.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Name for param dists (place holder only).

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name
        self.keys = keys

    def update(self, x: T, weight: float, estimate: Optional[IgnoredDistribution]) -> None:
        pass

    def seq_update(self, x: 'IgnoredEncodedDataSequence', weights: np.ndarray, estimate: Optional[IgnoredDistribution]) -> None:
        pass

    def initialize(self, x: T, weight: float, rng: Optional[RandomState]) -> None:
        pass

    def seq_initialize(self, x: 'IgnoredEncodedDataSequence', weight: np.ndarray, rng: Optional[RandomState]) -> None:
        pass

    def combine(self, suff_stat: Any) -> 'IgnoredAccumulator':
        return self

    def value(self) -> None:
        return None

    def from_value(self, x: Any) -> 'IgnoredAccumulator':
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'IgnoredDataEncoder':
        return IgnoredDataEncoder(encoder=self.encoder)


class IgnoredAccumulatorFactory(StatisticAccumulatorFactory):
    """IgnoredAccumulatorFactory for creating IgnoredAccumulator objects.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for base distribution.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Keys for distribution (just a place holder).

    """

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder(), name: Optional[str] = None, keys: Optional[str] = None):
        """IgnoredAccumulatorFactory object.

        Args:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for base distribution.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Keys for distribution (just a place holder).

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name
        self.keys = keys

    def make(self) -> 'IgnoredAccumulator':
        return IgnoredAccumulator(encoder=self.encoder, name=self.name, keys=self.keys)


class IgnoredEstimator(ParameterEstimator):
    """IgnoredEstimator object for consistency in estimation step.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution to be ignored.
        pseudo_count (Optional[float]): Place holder for consistency.
        suff_stat (Optional[Any]): Place holder for consistency.
        keys (Optional[str]): Place holder for consistency.
        name (Optional[str]): Set name for object instance.

    """

    def __init__(self, dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 pseudo_count: Optional[float] = None, suff_stat: Optional[Any] = None,
                 keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """IgnoredEstimator object.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to be ignored.
            pseudo_count (Optional[float]): Place holder for consistency.
            suff_stat (Optional[Any]): Place holder for consistency.
            keys (Optional[str]): Place holder for consistency.
            name (Optional[str]): Set name for object instance.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("IgnoredEstimator requires keys to be of type 'str'.")

        self.dist = dist if dist is not None else NullDistribution
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self):
        return IgnoredAccumulatorFactory(self.dist.dist_to_encoder(), name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Any) -> IgnoredDistribution:
        return IgnoredDistribution(self.dist, name=self.name)


class IgnoredDataEncoder(DataSequenceEncoder):
    """IgnoredDataEncoder object for encoding sequences of data of ignored distribution.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for ignored distribution.
        null (bool): True if the DataSequenceEncoder is NullDataEncoder.

    """

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        """IgnoredDataEncoder object.

        Attributes:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for ignored distribution.

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.null = isinstance(self.encoder, NullDataEncoder)

    def __str__(self) -> str:
        return 'IgnoredDataEncoder(dist=' + str(self.encoder) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IgnoredDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> 'IgnoredEncodedDataSequence':
        return IgnoredEncodedDataSequence(data=self.encoder.seq_encode(x))

class IgnoredEncodedDataSequence(EncodedDataSequence):
    """IgnoredEncodedDataSequence object for vectorized calls.

    Attributes:
        data (EncodedDataSequence): EncodedDataSequence object for ignored distribution.

    """

    def __init__(self, data: EncodedDataSequence):
        """IgnoredEncodedDataSequence object.

        Args:
            data (EncodedDataSequence): EncodedDataSequence object for ignored distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'IgnoredEncodedDataSequence(data={self.data})'




