"""Create, estimate, and sample from a null distribution.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory, NullAccumulator,
NullEstimator, and the NullDataEncoder classes for use with dmx-learn.

The NullDistribution object and its related classes are space filling objects meant for consistency in type hints.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    The sampler generates None for any size input.
    Sequence encodings return None for any input.

"""
from typing import Any, Optional, Dict

import numpy as np
from numpy.random import RandomState
import dmx.utils.vector as vec
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence


class NullDistribution(SequenceEncodableProbabilityDistribution):
    """NullDistribution object serves as placeholder for type checks.

    Attributes:
        name (Optional[str]): Name for object.

    """

    def __init__(self, name: Optional[str] = None) -> None:
        """NullDistribution object.

        Args:
            name (Optional[str]): Name for object.

        """
        self.name = name

    def __str__(self) -> str:
        return 'NullDistribution(name=%s)' % repr(self.name)

    def density(self, x: Optional[Any]) -> float:
        """Density for NullDistribution.

        Args:
            x (Optional[Any]): Can pass any value.

        Returns:
            float: Always evaluates to 1.0.

        """
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        """Log-density for NullDistribution.

        Args:
            x (Optional[Any]): Can pass any value.

        Returns:
            float: Always evaluates to 0.0.

        """
        return 0.0

    def seq_log_density(self, x: 'NullEncodedDataSequence') -> np.ndarray:
        return vec.zeros(1)

    def sampler(self, seed: Optional[int] = None) -> 'NullSampler':
        return NullSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'NullEstimator':
        if pseudo_count is None:
            return NullEstimator(name=self.name)

        else:
            return NullEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


class NullSampler(DistributionSampler):
    """NullSampler object, always generates None as sample type.

    Note:
        This generally serves as a place-holder for consistency with other classes. Try to remove it before sampling.

    Attributes:
        rng (RandomState): For consistency with other samplers.
        dist (NullDistribution): For consistency with other samplers.

    """

    def __init__(self, dist: 'NullDistribution', seed: Optional[int] = None) -> None:
        """NullSampler object.

        Args:
            seed (Optional[int]): For consistency with other samplers.
            dist (NullDistribution): For consistency with other samplers.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> None:
        """Generate samples from NullDistribution.

        Notes:
            Always returns None regardless of size.

        Args:
            size (Optional[int]): For consistency, does not control number of samples.

        Returns:
            None
        """
        return None


class NullAccumulator(SequenceEncodableStatisticAccumulator):
    """NullAccumulator object for accumulating sufficient statistics.

    Notes:
        All functions do nothing. They are kept for consistency with other classes to ensure type checks.

    Attributes:
        keys (Optional[str]): Set key for distribution.


    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """NullAccumulator object.

        Args:
            keys (Optional[str]): Set key for distribution.


        """
        self.key = keys

    def update(self, x: Optional[Any], weight: float, estimate: Optional['NullDistribution']) -> None:
        pass

    def seq_update(self,
                   x: 'NullEncodedDataSequence',
                   weights: np.ndarray,
                   estimate: Optional['NullDistribution']) -> None:
        pass

    def initialize(self, x: Optional[Any], weight: float, rng: Optional['np.random.RandomState']) -> None:
        self.update(x, weight, None)

    def seq_initialize(self,
                       x: 'NullEncodedDataSequence',
                       weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Optional[Any]) -> 'NullAccumulator':
        return self

    def value(self) -> None:
        return None

    def from_value(self, x: Optional[Any]) -> 'NullAccumulator':
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory):
    """NullAccumulatorFactory object for creating NullAccumulator objects.

    Notes:
        All functions do nothing. They are kept for consistency with other classes to ensure type checks.

    Attributes:
        keys (Optional[str]): Set key for distribution.


    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """NullAccumulatorFactory object.

        Args:
            keys (Optional[str]): Set key for distribution.

        """
        self.keys = keys

    def make(self) -> 'NullAccumulator':
        return NullAccumulator(keys=self.keys)


class NullEstimator(ParameterEstimator):
    """NullEstimator object for estimating NullDistribution.

    Notes:
        Always estimates to same NullDistribution object. This is simply a placeholder.

    Attributes:
        pseudo_count (Optional[float]): Regularize sufficient statistics (ignored).
        suff_stat (Optional[Any]): Can pass anything, is simply ignored.
        keys (Optional[str]): Key for distribution (not meaningful as all estimates are NullDistribution())
        name (Optional[str]): Name for estimator.


    """

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Any] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """NullEstimator object.

        Args:
            pseudo_count (Optional[float]): Regularize sufficient statistics (ignored).
            suff_stat (Optional[Any]): Can pass anything, is simply ignored.
            keys (Optional[str]): Key for distribution (not meaningful as all estimates are NullDistribution())
            name (Optional[str]): Name for estimator.


        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("NullEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'NullAccumulatorFactory':
        return NullAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Any] = None) -> 'NullDistribution':
        return NullDistribution(name=self.name)


class NullDataEncoder(DataSequenceEncoder):
    """NullDataEncoder object for consistency with DataSequenceEncoders.

    Notes:
        This enables consistency in type-hints and type-checks for other encodings.


    """

    def __str__(self) -> str:
        return 'NullDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Any) -> 'NullEncodedDataSequence':
        return NullEncodedDataSequence(data=None)

class NullEncodedDataSequence(EncodedDataSequence):
    """NullEncodedDataSequence object for vectorized calls.

    Notes:
        This enables consistency in type-hints and type-checks for other encodings.

    Attributes:
        data (None): None is passed as placeholder.

    """
    def __init__(self, data: None):
        """NullEncodedDataSequence object..

        Args:
            data (None): None is passed as placeholder.

        """
        super().__init__(data=data)
        
    def __repr__(self) -> str:
        return 'NullEncodedDataSequence(data=None}'
    
    