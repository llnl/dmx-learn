"""Create, estimate, and sample from a geometric distribution with probability of success p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulatorFactory, GeometricAccumulator,
GeometricEstimator, and the GeometricDataEncoder classes for use with dmx-learn.

Data type (int): The geometric distribution with probability of success p, has density

    P(x=k) = (k-1)*log(1-p) + log(p), for k = 1,2,...

"""
import numpy as np
from dmx.arithmetic import *
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    EncodedDataSequence,
)
from numpy.random import RandomState
from typing import Optional, Tuple, Sequence, Dict, Union, Any


class GeometricDistribution(SequenceEncodableProbabilityDistribution):
    """Geometric distribution with probability of success p.

    Attributes:
        p (float): Probability of success, must be between (0,1).
        log_p (float): Log of probability of success p.
        log_1p (float): Log of 1-p (probability of failure).
        name (Optional[str]): Name for the GeometricDistribution object.
        keys (Optional[str]): Key for parameter p.
    """

    def __init__(self, p: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize GeometricDistribution.

        Args:
            p (float): Probability of success, must be between (0,1).
            name (Optional[str], optional): Name for the GeometricDistribution object.
            keys (Optional[str], optional): Key for parameter p.
        """
        self.p = max(0.0, min(p, 1.0))
        self.log_p = np.log(self.p)
        self.log_1p = np.log1p(-self.p)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        return f'GeometricDistribution({repr(self.p)}, name={repr(self.name)}, keys={repr(self.keys)})'

    def density(self, x: int) -> float:
        """Evaluate the density of the geometric distribution at x.

        .. math::
            P(x=k) = (k-1)\\log(1-p) + \\log(p), \\quad x = 1,2,...

        Args:
            x (int): Observed geometric value (1,2,3,...).

        Returns:
            float: Density of geometric distribution evaluated at x.
        """
        return exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the geometric distribution at x.

        Args:
            x (int): Must be a natural number (1,2,3,...).

        Returns:
            float: Log-density of geometric distribution evaluated at x.
        """
        return (x - 1) * self.log_1p + self.log_p

    def seq_log_density(self, x: 'GeometricEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (GeometricEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, GeometricEncodedDataSequence):
            raise Exception("GeometricEncodedDataSequence required for seq_log_density().")

        rv = x.data - 1
        rv *= self.log_1p
        rv += self.log_p

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GeometricSampler':
        """Return a GeometricSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            GeometricSampler: Sampler object.
        """
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GeometricEstimator':
        """Return a GeometricEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            GeometricEstimator: Estimator object.
        """
        if pseudo_count is None:
            return GeometricEstimator(name=self.name, keys=self.keys)
        else:
            return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'GeometricDataEncoder':
        """Return a GeometricDataEncoder for this distribution.

        Returns:
            GeometricDataEncoder: Encoder object.
        """
        return GeometricDataEncoder()


class GeometricSampler(DistributionSampler):
    """Sampler for the geometric distribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GeometricDistribution): GeometricDistribution to sample from.
    """

    def __init__(self, dist: GeometricDistribution, seed: Optional[int] = None) -> None:
        """Initialize GeometricSampler.

        Args:
            dist (GeometricDistribution): GeometricDistribution to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from geometric distribution.

        Args:
            size (Optional[int], optional): Number of iid samples to draw. If None, returns a single sample.

        Returns:
            Union[int, np.ndarray]: Single sample (int) or numpy array of ints.
        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the geometric distribution.

    Attributes:
        sum (float): Aggregate weighted sum of observations.
        count (float): Aggregate sum of weighted observation count.
        name (Optional[str]): Name for the accumulator.
        keys (Optional[str]): Key for merging sufficient statistics.
    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize GeometricAccumulator.

        Args:
            name (Optional[str], optional): Name for the accumulator.
            keys (Optional[str], optional): Key for merging sufficient statistics.
        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional['GeometricDistribution']) -> None:
        """Update accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[GeometricDistribution]): Not used.
        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(
        self,
        x: 'GeometricEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['GeometricDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (GeometricEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[GeometricDistribution]): Not used.
        """
        self.sum += np.dot(x.data, weights)
        self.count += np.sum(weights)

    def initialize(self, x: int, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: 'GeometricEncodedDataSequence',
        weights: np.ndarray,
        rng: Optional[RandomState]
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (GeometricEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'GeometricAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[float, float]): (count, sum) to combine.

        Returns:
            GeometricAccumulator: Self after combining.
        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self) -> Tuple[float, float]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[float, float]: (count, sum)
        """
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'GeometricAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, float]): (count, sum) values.

        Returns:
            GeometricAccumulator: Self after setting values.
        """
        self.count = x[0]
        self.sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1 = stats_dict[self.keys]
                self.count += x0
                self.sum += x1
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.count, self.sum = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GeometricDataEncoder':
        """Return a GeometricDataEncoder for this accumulator.

        Returns:
            GeometricDataEncoder: Encoder object.
        """
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating GeometricAccumulator objects.

    Attributes:
        name (Optional[str]): Name for the factory.
        keys (Optional[str]): Key for merging sufficient statistics.
    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize GeometricAccumulatorFactory.

        Args:
            name (Optional[str], optional): Name for the factory.
            keys (Optional[str], optional): Key for merging sufficient statistics.
        """
        self.name = name
        self.keys = keys

    def make(self) -> 'GeometricAccumulator':
        """Create a new GeometricAccumulator.

        Returns:
            GeometricAccumulator: New accumulator instance.
        """
        return GeometricAccumulator(name=self.name, keys=self.keys)


class GeometricEstimator(ParameterEstimator):
    """Estimator for the geometric distribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Pseudo-count for regularization.
        suff_stat (Optional[float]): Probability of success (value between (0,1)).
        name (Optional[str]): Name for the estimator.
        keys (Optional[str]): Key for merging sufficient statistics.
    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize GeometricEstimator.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.
            suff_stat (Optional[float], optional): Probability of success (value between (0,1)).
            name (Optional[str], optional): Name for the estimator.
            keys (Optional[str], optional): Key for merging sufficient statistics.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("GeometricEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = min(max(suff_stat, 0.0), 1.0) if suff_stat is not None else None
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GeometricAccumulatorFactory':
        """Return a GeometricAccumulatorFactory for this estimator.

        Returns:
            GeometricAccumulatorFactory: Factory object.
        """
        return GeometricAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'GeometricDistribution':
        """Estimate a GeometricDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[float, float]): (count, sum) sufficient statistics.

        Returns:
            GeometricDistribution: Estimated distribution.
        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (suff_stat[1] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            p = suff_stat[0] / suff_stat[1]

        return GeometricDistribution(p, name=self.name)


class GeometricDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid geometric observations with data type int."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'GeometricDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(self, x: Union[Sequence[int], np.ndarray]) -> 'GeometricEncodedDataSequence':
        """Encode a sequence of geometric observations.

        Args:
            x (Union[Sequence[int], np.ndarray]): Sequence of iid geometric observations.

        Returns:
            GeometricEncodedDataSequence: Encoded data sequence.

        Raises:
            Exception: If any value in x is less than 1 or is NaN.
        """
        rv = np.asarray(x)
        if np.any(rv < 1) or np.any(np.isnan(rv)):
            raise Exception('GeometricDistribution requires integers greater than 0 for x.')
        else:
            return GeometricEncodedDataSequence(data=np.asarray(rv, dtype=float))


class GeometricEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for vectorized functions.

    Attributes:
        data (np.ndarray): Sequence of iid geometric observations.
    """

    def __init__(self, data: np.ndarray):
        """Initialize GeometricEncodedDataSequence.

        Args:
            data (np.ndarray): Sequence of iid geometric observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'GeometricEncodedDataSequence(data={self.data})'




