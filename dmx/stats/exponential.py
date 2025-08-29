"""Create, estimate, and sample from an exponential distribution with scale beta.

Defines the ExponentialDistribution, ExponentialSampler, ExponentialAccumulatorFactory, ExponentialAccumulator,
ExponentialEstimator, and the ExponentialDataEncoder classes for use with dmx-learn.
"""

from typing import Optional, Tuple, List, Union, Dict, Any
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
import numpy as np


class ExponentialDistribution(SequenceEncodableProbabilityDistribution):
    """Exponential distribution with scale parameter beta.

    Attributes:
        beta (float): Positive real number defining the scale of the exponential distribution.
        log_beta (float): Logarithm of the beta parameter.
        name (Optional[str]): Name for the ExponentialDistribution object.
        keys (Optional[str]): Key for parameters.
    """

    def __init__(self, beta: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize ExponentialDistribution.

        Args:
            beta (float): Positive real number defining the scale of the exponential distribution.
            name (Optional[str], optional): Name for the ExponentialDistribution object.
            keys (Optional[str], optional): Key for parameters.
        """
        self.beta = beta
        self.log_beta = np.log(beta)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        return f'ExponentialDistribution({repr(self.beta)}, name={repr(self.name)}, keys={repr(self.keys)})'

    def density(self, x: float) -> float:
        """Evaluate the density of the exponential distribution at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density evaluated at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of the exponential distribution at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Log-density evaluated at x.
        """
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.log_beta

    def seq_log_density(self, x: 'ExponentialEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (ExponentialEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, ExponentialEncodedDataSequence):
            raise Exception('ExponentialEncodedDataSequence required for seq_log_density().')

        rv = x.data * (-1.0 / self.beta)
        rv -= self.log_beta
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ExponentialSampler':
        """Return an ExponentialSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            ExponentialSampler: Sampler object.
        """
        return ExponentialSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ExponentialEstimator':
        """Return an ExponentialEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            ExponentialEstimator: Estimator object.
        """
        if pseudo_count is None:
            return ExponentialEstimator(name=self.name, keys=self.keys)
        else:
            return ExponentialEstimator(
                pseudo_count=pseudo_count,
                suff_stat=self.beta,
                name=self.name,
                keys=self.keys
            )

    def dist_to_encoder(self) -> 'ExponentialDataEncoder':
        """Return an ExponentialDataEncoder for this distribution.

        Returns:
            ExponentialDataEncoder: Encoder object.
        """
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):
    """Sampler for the exponential distribution.

    Attributes:
        dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
        rng (RandomState): Random number generator.
    """

    def __init__(self, dist: 'ExponentialDistribution', seed: Optional[int] = None) -> None:
        """Initialize ExponentialSampler.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw iid samples from the exponential distribution.

        Args:
            size (Optional[int], optional): Number of samples to draw. If None, returns a single sample.

        Returns:
            Union[float, np.ndarray]: Single sample or array of samples.
        """
        return self.rng.exponential(scale=self.dist.beta, size=size)


class ExponentialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the exponential distribution.

    Attributes:
        sum (float): Sum of observation values.
        count (float): Sum of weights for observations.
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize ExponentialAccumulator.

        Args:
            keys (Optional[str], optional): Key for merging sufficient statistics.
            name (Optional[str], optional): Name for object.
        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['ExponentialDistribution']) -> None:
        """Update accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[ExponentialDistribution]): Not used.
        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def initialize(self, x: float, weight: float, rng: Optional['np.random.RandomState']) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            rng (Optional[np.random.RandomState]): Not used.
        """
        self.update(x, weight, None)

    def seq_update(
        self,
        x: 'ExponentialEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['ExponentialDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (ExponentialEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[ExponentialDistribution]): Not used.
        """
        self.sum += np.dot(x.data, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(
        self,
        x: 'ExponentialEncodedDataSequence',
        weights: np.ndarray,
        rng: np.random.RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (ExponentialEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (np.random.RandomState): Not used.
        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'ExponentialAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[float, float]): (count, sum) to combine.

        Returns:
            ExponentialAccumulator: Self after combining.
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

    def from_value(self, x: Tuple[float, float]) -> 'ExponentialAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, float]): (count, sum) values.

        Returns:
            ExponentialAccumulator: Self after setting values.
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
                self.count = stats_dict[self.keys][0]
                self.sum = stats_dict[self.keys][1]

    def acc_to_encoder(self) -> 'ExponentialDataEncoder':
        """Return an ExponentialDataEncoder for this accumulator.

        Returns:
            ExponentialDataEncoder: Encoder object.
        """
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating ExponentialAccumulator objects.

    Attributes:
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize ExponentialAccumulatorFactory.

        Args:
            keys (Optional[str], optional): Key for merging sufficient statistics.
            name (Optional[str], optional): Name for object.
        """
        self.keys = keys
        self.name = name

    def make(self) -> 'ExponentialAccumulator':
        """Create a new ExponentialAccumulator.

        Returns:
            ExponentialAccumulator: New accumulator instance.
        """
        return ExponentialAccumulator(keys=self.keys, name=self.name)


class ExponentialEstimator(ParameterEstimator):
    """Estimator for the exponential distribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Used to weight sufficient statistics.
        suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
        name (Optional[str]): Name for the estimator.
        keys (Optional[str]): Key for combining sufficient statistics.
    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize ExponentialEstimator.

        Args:
            pseudo_count (Optional[float], optional): Used to weight sufficient statistics.
            suff_stat (Optional[float], optional): Positive float value for scale of exponential distribution.
            name (Optional[str], optional): Name for the estimator.
            keys (Optional[str], optional): Key for combining sufficient statistics.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("ExponentialEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'ExponentialAccumulatorFactory':
        """Return an ExponentialAccumulatorFactory for this estimator.

        Returns:
            ExponentialAccumulatorFactory: Factory object.
        """
        return ExponentialAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'ExponentialDistribution':
        """Estimate an ExponentialDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[float, float]): (count, sum) sufficient statistics.

        Returns:
            ExponentialDistribution: Estimated distribution.
        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[1] + self.suff_stat * self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[1] + self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        else:
            if suff_stat[0] > 0:
                p = suff_stat[1] / suff_stat[0]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, name=self.name)


class ExponentialDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid exponential observations with data type float."""

    def __str__(self) -> str:
        """Return string representation of ExponentialDataEncoder."""
        return 'ExponentialDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check if object is an instance of ExponentialDataEncoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if object is an instance of ExponentialDataEncoder, else False.
        """
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'ExponentialEncodedDataSequence':
        """Encode a sequence of exponential observations.

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid exponential observations.

        Returns:
            ExponentialEncodedDataSequence: Encoded data sequence.

        Raises:
            Exception: If any value in x is not positive or is NaN.
        """
        rv = np.asarray(x, dtype=float)

        if np.any(rv <= 0) or np.any(np.isnan(rv)):
            raise Exception('Exponential requires x > 0.')

        return ExponentialEncodedDataSequence(data=rv)


class ExponentialEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for vectorized function calls.

    Attributes:
        data (np.ndarray): Sequence of iid exponential observations.
    """

    def __init__(self, data: np.ndarray):
        """Initialize ExponentialEncodedDataSequence.

        Args:
            data (np.ndarray): Sequence of iid exponential observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'ExponentialEncodedDataSequence(data={self.data})'

