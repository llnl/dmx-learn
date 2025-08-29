"""Evaluate, estimate, and sample from a Gaussian distribution with mean mu and variance sigma2.

Defines the GaussianDistribution, GaussianSampler, GaussianAccumulatorFactory, GaussianAccumulator,
GaussianEstimator, and the GaussianDataEncoder classes for use with dmx-learn.

Data type: float
"""

import numpy as np
from numpy.random import RandomState
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
from typing import Optional, Tuple, List, Callable, Dict, Union, Any


class GaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Gaussian distribution with mean mu and variance sigma2.

    Attributes:
        mu (float): Mean of the Gaussian distribution.
        sigma2 (float): Variance of the Gaussian distribution.
        name (Optional[str]): Name of the object.
        const (float): Normalizing constant of the Gaussian (depends on sigma2).
        log_const (float): Log of the normalizing constant.
        keys (Optional[str]): Key for the distribution.
    """

    def __init__(
        self,
        mu: float,
        sigma2: float,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize GaussianDistribution.

        Args:
            mu (float): Mean of the Gaussian distribution.
            sigma2 (float): Variance of the Gaussian distribution (must be positive).
            name (Optional[str], optional): Name for the object.
            keys (Optional[str], optional): Key for the distribution.
        """
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f'GaussianDistribution({repr(float(self.mu))}, {repr(float(self.sigma2))}, '
            f'name={repr(self.name)}, keys={repr(self.keys)})'
        )

    def density(self, x: float) -> float:
        """Evaluate the density of the Gaussian distribution at x.

        Args:
            x (float): Observation.

        Returns:
            float: Density at x.
        """
        return self.const * exp(-0.5 * (x - self.mu) * (x - self.mu) / self.sigma2)

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of the Gaussian distribution at x.

        Args:
            x (float): Observation.

        Returns:
            float: Log-density at x.
        """
        return self.log_const - 0.5 * (x - self.mu) * (x - self.mu) / self.sigma2

    def seq_ld_lambda(self) -> List[Callable]:
        """Return a list containing the seq_log_density method."""
        return [self.seq_log_density]

    def seq_log_density(self, x: 'GaussianEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (GaussianEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, GaussianEncodedDataSequence):
            raise Exception('GaussianDistribution.seq_log_density() requires GaussianEncodedDataSequence.')

        rv = x.data - self.mu
        rv *= rv
        rv *= -0.5 / self.sigma2
        rv += self.log_const

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GaussianSampler':
        """Return a GaussianSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            GaussianSampler: Sampler object.
        """
        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GaussianEstimator':
        """Return a GaussianEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            GaussianEstimator: Estimator object.
        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count),
                suff_stat=suff_stat,
                name=self.name,
                keys=self.keys
            )
        else:
            return GaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'GaussianDataEncoder':
        """Return a GaussianDataEncoder for this distribution.

        Returns:
            GaussianDataEncoder: Encoder object.
        """
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):
    """Sampler for drawing samples from a GaussianDistribution instance.

    Attributes:
        dist (GaussianDistribution): GaussianDistribution instance to sample from.
        rng (RandomState): Random number generator.
    """

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None) -> None:
        """Initialize GaussianSampler.

        Args:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw iid samples from the Gaussian distribution.

        Args:
            size (Optional[int], optional): Number of samples to draw. If None, returns a single sample.

        Returns:
            Union[float, np.ndarray]: Single sample or array of samples.
        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the Gaussian distribution.

    Attributes:
        sum (float): Sum of weighted observations.
        sum2 (float): Sum of weighted squared observations.
        count (float): Sum of weights for observations.
        count2 (float): Sum of weights for squared observations.
        keys (Optional[str]): Key for mean and variance.
        name (Optional[str]): Name for the accumulator.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize GaussianAccumulator.

        Args:
            keys (Optional[str], optional): Key for mean and variance.
            name (Optional[str], optional): Name for the accumulator.
        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['GaussianDistribution']) -> None:
        """Update accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[GaussianDistribution]): Not used.
        """
        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += x * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: 'GaussianEncodedDataSequence',
        weights: np.ndarray,
        rng: Optional[RandomState]
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (GaussianEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: 'GaussianEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['GaussianDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (GaussianEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[GaussianDistribution]): Not used.
        """
        self.sum += np.dot(x.data, weights)
        self.sum2 += np.dot(x.data * x.data, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[float, float, float, float]): (sum, sum2, count, count2) to combine.

        Returns:
            GaussianAccumulator: Self after combining.
        """
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]
        return self

    def value(self) -> Tuple[float, float, float, float]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[float, float, float, float]: (sum, sum2, count, count2)
        """
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, float, float, float]): (sum, sum2, count, count2) values.

        Returns:
            GaussianAccumulator: Self after setting values.
        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2, x3 = stats_dict[self.keys].value()
                self.sum += x0
                self.sum2 += x1
                self.count += x2
                self.count2 += x3
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, 'GaussianAccumulator']) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, GaussianAccumulator]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GaussianDataEncoder':
        """Return a GaussianDataEncoder for this accumulator.

        Returns:
            GaussianDataEncoder: Encoder object.
        """
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating GaussianAccumulator objects.

    Attributes:
        name (Optional[str]): Name of the factory.
        keys (Optional[str]): Key for merging sufficient statistics.
    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize GaussianAccumulatorFactory.

        Args:
            name (Optional[str], optional): Name for the factory.
            keys (Optional[str], optional): Key for merging sufficient statistics.
        """
        self.keys = keys
        self.name = name

    def make(self) -> 'GaussianAccumulator':
        """Create a new GaussianAccumulator.

        Returns:
            GaussianAccumulator: New accumulator instance.
        """
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):
    """Estimator for the Gaussian distribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for sufficient statistics.
        suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
        name (Optional[str]): Name of the estimator.
        keys (Optional[str]): Key for mean and variance.
    """

    def __init__(
        self,
        pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
        suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
        name: Optional[str] = None,
        keys: Optional[str] = None
    ):
        """Initialize GaussianEstimator.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean and variance.
            name (Optional[str], optional): Name for the estimator.
            keys (Optional[str], optional): Key for mean and variance.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("GaussianEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GaussianAccumulatorFactory':
        """Return a GaussianAccumulatorFactory for this estimator.

        Returns:
            GaussianAccumulatorFactory: Factory object.
        """
        return GaussianAccumulatorFactory(self.name, self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, float, float, float]
    ) -> 'GaussianDistribution':
        """Estimate a GaussianDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[float, float, float, float]): (sum, sum2, count, count2) sufficient statistics.

        Returns:
            GaussianDistribution: Estimated distribution.
        """
        nobs_loc1 = suff_stat[2]
        nobs_loc2 = suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (suff_stat[0] + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (
                (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1])
                / (nobs_loc2 + self.pseudo_count[1])
            )
        else:
            sigma2 = suff_stat[1] / nobs_loc2 - mu * mu

        return GaussianDistribution(mu, sigma2, name=self.name)


class GaussianDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'GaussianDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'GaussianEncodedDataSequence':
        """Encode a sequence of iid Gaussian observations.

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid Gaussian observations.

        Returns:
            GaussianEncodedDataSequence: Encoded data sequence.

        Raises:
            Exception: If any value in x is NaN or infinite.
        """
        rv = np.asarray(x, dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('GaussianDistribution requires support x in (-inf, inf).')

        return GaussianEncodedDataSequence(data=rv)


class GaussianEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for use with vectorized function calls.

    Attributes:
        data (np.ndarray): Sequence of iid Gaussian observations.
    """

    def __init__(self, data: np.ndarray):
        """Initialize GaussianEncodedDataSequence.

        Args:
            data (np.ndarray): Sequence of iid Gaussian observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'GaussianEncodedDataSequence(data={self.data})'


