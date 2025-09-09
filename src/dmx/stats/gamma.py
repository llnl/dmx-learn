"""Create, estimate, and sample from a gamma distribution with shape k and scale theta.

Defines the GammaDistribution, GammaSampler, GammaAccumulatorFactory, GammaAccumulator, GammaEstimator,
and the GammaDataEncoder classes for use with dmx-learn.
"""

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln
from typing import Tuple, List, Optional, Union, Dict, Any
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
from dmx.utils.special import digamma, trigamma


class GammaDistribution(SequenceEncodableProbabilityDistribution):
    """Gamma distribution with shape k and scale theta.

    Attributes:
        k (float): Positive real-valued shape parameter.
        theta (float): Positive real-valued scale parameter.
        name (Optional[str]): Name for the GammaDistribution instance.
        log_const (float): Normalizing constant of gamma distribution.
        keys (Optional[str]): Key for parameters of distribution.
    """

    def __init__(
        self,
        k: float,
        theta: float,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize GammaDistribution.

        Args:
            k (float): Positive real-valued shape parameter.
            theta (float): Positive real-valued scale parameter.
            name (Optional[str], optional): Name for the GammaDistribution instance.
            keys (Optional[str], optional): Key for parameters of distribution.
        """
        self.k = k
        self.theta = theta
        self.log_const = -(gammaln(k) + k * log(theta))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        s0 = repr(float(self.k))
        s1 = repr(float(self.theta))
        s2 = repr(self.name)
        s3 = repr(self.keys)
        return f'GammaDistribution({s0}, {s1}, name={s2}, keys={s3})'

    def density(self, x: float) -> float:
        """Evaluate the density of the gamma distribution at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density evaluated at x.
        """
        return exp(self.log_const + (self.k - one) * log(x) - x / self.theta)

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of the gamma distribution at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Log-density evaluated at x.
        """
        return self.log_const + (self.k - one) * log(x) - x / self.theta

    def seq_log_density(self, x: 'GammaEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (GammaEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, GammaEncodedDataSequence):
            raise Exception("GammaEncodedDataSequence required for seq_log_density().")

        rv = x.data[0] * (-1.0 / self.theta)
        if self.k != 1.0:
            rv += x.data[1] * (self.k - 1.0)
        rv += self.log_const
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GammaSampler':
        """Return a GammaSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            GammaSampler: Sampler object.
        """
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GammaEstimator':
        """Return a GammaEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            GammaEstimator: Estimator object.
        """
        if pseudo_count is None:
            return GammaEstimator(name=self.name, keys=self.keys)
        else:
            suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
            return GammaEstimator(
                pseudo_count=(pseudo_count, pseudo_count),
                suff_stat=suff_stat,
                name=self.name,
                keys=self.keys
            )

    def dist_to_encoder(self) -> 'GammaDataEncoder':
        """Return a GammaDataEncoder for this distribution.

        Returns:
            GammaDataEncoder: Encoder object.
        """
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):
    """Sampler for the gamma distribution.

    Attributes:
        rng (RandomState): Random number generator.
        dist (GammaDistribution): GammaDistribution to sample from.
        seed (Optional[int]): Seed for random number generator.
    """

    def __init__(self, dist: 'GammaDistribution', seed: Optional[int] = None) -> None:
        """Initialize GammaSampler.

        Args:
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: Optional[int] = None) -> Union[float, List[float]]:
        """Draw iid samples from the gamma distribution.

        Args:
            size (Optional[int], optional): Number of iid samples to draw. If None, returns a single sample.

        Returns:
            Union[float, List[float]]: Single sample (float) if size is None, else a list of samples.
        """
        if size:
            return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size).tolist()
        else:
            return float(self.rng.gamma(shape=self.dist.k, scale=self.dist.theta))


class GammaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the gamma distribution.

    Attributes:
        nobs (float): Number of observations accumulated.
        sum (float): Weighted sum of observations.
        sum_of_logs (float): Weighted sum of log(observations).
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize GammaAccumulator.

        Args:
            keys (Optional[str], optional): Key for merging sufficient statistics.
            name (Optional[str], optional): Name for object.
        """
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.keys = keys
        self.name = name

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.update(x, weight, None)

    def seq_initialize(self, x: 'GammaEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (GammaEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.seq_update(x, weights, None)

    def update(self, x: float, weight: float, estimate: Optional['GammaDistribution']) -> None:
        """Update accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[GammaDistribution]): Not used.
        """
        self.nobs += weight
        self.sum += x * weight
        self.sum_of_logs += log(x) * weight

    def seq_update(
        self,
        x: 'GammaEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['GammaDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (GammaEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[GammaDistribution]): Not used.
        """
        self.sum += np.dot(x.data[0], weights)
        self.sum_of_logs += np.dot(x.data[1], weights)
        self.nobs += np.sum(weights)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'GammaAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[float, float, float]): (nobs, sum, sum_of_logs) to combine.

        Returns:
            GammaAccumulator: Self after combining.
        """
        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]
        return self

    def value(self) -> Tuple[float, float, float]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[float, float, float]: (nobs, sum, sum_of_logs)
        """
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: Tuple[float, float, float]) -> 'GammaAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, float, float]): (nobs, sum, sum_of_logs) values.

        Returns:
            GammaAccumulator: Self after setting values.
        """
        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.nobs += x0
                self.sum += x1
                self.sum_of_logs += x2
            else:
                stats_dict[self.keys] = (self.nobs, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.nobs = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> 'GammaDataEncoder':
        """Return a GammaDataEncoder for this accumulator.

        Returns:
            GammaDataEncoder: Encoder object.
        """
        return GammaDataEncoder()


class GammaAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating GammaAccumulator objects.

    Attributes:
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize GammaAccumulatorFactory.

        Args:
            keys (Optional[str], optional): Key for merging sufficient statistics.
            name (Optional[str], optional): Name for object.
        """
        self.keys = keys
        self.name = name

    def make(self) -> 'GammaAccumulator':
        """Create a new GammaAccumulator.

        Returns:
            GammaAccumulator: New accumulator instance.
        """
        return GammaAccumulator(keys=self.keys, name=self.name)


class GammaEstimator(ParameterEstimator):
    """Estimator for the gamma distribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Tuple[float, float]): Values used to re-weight sufficient statistics.
        suff_stat (Tuple[float, float]): Prior shape 'k' and scale 'theta'.
        threshold (float): Threshold for estimating the shape of gamma.
        name (Optional[str]): Name for the estimator.
        keys (Optional[str]): Key for combining sufficient statistics.
    """

    def __init__(
        self,
        pseudo_count: Tuple[float, float] = (0.0, 0.0),
        suff_stat: Tuple[float, float] = (1.0, 0.0),
        threshold: float = 1.0e-8,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize GammaEstimator.

        Args:
            pseudo_count (Tuple[float, float], optional): Values used to re-weight sufficient statistics.
            suff_stat (Tuple[float, float], optional): Prior shape 'k' and scale 'theta'.
            threshold (float, optional): Threshold for estimating the shape of gamma.
            name (Optional[str], optional): Name for the estimator.
            keys (Optional[str], optional): Key for combining sufficient statistics.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("GammaEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GammaAccumulatorFactory':
        """Return a GammaAccumulatorFactory for this estimator.

        Returns:
            GammaAccumulatorFactory: Factory object.
        """
        return GammaAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, float, float]
    ) -> 'GammaDistribution':
        """Estimate a GammaDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[float, float, float]): (nobs, sum, sum_of_logs) sufficient statistics.

        Returns:
            GammaDistribution: Estimated distribution.
        """
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] == 0:
            return GammaDistribution(1.0, 1.0)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        return GammaDistribution(k, adj_sum / (k * adj_lcnt), name=self.name)

    @staticmethod
    def estimate_shape(avg_sum: float, avg_sum_of_logs: float, threshold: float) -> float:
        """Estimate the shape parameter of the GammaDistribution.

        Args:
            avg_sum (float): Weighted mean of gamma observations.
            avg_sum_of_logs (float): Weighted mean of log gamma observations.
            threshold (float): Threshold for convergence of shape estimation.

        Returns:
            float: Estimate of shape parameter 'k'.
        """
        s = log(avg_sum) - avg_sum_of_logs
        old_k = inf
        k = (3 - s + sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
        while abs(old_k - k) > threshold:
            old_k = k
            k -= (log(k) - digamma(k) - s) / (one / k - trigamma(k))
        return k


class GammaDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid Gamma observations with data type float."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'GammaDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, GammaDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'GammaEncodedDataSequence':
        """Encode a sequence of gamma observations.

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid gamma observations.

        Returns:
            GammaEncodedDataSequence: Encoded data sequence.

        Raises:
            Exception: If any value in x is not positive or is NaN.
        """
        rv1 = np.asarray(x, dtype=float)
        if np.any(rv1 <= 0) or np.any(np.isnan(rv1)):
            raise Exception('GammaDistribution has support x > 0.')
        else:
            rv2 = np.log(rv1)
            return GammaEncodedDataSequence(data=(rv1, rv2))


class GammaEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for vectorized function calls.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray]): Encoded data for gamma distribution.
    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]):
        """GammaEncodedDataSequence object.

        Args:
            data (Tuple[np.ndarray, np.ndarray]): Encoded data for gamma distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'GammaEncodedDataSequence(data={self.data}'

