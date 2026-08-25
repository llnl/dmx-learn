"""Geometric distributions, sampling, estimation, and sequence encoding.

``GeometricDistribution`` models the one-based trial count through the first
success: ``P(X = k) = p * (1 - p) ** (k - 1)`` for positive integers ``k``.
Scalar methods accept one integer, while sequence methods consume the
one-dimensional array produced by ``GeometricDataEncoder``.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import exp
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class GeometricDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a one-based geometric distribution.

    The support is ``{1, 2, ...}``, and ``p`` is the probability of success on
    each trial. The constructor clips ``p`` to the closed interval ``[0, 1]``.

    Attributes:
        p (float): Success probability clipped to ``[0, 1]``.
        log_p (float): Log of probability of success p.
        log_1p (float): Log of 1-p (probability of failure).
        name (Optional[str]): Name for the GeometricDistribution object.
        keys (Optional[str]): Key for parameter p.
    """

    def __init__(
        self, p: float, name: Optional[str] = None, keys: Optional[str] = None
    ) -> None:
        """Initialize GeometricDistribution.

        Args:
            p: Success probability; values outside ``[0, 1]`` are clipped.
            name: Optional name for the distribution.
            keys: Optional key for tying sufficient statistics.
        """
        super().__init__()
        self.p = max(0.0, min(p, 1.0))
        self.log_p = float(np.log(self.p))
        self.log_1p = float(np.log1p(-self.p))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"GeometricDistribution({repr(self.p)}, name={repr(self.name)}, "
            f"keys={repr(self.keys)})"
        )

    def density(self, x: int) -> float:
        """Evaluate the probability mass at an integer observation.

        Args:
            x (int): Observed geometric value (1,2,3,...).

        Returns:
            Probability mass at ``x``.
        """
        return float(exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        """Evaluate the log-probability mass at an integer observation.

        Args:
            x (int): Must be a natural number (1,2,3,...).

        Returns:
            Log-probability mass at ``x``.
        """
        return float((x - 1) * self.log_1p + self.log_p)

    def seq_log_density(self, x: "GeometricEncodedDataSequence") -> np.ndarray:
        """Evaluate log-probability masses for an encoded sequence.

        Args:
            x: Encoded sequence containing ``N`` observations.

        Returns:
            Array of shape ``(N,)`` containing one log-probability mass per
            observation.
        """
        if not isinstance(x, GeometricEncodedDataSequence):
            raise TypeError(
                "GeometricEncodedDataSequence required for seq_log_density()."
            )

        rv = x.data - 1
        rv *= self.log_1p
        rv += self.log_p

        return np.asarray(rv)

    def sampler(self, seed: Optional[int] = None) -> "GeometricSampler":
        """Return a GeometricSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            GeometricSampler: Sampler object.
        """
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "GeometricEstimator":
        """Create an estimator initialized from this distribution.

        With ``pseudo_count``, estimation shrinks the weighted success count
        toward this distribution's success probability.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            GeometricEstimator: Estimator object.
        """
        if pseudo_count is None:
            return GeometricEstimator(name=self.name, keys=self.keys)
        return GeometricEstimator(
            pseudo_count=pseudo_count,
            suff_stat=self.p,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "GeometricDataEncoder":
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
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from geometric distribution.

        Args:
            size (Optional[int], optional): Number of iid samples to draw. If None,
                returns a single sample.

        Returns:
            Union[int, np.ndarray]: Single sample (int) or numpy array of ints.
        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted geometric sufficient statistics.

    The sufficient-statistic tuple is ``(count, sum_x)``. Scalar updates accept
    nonnegative values, while the sequence encoder enforces the documented
    positive-integer support.

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

    def update(
        self, x: int, weight: float, estimate: Optional["GeometricDistribution"]
    ) -> None:
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
        x: "GeometricEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["GeometricDistribution"],
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
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: "GeometricEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (GeometricEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Random number generator (not used).
        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> "GeometricAccumulator":
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

    def from_value(self, x: Tuple[float, float]) -> "GeometricAccumulator":
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

    def acc_to_encoder(self) -> "GeometricDataEncoder":
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

    def make(self) -> "GeometricAccumulator":
        """Create a new GeometricAccumulator.

        Returns:
            GeometricAccumulator: New accumulator instance.
        """
        return GeometricAccumulator(name=self.name, keys=self.keys)


class GeometricEstimator(ParameterEstimator):
    """Estimate a geometric success probability from weighted statistics.

    Without prior information, the maximum-likelihood estimate is
    ``p = count / sum_x``. A pseudo-count adds weighted prior successes from
    ``suff_stat``; when no prior statistic is supplied, it adds the same amount
    to the numerator and denominator.

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
        keys: Optional[str] = None,
    ) -> None:
        """Initialize GeometricEstimator.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.
            suff_stat (Optional[float], optional): Probability of success (value between
                (0,1)).
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
        self.suff_stat = (
            min(max(suff_stat, 0.0), 1.0) if suff_stat is not None else None
        )
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "GeometricAccumulatorFactory":
        """Return a GeometricAccumulatorFactory for this estimator.

        Returns:
            GeometricAccumulatorFactory: Factory object.
        """
        return GeometricAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, float]
    ) -> "GeometricDistribution":
        """Estimate a GeometricDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[float, float]): (count, sum) sufficient statistics.

        Returns:
            GeometricDistribution: Estimated distribution.
        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (
                suff_stat[1] + self.pseudo_count
            )
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            p = suff_stat[0] / suff_stat[1]

        return GeometricDistribution(p, name=self.name)


class GeometricDataEncoder(DataSequenceEncoder):
    """Encode positive geometric observations as a float array of shape ``(N,)``."""

    def __str__(self) -> str:
        """Return string representation."""
        return "GeometricDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(
        self, x: Union[Sequence[int], np.ndarray]
    ) -> "GeometricEncodedDataSequence":
        """Encode a sequence of geometric observations.

        Args:
            x: One-dimensional sequence of ``N`` positive integer observations.

        Returns:
            Encoded sequence backed by a float array with shape ``(N,)``.

        Raises:
            ValueError: If any value is less than one or NaN.
        """
        rv = np.asarray(x)
        if np.any(rv < 1) or np.any(np.isnan(rv)):
            raise ValueError(
                "GeometricDistribution requires integers greater than 0 for x."
            )
        return GeometricEncodedDataSequence(data=np.asarray(rv, dtype=float))


class GeometricEncodedDataSequence(EncodedDataSequence):
    """Store geometric observations for vectorized operations.

    Attributes:
        data (np.ndarray): Float array with shape ``(N,)``.
    """

    def __init__(self, data: np.ndarray):
        """Initialize GeometricEncodedDataSequence.

        Args:
            data: Float array with shape ``(N,)``.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GeometricEncodedDataSequence(data={self.data})"
