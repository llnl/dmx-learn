r"""Provide log-normal distributions, estimation, sampling, and encoding.

``LogGaussianDistribution`` uses the parameterization
:math:`\log X \sim \mathcal{N}(\mu, \sigma^2)` on :math:`X \in (0, \infty)`.
The remaining classes implement the sampler, sufficient statistics, maximum-
likelihood estimator, and sequence encoding used by :mod:`dmx.stats`.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import exp, isinf, isnan, log, pi, sqrt
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class LogGaussianDistribution(SequenceEncodableProbabilityDistribution):
    r"""Represent a univariate log-normal distribution.

    The log observation has mean ``mu`` and variance ``sigma2``:

    .. math::

       p(x) = \frac{1}{x\sqrt{2\pi\sigma^2}}
              \exp\left[-\frac{(\log x-\mu)^2}{2\sigma^2}\right],
       \qquad x > 0.

    A nonpositive or nonfinite ``sigma2`` is replaced by ``1.0``.

    Attributes:
        mu: Mean of the log observation.
        sigma2: Variance of the log observation.
        const: Gaussian normalizing constant before the ``1 / x`` factor.
        log_const: Logarithm of ``const``.
        name: Optional instance name.
        keys: Optional key for sharing sufficient statistics.
    """

    def __init__(
        self,
        mu: float,
        sigma2: float,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a log-normal distribution.

        Args:
            mu: Mean of ``log(X)``.
            sigma2: Variance of ``log(X)``; invalid values are replaced by ``1.0``.
            name: Optional instance name.
            keys: Optional key for sharing sufficient statistics.
        """
        super().__init__()
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = float(-0.5 * log(2.0 * pi * self.sigma2))
        self.const = float(1.0 / sqrt(2.0 * pi * self.sigma2))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        return (
            f"LogGaussianDistribution({repr(self.mu)}, {repr(self.sigma2)}, "
            f"name={repr(self.name)}, keys={repr(self.keys)})"
        )

    def density(self, x: float) -> float:
        """Density of Log-Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density of Log-Gaussian at x.

        """
        return float(
            self.const * exp(-0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2) / x
        )

    def log_density(self, x: float) -> float:
        """Log-density of log-Gaussian distribution at observation x.

        Args:
            x (float): Positive valued observation of log-Gaussian.

        Returns:
            float: Log-density at observation x.

        """
        return float(
            self.log_const - 0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2 - np.log(x)
        )

    def seq_ld_lambda(self) -> List[Callable]:
        """Return the vectorized log-density callable."""
        return [self.seq_log_density]

    def seq_log_density(self, x: "LogGaussianEncodedDataSequence") -> np.ndarray:
        """Evaluate log densities for encoded log observations.

        The encoded data array and returned array both have shape ``(n,)``.
        """
        if not isinstance(x, LogGaussianEncodedDataSequence):
            raise TypeError(
                "LogGaussianEncodedDataSequence required for seq_log_density()."
            )

        rv = x.data - self.mu
        rv *= rv
        rv *= -0.5 / self.sigma2
        rv += self.log_const
        rv -= x.data

        return np.asarray(rv)

    def sampler(self, seed: Optional[int] = None) -> "LogGaussianSampler":
        """Create a sampler, optionally initialized with ``seed``."""
        return LogGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "LogGaussianEstimator":
        """Create an estimator centered on this distribution when regularized."""
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return LogGaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count),
                suff_stat=suff_stat,
                name=self.name,
                keys=self.keys,
            )
        return LogGaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "LogGaussianDataEncoder":
        """Create an encoder for positive scalar observations."""
        return LogGaussianDataEncoder()


class LogGaussianSampler(DistributionSampler):
    """Draw independent samples from a log-normal distribution.

    Attributes:
        dist: Distribution to sample from.
        rng: Pseudorandom number generator.
    """

    def __init__(
        self, dist: LogGaussianDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler.

        Args:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample
                from.
            seed (Optional[int]): Used to set seed in random sampler.

        """
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from LogGaussianSampler object.

        Numpy array of length 'size' from log-Gaussian distribution with scale beta if
        size not None. Else a single
        sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return np.exp(
            self.rng.normal(
                loc=self.dist.mu, scale=np.sqrt(self.dist.sigma2), size=size
            )
        )


class LogGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    r"""Accumulate weighted log-normal sufficient statistics.

    The public statistic is ``(log_sum, log_sum2, count, count2)``, containing
    weighted sums of :math:`\log x`, squared log observations, and their two
    effective counts.

    Attributes:
        log_sum: Weighted sum of log observations.
        log_sum2: Weighted sum of squared log observations.
        count: Effective count for the mean statistic.
        count2: Effective count for the variance statistic.
        keys: Optional key for sharing statistics.
        name: Optional instance name.
    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize an empty accumulator."""
        self.log_sum = 0.0
        self.log_sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(
        self, x: float, weight: float, estimate: Optional["LogGaussianDistribution"]
    ) -> None:
        """Add one weighted positive observation."""
        x_weight = np.log(x) * weight
        self.log_sum += x_weight
        self.log_sum2 += np.log(x) * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize from one observation; ``rng`` is unused."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: "LogGaussianEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Initialize from encoded observations; ``rng`` is unused."""
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "LogGaussianEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[LogGaussianDistribution],
    ) -> None:
        """Add weighted encoded observations with shape ``(n,)``."""
        self.log_sum += np.dot(x.data, weights)
        self.log_sum2 += np.dot(x.data * x.data, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(
        self, suff_stat: Tuple[float, float, float, float]
    ) -> "LogGaussianAccumulator":
        """Add a sufficient-statistic tuple to this accumulator."""
        self.log_sum += suff_stat[0]
        self.log_sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> Tuple[float, float, float, float]:
        """Return ``(log_sum, log_sum2, count, count2)``."""
        return self.log_sum, self.log_sum2, self.count, self.count2

    def from_value(
        self, x: Tuple[float, float, float, float]
    ) -> "LogGaussianAccumulator":
        """Replace this accumulator from a sufficient-statistic tuple."""
        self.log_sum = x[0]
        self.log_sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under ``keys``."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator from ``stats_dict`` when keyed."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "LogGaussianDataEncoder":
        """Create the matching data encoder."""
        return LogGaussianDataEncoder()


class LogGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Create log-normal sufficient-statistic accumulators.

    Attributes:
        name (Optional[str]): Name of the LogGaussianAccumulatorFactory object.
        keys (Optional[str]): String id for merging sufficient statistics of
            LogGaussianAccumulator.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize the factory."""
        self.keys = keys
        self.name = name

    def make(self) -> "LogGaussianAccumulator":
        """Create an empty accumulator."""
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(ParameterEstimator):
    """Estimate log-normal mean and variance from weighted statistics.

    ``pseudo_count`` independently regularizes the log mean and log variance.
    The corresponding ``suff_stat`` entries are the centering mean and variance.

    Attributes:
        pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
        suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and
            variance (sigma2).
        name (Optional[str]): String name of LogGaussianEstimator instance.
        keys (Optional[str]): String keys of LogGaussianEstimator instance for combining
            sufficient statistics.

    """

    def __init__(
        self,
        pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
        suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ):
        """Initialize the estimator and optional regularization targets.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two
                positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and
                positive float.
            name (Optional[str]): Assign a name to LogGaussianEstimator.
            keys (Optional[str]): Assign keys to LogGaussianEstimator for combining
                sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("LogGaussianEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "LogGaussianAccumulatorFactory":
        """Create a matching accumulator factory."""
        return LogGaussianAccumulatorFactory(self.name, self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float]
    ) -> "LogGaussianDistribution":
        """Estimate a distribution from four log-domain statistics.

        ``nobs`` is accepted for protocol compatibility; the two counts in
        ``suff_stat`` control estimation.
        """
        log_x, log_x2 = suff_stat[0], suff_stat[1]
        nobs_loc1, nobs_loc2 = suff_stat[2], suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (log_x + self.pseudo_count[0] * self.suff_stat[0]) / (
                nobs_loc1 + self.pseudo_count[0]
            )
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (
                suff_stat[1]
                - mu * mu * nobs_loc2
                + self.pseudo_count[1] * self.suff_stat[1]
            ) / (nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = np.sum(log_x2 - np.sum(log_x) ** 2 / nobs_loc1) / nobs_loc2

        return LogGaussianDistribution(mu, sigma2, name=self.name)


class LogGaussianDataEncoder(DataSequenceEncoder):
    """Encode positive scalar observations as their logarithms."""

    def __str__(self) -> str:
        """Return the encoder name."""
        return "LogGaussianDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a log-normal encoder."""
        return isinstance(other, LogGaussianDataEncoder)

    def seq_encode(
        self, x: Union[List[float], np.ndarray]
    ) -> "LogGaussianEncodedDataSequence":
        """Encode sequence of iid Log-Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid log-Gaussian
                observations.

        Returns:
            Encoded sequence whose data has shape ``(n,)``.

        """
        rv = np.asarray(np.log(x), dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise ValueError("LogGaussianDistribution requires support x in (0,inf).")
        return LogGaussianEncodedDataSequence(data=rv)


class LogGaussianEncodedDataSequence(EncodedDataSequence):
    """Store a one-dimensional array of log observations.

    Attributes:
        data (np.ndarray): IID log Gaussian observations.

    """

    def __init__(self, data: np.ndarray):
        """Initialize the encoded sequence."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded data."""
        return f"LogGaussianEncodedDataSequence(data={self.data})"
