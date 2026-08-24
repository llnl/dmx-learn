"""Define a gamma prior and its legacy maximum-likelihood estimator.

``GammaDistribution(k, theta)`` uses shape ``k`` and scale ``theta``,
with support ``x > 0``. Bayesian likelihood modules use these values as the
current prior or posterior hyperparameters. The nested ``prior`` attribute is
estimator metadata and does not alter density, entropy, or cross-entropy.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional, cast

import numpy as np
import scipy.integrate
from numpy.random import RandomState

from dmx.bstats.nulldist import null_dist
from dmx.bstats.pdist import (
    DistributionSampler,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma, gammaln, trigamma

# Gamma retains the legacy tuple encoder rather than exposing a dedicated
# DataSequenceEncoder object.
# pylint: disable=abstract-method

GammaParameters = tuple[float, float]
GammaEncoded = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
GammaSuffStat = tuple[float, float, float]
Model = ProbabilityDistribution[Any, Any, Any]


class GammaDistribution(ProbabilityDistribution[float, GammaParameters, GammaEncoded]):
    """Gamma distribution parameterized by shape ``k`` and scale ``theta``."""

    def __init__(
        self,
        k: float,
        theta: float,
        name: Optional[str] = None,
        prior: Model = null_dist,
    ) -> None:
        """Initialize a gamma distribution.

        Args:
            k: Positive gamma shape.
            theta: Positive gamma scale.
            name: Optional model name.
            prior: Estimator metadata, neutral by default.

        Raises:
            ValueError: If shape or scale is not finite and positive.
        """
        super().__init__()
        self.set_parameters((k, theta))
        self.prior = prior
        self.name = name
        self.parents = []

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"GammaDistribution({self.k!r}, {self.theta!r}, name={self.name!r}, "
            f"prior={self.prior})"
        )

    def get_parameters(self) -> GammaParameters:
        """Return the ``(shape, scale)`` parameters."""
        return self.k, self.theta

    def set_parameters(self, value: GammaParameters) -> None:
        """Replace shape and scale and refresh the cached normalizer.

        Args:
            value: Positive ``(shape, scale)`` parameters.

        Raises:
            ValueError: If shape or scale is not finite and positive.
        """
        k, theta = value
        if not np.isfinite(k) or k <= 0 or not np.isfinite(theta) or theta <= 0:
            raise ValueError("Gamma shape and scale must be finite and positive.")
        self.k = float(k)
        self.theta = float(theta)
        self.log_const = float(-(gammaln(k) + k * np.log(theta)))

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]``.

        Gamma-to-gamma cross-entropy is analytic. Other distributions are
        integrated numerically over positive values.

        Args:
            dist: Distribution whose log-density is averaged.

        Returns:
            Cross-entropy from this distribution to ``dist``.
        """
        if isinstance(dist, GammaDistribution):
            expected_log_x = digamma(self.k) + np.log(self.theta)
            expected_x = self.k * self.theta
            value = (
                gammaln(dist.k)
                + dist.k * np.log(dist.theta)
                - (dist.k - 1) * expected_log_x
                + expected_x / dist.theta
            )
            return float(value)
        value, _ = scipy.integrate.quad(
            lambda x: -dist.log_density(x) * self.density(x), 0.0, np.inf
        )
        return float(value)

    def entropy(self) -> float:
        """Return the differential entropy for shape-scale parameters."""
        return float(
            self.k
            + np.log(self.theta)
            + gammaln(self.k)
            + (1 - self.k) * digamma(self.k)
        )

    def density(self, x: float) -> float:
        """Evaluate the density, returning zero outside ``x > 0``."""
        if x <= 0.0 or not np.isfinite(x):
            return 0.0
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density, returning ``-inf`` outside the support."""
        if x <= 0.0 or not np.isfinite(x):
            return float(-np.inf)
        return float(self.log_const + (self.k - 1.0) * np.log(x) - x / self.theta)

    def expected_log_density(self, x: float) -> float:
        """Return the plug-in log-density because parameters are fixed."""
        return self.log_density(x)

    def seq_log_density(self, x: GammaEncoded) -> np.ndarray[Any, Any]:
        """Evaluate log-densities from encoded values and their logarithms."""
        values, log_values = x
        result = values * (-1.0 / self.theta)
        if self.k != 1.0:
            result = result + log_values * (self.k - 1.0)
        result = np.asarray(result + self.log_const, dtype=float)
        return np.where(np.isfinite(values) & (values > 0.0), result, float(-np.inf))

    def seq_expected_log_density(self, x: GammaEncoded) -> np.ndarray[Any, Any]:
        """Return vectorized plug-in log-densities for encoded observations."""
        return self.seq_log_density(x)

    def seq_encode(self, x: Iterable[float]) -> GammaEncoded:
        """Encode values as arrays of values and logarithms."""
        values = np.asarray(tuple(x), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.log(values)
        return values, log_values

    def sampler(self, seed: Optional[int] = None) -> "GammaSampler":
        """Create a gamma sampler using an optional deterministic seed."""
        return GammaSampler(self, seed)

    def estimator(self) -> "GammaEstimator":
        """Create the legacy gamma maximum-likelihood estimator."""
        return GammaEstimator(name=self.name, prior=self.prior)


class GammaSampler(DistributionSampler[float]):
    """Draw independent values from a :class:`GammaDistribution`."""

    def __init__(self, dist: GammaDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one float or an array of ``size`` gamma values."""
        value = self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)
        return float(value) if size is None else value


class GammaAccumulator(StatisticAccumulator[float, GammaSuffStat, GammaEncoded]):
    """Accumulate count, value sum, and log-value sum for gamma estimation."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """Initialize empty sufficient statistics and an optional shared key."""
        self.nobs = 0.0
        self.sum = 0.0
        self.sum_of_logs = 0.0
        self.key = keys

    def initialize(self, x: float, weight: float, rng: RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def update(self, x: float, weight: float, estimate: Optional[Model]) -> None:
        """Accumulate one weighted positive observation."""
        del estimate
        self.nobs += weight
        self.sum += x * weight
        self.sum_of_logs += np.log(x) * weight

    def seq_update(
        self,
        x: GammaEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Accumulate an encoded sequence of weighted observations."""
        del estimate
        self.sum += float(np.dot(x[0], weights))
        self.sum_of_logs += float(np.dot(x[1], weights))
        self.nobs += float(np.sum(weights))

    def combine(self, suff_stat: GammaSuffStat) -> "GammaAccumulator":
        """Merge gamma sufficient statistics."""
        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]
        return self

    def value(self) -> GammaSuffStat:
        """Return ``(count, sum, sum_of_logs)``."""
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: GammaSuffStat) -> "GammaAccumulator":
        """Restore ``(count, sum, sum_of_logs)``."""
        self.nobs, self.sum, self.sum_of_logs = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge these statistics through the configured shared key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace these statistics from the configured shared key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())


class GammaAccumulatorFactory(
    StatisticAccumulatorFactory[float, GammaSuffStat, GammaEncoded]
):
    """Create gamma accumulators sharing one optional key."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """Store the key applied to each new accumulator."""
        self.keys = keys

    def make(self) -> GammaAccumulator:
        """Create an empty gamma accumulator."""
        return GammaAccumulator(self.keys)


class GammaEstimator(
    ParameterEstimator[float, GammaParameters, GammaEncoded, GammaSuffStat]
):
    """Estimate gamma shape and scale from sufficient statistics."""

    def __init__(
        self,
        pseudo_count: tuple[float, float] = (0.0, 0.0),
        suff_stat: tuple[float, float] = (1.0, 0.0),
        threshold: float = 1.0e-8,
        keys: Optional[str] = None,
        name: Optional[str] = None,
        prior: Model = null_dist,
    ) -> None:
        """Initialize regularization and numerical convergence settings."""
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name
        self.prior = prior

    def accumulator_factory(self) -> GammaAccumulatorFactory:
        """Create a factory for compatible sufficient-statistic accumulators."""
        return GammaAccumulatorFactory(self.keys)

    def get_prior(self) -> Model:
        """Return estimator metadata describing its prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace estimator prior metadata."""
        self.prior = prior

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> GammaDistribution:
        """Estimate a gamma distribution from either legacy call form.

        Args:
            *args: Either ``suff_stat`` or ignored ``nobs, suff_stat``.

        Returns:
            Estimated gamma distribution.

        Raises:
            TypeError: If neither supported call form is supplied.
        """
        if len(args) not in (1, 2):
            raise TypeError("estimate expects suff_stat or nobs, suff_stat")
        suff_stat = cast(GammaSuffStat, args[-1])
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] == 0:
            return GammaDistribution(1.0, 1.0, name=self.name, prior=self.prior)

        adjusted_sum = suff_stat[1] + ss1 * pc1
        adjusted_count = suff_stat[0] + pc1
        adjusted_mean = adjusted_sum / adjusted_count

        adjusted_log_sum = suff_stat[2] + ss2 * pc2
        adjusted_log_count = suff_stat[0] + pc2
        adjusted_log_mean = adjusted_log_sum / adjusted_log_count

        shape = self.estimate_shape(adjusted_mean, adjusted_log_mean, self.threshold)
        scale = adjusted_sum / (shape * adjusted_log_count)
        return GammaDistribution(shape, scale, name=self.name, prior=self.prior)

    @staticmethod
    def estimate_shape(
        avg_sum: float, avg_sum_of_logs: float, threshold: float
    ) -> float:
        """Solve the gamma shape equation by Newton iteration."""
        target = np.log(avg_sum) - avg_sum_of_logs
        old_shape = np.inf
        shape = (3 - target + np.sqrt((target - 3) ** 2 + 24 * target)) / (12 * target)
        while abs(old_shape - shape) > threshold:
            old_shape = shape
            shape -= (np.log(shape) - digamma(shape) - target) / (
                1 / shape - trigamma(shape)
            )
        return float(shape)
