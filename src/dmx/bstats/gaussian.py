"""Bayesian univariate Gaussian likelihoods.

``GaussianDistribution(mu, sigma2)`` uses a mean and strictly positive
variance on the real line. Its default normal-gamma prior models the mean and
precision ``1 / sigma2``. Accumulators retain the legacy five statistics used
for separately shared location and variance estimates. Expected log-density
is averaged under a normal-gamma prior and otherwise uses plug-in parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np
import pandas as pd

from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma

# Legacy bstats implementations are concrete protocol classes.
# pylint: disable=abstract-method,too-few-public-methods

Array = np.ndarray[Any, np.dtype[np.float64]]
GaussianParameters = tuple[float, float]
GaussianSuffStat = tuple[float, float, float, float, float]
GaussianKeys = tuple[Optional[str], Optional[str]]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


class GaussianDistribution(ProbabilityDistribution[float, GaussianParameters, Array]):
    """Represent a univariate Gaussian likelihood by mean and variance.

    ``mu`` is finite and ``sigma2`` is a strictly positive finite variance. A
    normal-gamma prior over mean and precision enables posterior-expected
    scoring; another prior uses the fixed likelihood parameters.
    """

    def __init__(
        self,
        mu: float,
        sigma2: float,
        name: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize finite Gaussian parameters, metadata, and prior."""
        super().__init__()
        self.set_parameters((mu, sigma2))
        self.set_prior(prior)
        self.set_name(name)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"GaussianDistribution({self.mu!r}, {self.sigma2!r}, "
            f"name={self.name!r}, prior={self.prior})"
        )

    def get_parameters(self) -> GaussianParameters:
        """Return ``(mean, variance)``."""
        return self.mu, self.sigma2

    def set_parameters(self, value: GaussianParameters) -> None:
        """Replace the finite mean and positive finite variance."""
        mu, sigma2 = value
        if not np.isfinite(mu):
            raise ValueError("Gaussian mean must be finite.")
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            raise ValueError("Gaussian variance must be finite and positive.")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.logConst = float(-0.5 * np.log(2.0 * np.pi * self.sigma2))
        self.const = float(1.0 / np.sqrt(2.0 * np.pi * self.sigma2))

    def get_prior(self) -> Model:
        """Return the current normal-gamma or alternate prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache expected natural parameters."""
        self.prior = prior
        if isinstance(prior, NormalGammaDistribution):
            mu, lam, shape, rate = prior.get_parameters()
            expected_const = (
                0.5 * mu * mu * shape / rate
                + 0.5 / lam
                + 0.5 * (np.log(rate) - digamma(shape))
            )
            self.expected_nparams: Optional[tuple[float, float, float, float]] = (
                float(expected_const),
                float(-0.5 * np.log(2.0 * np.pi)),
                float(mu * shape / rate),
                float(-0.5 * shape / rate),
            )
        else:
            self.expected_nparams = None

    def log_density(self, x: float) -> float:
        """Evaluate the Gaussian log-density on the real line."""
        if not np.isfinite(x):
            return float(-np.inf)
        return float(self.logConst - 0.5 * (x - self.mu) ** 2 / self.sigma2)

    def expected_log_density(self, x: float) -> float:
        """Evaluate prior-averaged log-density when conjugate."""
        if not np.isfinite(x):
            return float(-np.inf)
        if self.expected_nparams is None:
            return self.log_density(x)
        expected_const, base_const, linear, quadratic = self.expected_nparams
        return float(x * (linear + x * quadratic) - expected_const + base_const)

    def seq_encode(self, x: Iterable[float]) -> Array:
        """Encode ``n`` observations as a float array of shape ``(n,)``."""
        return np.asarray(tuple(x), dtype=np.float64)

    def seq_log_density(self, x: Array) -> Array:
        """Evaluate fixed-parameter log-densities for encoded observations."""
        result = self.logConst - 0.5 * (x - self.mu) ** 2 / self.sigma2
        return np.where(np.isfinite(x), result, -np.inf).astype(float)

    def seq_expected_log_density(self, x: Array) -> Array:
        """Evaluate prior-averaged log-densities for encoded observations."""
        if self.expected_nparams is None:
            return self.seq_log_density(x)
        expected_const, base_const, linear, quadratic = self.expected_nparams
        result = x * (linear + x * quadratic) - expected_const + base_const
        return np.where(np.isfinite(x), result, -np.inf).astype(float)

    def sampler(self, seed: Optional[int] = None) -> "GaussianSampler":
        """Create a repeatable Gaussian sampler."""
        return GaussianSampler(self, seed)

    def estimator(self) -> "GaussianEstimator":
        """Create an estimator retaining the current name and prior."""
        return GaussianEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "GaussianDataEncoder":
        """Create the Gaussian sequence encoder."""
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler[float]):
    """Draw independent Gaussian observations."""

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one float or an array of ``size`` observations."""
        value = self.rng.normal(self.dist.mu, np.sqrt(self.dist.sigma2), size=size)
        return float(value) if size is None else value


class GaussianAccumulator(StatisticAccumulator[float, GaussianSuffStat, Array]):
    """Accumulate legacy location and variance sufficient statistics.

    The tuple is ``(sum_x, sum_x_squared, variance_sum_x, mean_weight,
    variance_weight)``. Separate keys allow mean and variance statistics to be
    shared independently between model components.
    """

    def __init__(
        self, name: Optional[str] = None, keys: GaussianKeys = (None, None)
    ) -> None:
        """Initialize empty statistics and optional sharing metadata."""
        self.name = name
        self.sum = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.sum_key, self.sum2_key = keys

    def initialize(self, x: float, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def update(self, x: float, weight: float, estimate: Optional[Model]) -> None:
        """Accumulate one weighted value for location and variance."""
        del estimate
        weighted = x * weight
        self.sum += weighted
        self.sum2 += x * weighted
        self.sum3 += weighted
        self.count += weight
        self.count2 += weight

    def seq_initialize(
        self, x: Array, weights: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def seq_update(
        self, x: Array, weights: np.ndarray[Any, Any], estimate: Optional[Model]
    ) -> None:
        """Accumulate encoded observations with corresponding weights."""
        del estimate
        weighted_sum = float(np.dot(x, weights))
        self.sum += weighted_sum
        self.sum2 += float(np.dot(x * x, weights))
        self.sum3 += weighted_sum
        weight_sum = float(weights.sum())
        self.count += weight_sum
        self.count2 += weight_sum

    def df_initialize(
        self, df: pd.DataFrame, weights: pd.Series, rng: np.random.RandomState
    ) -> None:
        """Accumulate a named DataFrame column during initialization."""
        del rng
        self.df_update(df, weights, None)

    def df_update(
        self, df: pd.DataFrame, weights: pd.Series, estimate: Optional[Model]
    ) -> None:
        """Accumulate a named DataFrame column and corresponding weights."""
        del estimate
        if self.name is None:
            raise ValueError("Gaussian accumulator requires a DataFrame column name.")
        column = df[self.name]
        weight_values = np.asarray(weights, dtype=np.float64)
        values = column.to_numpy(dtype=np.float64)
        weighted_sum = float(np.dot(values, weight_values))
        self.sum += weighted_sum
        self.sum2 += float(np.dot(values * values, weight_values))
        self.sum3 += weighted_sum
        weight_sum = float(weight_values.sum())
        self.count += weight_sum
        self.count2 += weight_sum

    def combine(self, suff_stat: GaussianSuffStat) -> "GaussianAccumulator":
        """Merge five legacy sufficient statistics."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        self.count2 += suff_stat[4]
        return self

    def value(self) -> GaussianSuffStat:
        """Return the five location and variance statistics."""
        return self.sum, self.sum2, self.sum3, self.count, self.count2

    def from_value(self, x: GaussianSuffStat) -> "GaussianAccumulator":
        """Restore the five location and variance statistics."""
        self.sum, self.sum2, self.sum3, self.count, self.count2 = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge location and variance statistics through shared keys."""
        if self.sum_key is not None:
            if self.sum_key in stats_dict:
                count, total = stats_dict[self.sum_key]
                stats_dict[self.sum_key] = (count + self.count, total + self.sum)
            else:
                stats_dict[self.sum_key] = (self.count, self.sum)
        if self.sum2_key is not None:
            if self.sum2_key in stats_dict and self.count2 > 0.0:
                old_count, old_sum2, old_sum = stats_dict[self.sum2_key]
                mean = self.sum3 / self.count2
                old_mean = 0.0 if old_count == 0.0 else old_sum / old_count
                merged_mean = (self.sum3 + old_sum) / (self.count2 + old_count)
                centered = self.sum2 - mean * self.sum3
                merged_sum2 = (
                    old_sum2
                    - old_mean * old_sum
                    + centered
                    + merged_mean * (self.sum3 + old_sum)
                )
                stats_dict[self.sum2_key] = (
                    old_count + self.count2,
                    merged_sum2,
                    old_sum + self.sum3,
                )
            else:
                stats_dict[self.sum2_key] = (self.count2, self.sum2, self.sum3)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace location and variance statistics through shared keys."""
        if self.sum_key is not None and self.sum_key in stats_dict:
            self.count, self.sum = stats_dict[self.sum_key]
        if self.sum2_key is not None and self.sum2_key in stats_dict:
            self.count2, self.sum2, self.sum3 = stats_dict[self.sum2_key]

    def acc_to_encoder(self) -> "GaussianDataEncoder":
        """Create the compatible Gaussian encoder."""
        return GaussianDataEncoder()


class GaussianEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[float, GaussianSuffStat, Array]
):
    """Create Gaussian accumulators with shared metadata."""

    def __init__(self, name: Optional[str], keys: GaussianKeys) -> None:
        """Store metadata copied into each accumulator."""
        self.name = name
        self.keys = keys

    def make(self) -> GaussianAccumulator:
        """Create an empty Gaussian accumulator."""
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(
    ParameterEstimator[float, GaussianParameters, Array, GaussianSuffStat]
):
    """Estimate Gaussian parameters from weighted first and second moments.

    A normal-gamma prior is updated conjugately and attached as the posterior
    of the returned likelihood. Otherwise, separate weighted maximum-likelihood
    estimates are used for the mean and variance.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: GaussianKeys = (None, None),
    ) -> None:
        """Initialize estimator metadata and prior."""
        self.keys = keys
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self) -> GaussianEstimatorAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return GaussianEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior: Model) -> None:
        """Replace the estimator prior and update its conjugacy flag."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return self.prior

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> GaussianDistribution:
        """Estimate from the five legacy Gaussian sufficient statistics.

        Args:
            *args: Either the statistic tuple alone or legacy ``nobs`` followed
                by it. ``nobs`` is ignored. The tuple contains ``sum_x``,
                ``sum_x_squared``, a separately shareable copy of ``sum_x``,
                mean weight, and variance weight.

        Returns:
            Fitted Gaussian likelihood carrying an updated normal-gamma
            posterior when the prior is conjugate.
        """
        sum_x, sum_xx, sum_xxx, count_mean, count_variance = args[-1]
        if self.has_conj_prior:
            assert isinstance(self.prior, NormalGammaDistribution)
            old_mu, old_lam, old_shape, old_rate = self.prior.get_parameters()
            new_lam = old_lam + count_mean
            new_shape = old_shape + count_variance / 2.0
            sample_mean1 = sum_x / count_mean if count_mean > 0.0 else 0.0
            sample_mean2 = sum_xxx / count_variance if count_variance > 0.0 else 0.0
            new_mu = (sum_x + old_mu * old_lam) / new_lam
            centered_sum = sum_xx - sample_mean2 * sum_xxx
            mean_shift = old_lam * count_mean / new_lam * (sample_mean1 - old_mu) ** 2
            new_rate = old_rate + 0.5 * (centered_sum + mean_shift)
            new_variance = new_rate / (new_shape - 0.5)
            posterior = NormalGammaDistribution(new_mu, new_lam, new_shape, new_rate)
            return GaussianDistribution(
                new_mu, new_variance, name=self.name, prior=posterior
            )
        mean = sum_x / count_mean if count_mean > 0.0 else 0.0
        second_mean = sum_xxx / count_variance if count_variance > 0.0 else 0.0
        variance = sum_xx / count_variance - second_mean**2
        return GaussianDistribution(mean, variance, name=self.name, prior=self.prior)


class GaussianDataEncoder(DataSequenceEncoder[float, Array]):
    """Encode Gaussian observations as floating-point arrays."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "GaussianDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has Gaussian semantics."""
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: Iterable[float]) -> "GaussianEncodedData":
        """Encode observations in a typed container."""
        return GaussianEncodedData(np.asarray(tuple(x), dtype=np.float64))


class GaussianEncodedData(EncodedDataSequence[Array]):
    """Contain an encoded Gaussian sequence."""

    def __init__(self, data: Array) -> None:
        """Store the floating-point array."""
        super().__init__(data)
