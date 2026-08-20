"""Bayesian diagonal multivariate Gaussian likelihoods.

``DiagonalGaussianDistribution(mu, covariance)`` accepts equal-length vectors
of means and strictly positive diagonal variances. Its support is all finite
real vectors of that dimension. The default multivariate normal-gamma prior
models each coordinate mean and precision independently. Accumulators store
``(sum, sum_of_squares, count)``. Expected log-density averages over a
conjugate prior and otherwise uses the fixed likelihood parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional

import numpy as np

from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
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
DiagonalGaussianDatum = Sequence[float] | Array
DiagonalGaussianParameters = tuple[Array, Array]
DiagonalGaussianEncoded = tuple[Array, Array]
DiagonalGaussianSuffStat = tuple[Optional[Array], Optional[Array], float]
Model = ProbabilityDistribution[Any, Any, Any]


def _default_prior(dim: int) -> MultivariateNormalGammaDistribution:
    """Create the weak default normal-gamma prior for ``dim`` coordinates."""
    return MultivariateNormalGammaDistribution(
        np.zeros(dim),
        np.full(dim, 1.0e-8),
        np.full(dim, 0.500001),
        np.ones(dim),
    )


class DiagonalGaussianDistribution(
    ProbabilityDistribution[
        DiagonalGaussianDatum, DiagonalGaussianParameters, DiagonalGaussianEncoded
    ]
):
    """Multivariate Gaussian likelihood with diagonal covariance."""

    def __init__(
        self,
        mu: Sequence[float] | Array,
        covariance: Sequence[float] | Array,
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize mean, diagonal variance, metadata, and prior."""
        super().__init__()
        self.set_parameters(
            (np.asarray(mu, dtype=np.float64), np.asarray(covariance, dtype=np.float64))
        )
        self.name = name
        self.set_prior(_default_prior(self.dim) if prior is None else prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"DiagonalGaussianDistribution({self.mu.tolist()!r}, "
            f"{self.covar.tolist()!r}, name={self.name!r}, prior={self.prior})"
        )

    def get_prior(self) -> Model:
        """Return the current multivariate normal-gamma or alternate prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache expected natural parameters."""
        self.prior = prior
        if isinstance(prior, MultivariateNormalGammaDistribution):
            mu, lam, shape, rate = prior.get_parameters()
            if len(mu) != self.dim:
                raise ValueError("Diagonal Gaussian prior dimension must match mean.")
            expected_const = np.sum(
                0.5 * mu * mu * shape / rate
                + 0.5 / lam
                + 0.5 * (np.log(rate) - digamma(shape))
            )
            self.expected_nparams: Optional[tuple[float, float, Array, Array]] = (
                float(expected_const),
                float(-0.5 * np.log(2.0 * np.pi) * self.dim),
                np.asarray(mu * shape / rate, dtype=np.float64),
                np.asarray(-0.5 * shape / rate, dtype=np.float64),
            )
        else:
            self.expected_nparams = None

    def get_parameters(self) -> DiagonalGaussianParameters:
        """Return mean and diagonal variance arrays."""
        return self.mu, self.covar

    def set_parameters(self, value: DiagonalGaussianParameters) -> None:
        """Replace mean and diagonal variance and refresh cached terms."""
        mu, covariance = value
        mean = np.asarray(mu, dtype=np.float64)
        variances = np.asarray(covariance, dtype=np.float64)
        if mean.ndim != 1 or variances.ndim != 1 or mean.shape != variances.shape:
            raise ValueError("Mean and covariance must be equal-length vectors.")
        if mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("Diagonal Gaussian mean must be finite and nonempty.")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("Diagonal Gaussian variances must be finite and positive.")
        self.dim = int(mean.size)
        self.mu = mean
        self.covar = variances
        self.log_c = float(
            -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())
        )
        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = float((-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c)

    def _observation(self, x: DiagonalGaussianDatum) -> Optional[Array]:
        """Return a valid observation array or ``None`` outside support."""
        value = np.asarray(x, dtype=np.float64)
        if value.shape != (self.dim,) or not np.all(np.isfinite(value)):
            return None
        return value

    def log_density(self, x: DiagonalGaussianDatum) -> float:
        """Evaluate log-density, returning ``-inf`` outside vector support."""
        value = self._observation(x)
        if value is None:
            return float(-np.inf)
        return float(np.dot(value * value, self.ca) + np.dot(value, self.cb) + self.cc)

    def expected_log_density(self, x: DiagonalGaussianDatum) -> float:
        """Evaluate prior-averaged log-density when conjugate."""
        value = self._observation(x)
        if value is None:
            return float(-np.inf)
        if self.expected_nparams is None:
            return self.log_density(value)
        expected_const, base_const, linear, quadratic = self.expected_nparams
        return float(
            np.dot(value, linear)
            + np.dot(value * value, quadratic)
            - expected_const
            + base_const
        )

    def seq_log_density(self, x: DiagonalGaussianEncoded) -> Array:
        """Evaluate fixed-parameter log-densities from encoded observations."""
        values, squares = x
        result = np.dot(squares, self.ca) + np.dot(values, self.cb) + self.cc
        valid = np.all(np.isfinite(values), axis=1)
        return np.where(valid, result, -np.inf).astype(float)

    def seq_expected_log_density(self, x: DiagonalGaussianEncoded) -> Array:
        """Evaluate prior-averaged log-densities from encoded observations."""
        if self.expected_nparams is None:
            return self.seq_log_density(x)
        values, squares = x
        expected_const, base_const, linear, quadratic = self.expected_nparams
        result = (
            np.dot(values, linear)
            + np.dot(squares, quadratic)
            - expected_const
            + base_const
        )
        valid = np.all(np.isfinite(values), axis=1)
        return np.where(valid, result, -np.inf).astype(float)

    def seq_encode(self, x: Iterable[DiagonalGaussianDatum]) -> DiagonalGaussianEncoded:
        """Encode observations as values and coordinate-wise squares."""
        values = np.asarray(tuple(x), dtype=np.float64).reshape((-1, self.dim))
        return values, values * values

    def sampler(self, seed: Optional[int] = None) -> "DiagonalGaussianSampler":
        """Create a repeatable diagonal Gaussian sampler."""
        return DiagonalGaussianSampler(self, seed)

    def estimator(self) -> "DiagonalGaussianEstimator":
        """Create an estimator retaining dimension, name, and prior."""
        return DiagonalGaussianEstimator(dim=self.dim, name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Create a sequence encoder for this dimensionality."""
        return DiagonalGaussianDataEncoder(self.dim)


class DiagonalGaussianSampler(DistributionSampler[DiagonalGaussianDatum]):
    """Draw independent vectors from a diagonal Gaussian likelihood."""

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one vector or a list of ``size`` vectors."""
        shape = self.dist.dim if size is None else (size, self.dist.dim)
        value = (
            self.rng.standard_normal(size=shape) * np.sqrt(self.dist.covar)
            + self.dist.mu
        )
        return value.tolist()


class DiagonalGaussianAccumulator(
    StatisticAccumulator[
        DiagonalGaussianDatum, DiagonalGaussianSuffStat, DiagonalGaussianEncoded
    ]
):
    """Accumulate coordinate sums, squared sums, and observation count."""

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize empty statistics, optionally with known dimension."""
        self.dim = dim
        self.count = 0.0
        self.sum = None if dim is None else np.zeros(dim, dtype=np.float64)
        self.sum2 = None if dim is None else np.zeros(dim, dtype=np.float64)

    def _ensure_dimension(self, dim: int) -> None:
        """Allocate statistic arrays on the first observation."""
        if self.dim is None:
            self.dim = dim
            self.sum = np.zeros(dim, dtype=np.float64)
            self.sum2 = np.zeros(dim, dtype=np.float64)
        elif self.dim != dim:
            raise ValueError("Observation dimension does not match accumulator.")

    def update(
        self,
        x: DiagonalGaussianDatum,
        weight: float,
        estimate: Optional[Model],
    ) -> None:
        """Accumulate one weighted vector."""
        del estimate
        value = np.asarray(x, dtype=np.float64)
        self._ensure_dimension(len(value))
        assert self.sum is not None and self.sum2 is not None
        self.count += weight
        self.sum += value * weight
        self.sum2 += value * value * weight

    def initialize(
        self,
        x: DiagonalGaussianDatum,
        weight: float,
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate one vector during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: DiagonalGaussianEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Accumulate encoded vectors with corresponding weights."""
        del estimate
        self._ensure_dimension(x[0].shape[1])
        assert self.sum is not None and self.sum2 is not None
        self.count += float(weights.sum())
        self.sum += np.dot(x[0].T, weights)
        self.sum2 += np.dot(x[1].T, weights)

    def seq_initialize(
        self,
        x: DiagonalGaussianEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded vectors during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: DiagonalGaussianSuffStat
    ) -> "DiagonalGaussianAccumulator":
        """Merge ``(sum, sum_of_squares, count)`` statistics."""
        other_sum, other_sum2, other_count = suff_stat
        if other_sum is None or other_sum2 is None:
            return self
        self._ensure_dimension(len(other_sum))
        assert self.sum is not None and self.sum2 is not None
        self.sum += other_sum
        self.sum2 += other_sum2
        self.count += other_count
        return self

    def value(self) -> DiagonalGaussianSuffStat:
        """Return sums, squared sums, and weighted count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: DiagonalGaussianSuffStat) -> "DiagonalGaussianAccumulator":
        """Restore sums, squared sums, and weighted count."""
        self.sum, self.sum2, self.count = x
        self.dim = None if self.sum is None else len(self.sum)
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged; this accumulator has no keys."""
        del stats_dict

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged; this accumulator has no keys."""
        del stats_dict

    def acc_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Create the compatible dimensional encoder."""
        if self.dim is None:
            raise ValueError("Accumulator dimension is unknown before an update.")
        return DiagonalGaussianDataEncoder(self.dim)


class DiagonalGaussianAccumulatorFactory(
    StatisticAccumulatorFactory[
        DiagonalGaussianDatum, DiagonalGaussianSuffStat, DiagonalGaussianEncoded
    ]
):
    """Create diagonal Gaussian accumulators of a fixed dimension."""

    def __init__(self, dim: Optional[int]) -> None:
        """Store the optional dimension copied into each accumulator."""
        self.dim = dim

    def make(self) -> DiagonalGaussianAccumulator:
        """Create an empty diagonal Gaussian accumulator."""
        return DiagonalGaussianAccumulator(self.dim)


class DiagonalGaussianEstimator(
    ParameterEstimator[
        DiagonalGaussianDatum,
        DiagonalGaussianParameters,
        DiagonalGaussianEncoded,
        DiagonalGaussianSuffStat,
    ]
):
    """Estimate diagonal Gaussian parameters and update their posterior."""

    def __init__(
        self,
        dim: Optional[int] = None,
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize dimension, metadata, and optional conjugate prior."""
        if prior is None and dim is not None:
            prior = _default_prior(dim)
        self.dim = dim
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self) -> DiagonalGaussianAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return DiagonalGaussianAccumulatorFactory(self.dim)

    def set_prior(self, prior: Optional[Model]) -> None:
        """Replace the estimator prior and update its conjugacy flag."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)

    def get_prior(self) -> Any:
        """Return the estimator prior."""
        return self.prior

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> DiagonalGaussianDistribution:
        """Estimate from coordinate sums, squared sums, and count."""
        sum_x, sum_xx, count = args[-1]
        if sum_x is None or sum_xx is None:
            raise ValueError("Diagonal Gaussian estimation requires observations.")
        if self.has_conj_prior:
            assert isinstance(self.prior, MultivariateNormalGammaDistribution)
            old_mu, old_lam, old_shape, old_rate = self.prior.get_parameters()
            new_lam = old_lam + count
            new_shape = old_shape + count / 2.0
            sample_mean = sum_x / count if count > 0.0 else np.zeros_like(sum_x)
            new_mu = (sum_x + old_mu * old_lam) / new_lam
            centered_sum = sum_xx - sample_mean * sum_x
            mean_shift = old_lam * count / new_lam * (sample_mean - old_mu) ** 2
            new_rate = old_rate + 0.5 * (centered_sum + mean_shift)
            new_variance = new_rate / (new_shape - 0.5)
            posterior = MultivariateNormalGammaDistribution(
                new_mu, new_lam, new_shape, new_rate
            )
            return DiagonalGaussianDistribution(
                new_mu, new_variance, name=self.name, prior=posterior
            )
        if count <= 0.0:
            raise ValueError("Diagonal Gaussian estimation requires positive weight.")
        mean = sum_x / count
        variance = sum_xx / count - mean * mean
        return DiagonalGaussianDistribution(
            mean, variance, name=self.name, prior=self.prior
        )


class DiagonalGaussianDataEncoder(
    DataSequenceEncoder[DiagonalGaussianDatum, DiagonalGaussianEncoded]
):
    """Encode vectors as values and coordinate-wise squares."""

    def __init__(self, dim: int) -> None:
        """Store the required observation dimension."""
        self.dim = dim

    def __str__(self) -> str:
        """Return a stable dimensional encoder representation."""
        return f"DiagonalGaussianDataEncoder(dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same dimension."""
        return isinstance(other, DiagonalGaussianDataEncoder) and self.dim == other.dim

    def seq_encode(
        self, x: Iterable[DiagonalGaussianDatum]
    ) -> "DiagonalGaussianEncodedData":
        """Encode observations in a typed container."""
        values = np.asarray(tuple(x), dtype=np.float64).reshape((-1, self.dim))
        return DiagonalGaussianEncodedData((values, values * values))


class DiagonalGaussianEncodedData(EncodedDataSequence[DiagonalGaussianEncoded]):
    """Contain an encoded diagonal Gaussian sequence."""

    def __init__(self, data: DiagonalGaussianEncoded) -> None:
        """Store value and squared-value arrays."""
        super().__init__(data)
