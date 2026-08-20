"""Bayesian Poisson likelihoods for nonnegative integer counts.

``PoissonDistribution(lam)`` has support on integers ``x >= 0`` and returns
``-inf`` for all other observations. The default gamma prior uses shape
``1.0001`` and scale ``1e6``. Accumulators store weighted ``(count, sum)``
statistics. Under a gamma prior, expected log-density integrates both
``log(lam)`` and ``lam``; without a conjugate prior it uses plug-in scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.nulldist import NullDistribution, null_dist
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma, gammaln, stirling2

# Legacy bstats implementations are intentionally concrete protocol classes.
# pylint: disable=abstract-method

Array = np.ndarray[Any, np.dtype[np.float64]]
PoissonEncoded = tuple[Array, Array]
PoissonSuffStat = tuple[float, float]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = GammaDistribution(1.0001, 1.0e6)


class PoissonDistribution(ProbabilityDistribution[int, float, PoissonEncoded]):
    """Poisson count likelihood with a gamma parameter prior."""

    def __init__(
        self,
        lam: float,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a positive rate, metadata, and prior."""
        super().__init__()
        self.name = name
        self.keys = keys
        self.set_parameters(lam)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"PoissonDistribution({self.lam!r}, name={self.name!r}, "
            f"prior={self.prior}, keys={self.keys!r})"
        )

    def get_parameters(self) -> float:
        """Return the Poisson rate."""
        return self.lam

    def set_parameters(self, value: float) -> None:
        """Replace the positive finite rate and cached logarithm."""
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Poisson rate must be finite and positive.")
        self.lam = float(value)
        self.log_lambda = float(np.log(self.lam))

    def get_prior(self) -> Model:
        """Return the current gamma or alternate prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache gamma hyperparameters."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, GammaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)
        self.conj_prior_params: Optional[tuple[float, float]] = (
            prior.get_parameters() if isinstance(prior, GammaDistribution) else None
        )

    def get_data_type(self) -> type[int]:
        """Return the intended observation type."""
        return int

    @staticmethod
    def _in_support(x: Any) -> bool:
        """Return whether ``x`` is a finite nonnegative integer."""
        return bool(
            isinstance(x, (int, np.integer))
            and not isinstance(x, (bool, np.bool_))
            and x >= 0
        )

    def log_density(self, x: int) -> float:
        """Evaluate log mass, returning ``-inf`` outside count support."""
        if not self._in_support(x):
            return float(-np.inf)
        return float(x * self.log_lambda - gammaln(x + 1.0) - self.lam)

    def expected_log_density(self, x: int) -> float:
        """Evaluate prior-averaged log mass when the prior is gamma."""
        if not self._in_support(x):
            return float(-np.inf)
        if self.conj_prior_params is None:
            return self.log_density(x)
        shape, scale = self.conj_prior_params
        return float(
            (digamma(shape) + np.log(scale)) * x - shape * scale - gammaln(x + 1.0)
        )

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` for another Poisson likelihood."""
        if not isinstance(dist, PoissonDistribution):
            return super().cross_entropy(dist)
        return float(
            self.entropy()
            + dist.lam
            - self.lam
            + self.lam * (self.log_lambda - dist.log_lambda)
        )

    def entropy(self) -> float:
        """Return the Poisson Shannon entropy."""
        return float(poisson.entropy(self.lam))

    def moment(self, order: int) -> float:
        """Return a raw integer moment using Stirling numbers."""
        if order == 0:
            return 1.0
        return float(
            sum(
                np.power(self.lam, index) * stirling2(order, index)
                for index in range(1, order + 1)
            )
        )

    def seq_log_density(self, x: PoissonEncoded) -> np.ndarray[Any, Any]:
        """Evaluate log masses from encoded values and log-factorials."""
        values, log_factorials = x
        result = values * self.log_lambda - log_factorials - self.lam
        valid = np.isfinite(values) & (values >= 0.0) & (values == np.floor(values))
        return np.where(valid, result, -np.inf).astype(float)

    def seq_expected_log_density(self, x: PoissonEncoded) -> np.ndarray[Any, Any]:
        """Evaluate prior-averaged log masses for encoded counts."""
        if self.conj_prior_params is None:
            return self.seq_log_density(x)
        values, log_factorials = x
        shape, scale = self.conj_prior_params
        result = (
            (digamma(shape) + np.log(scale)) * values - shape * scale - log_factorials
        )
        valid = np.isfinite(values) & (values >= 0.0) & (values == np.floor(values))
        return np.where(valid, result, -np.inf).astype(float)

    def seq_encode(self, x: Iterable[int]) -> PoissonEncoded:
        """Encode count values and their log-factorials."""
        values = np.asarray(tuple(x), dtype=np.float64)
        return values, np.asarray(gammaln(values + 1.0), dtype=np.float64)

    def sampler(self, seed: Optional[int] = None) -> "PoissonSampler":
        """Create a repeatable Poisson sampler."""
        return PoissonSampler(self, seed)

    def estimator(self) -> "PoissonEstimator":
        """Create an estimator retaining metadata and the current prior."""
        return PoissonEstimator(name=self.name, keys=self.keys, prior=self.prior)

    def dist_to_encoder(self) -> "PoissonDataEncoder":
        """Create the Poisson sequence encoder."""
        return PoissonDataEncoder()


class PoissonSampler(DistributionSampler[int]):
    """Draw independent nonnegative Poisson counts."""

    def __init__(self, dist: PoissonDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one integer or an array of ``size`` integers."""
        value = self.rng.poisson(lam=self.dist.lam, size=size)
        return int(value) if size is None else value


class PoissonEstimatorAccumulator(
    StatisticAccumulator[int, PoissonSuffStat, PoissonEncoded]
):
    """Accumulate weighted observation count and count sum."""

    def __init__(self, name: Optional[str], keys: Optional[str]) -> None:
        """Initialize empty statistics and sharing metadata."""
        self.name = name
        self.key = keys
        self.sum = 0.0
        self.count = 0.0

    def initialize(self, x: int, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: PoissonEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def update(self, x: int, weight: float, estimate: Optional[Model]) -> None:
        """Add one weighted nonnegative count."""
        del estimate
        if not PoissonDistribution._in_support(x):  # pylint: disable=protected-access
            raise ValueError("Poisson statistics require nonnegative integer counts.")
        self.sum += x * weight
        self.count += weight

    def seq_update(
        self,
        x: PoissonEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Add encoded counts with corresponding weights."""
        del estimate
        values = x[0]
        if np.any((values < 0.0) | (values != np.floor(values))):
            raise ValueError("Poisson statistics require nonnegative integer counts.")
        self.sum += float(np.dot(values, weights))
        self.count += float(weights.sum())

    def combine(self, suff_stat: PoissonSuffStat) -> "PoissonEstimatorAccumulator":
        """Merge ``(observation_count, count_sum)`` statistics."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> PoissonSuffStat:
        """Return ``(observation_count, count_sum)``."""
        return self.count, self.sum

    def from_value(self, x: PoissonSuffStat) -> "PoissonEstimatorAccumulator":
        """Restore ``(observation_count, count_sum)``."""
        self.count, self.sum = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics through the configured sharing key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace statistics through the configured sharing key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "PoissonDataEncoder":
        """Create the compatible Poisson encoder."""
        return PoissonDataEncoder()


class PoissonEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[int, PoissonSuffStat, PoissonEncoded]
):
    """Create Poisson accumulators with shared metadata."""

    def __init__(self, name: Optional[str], keys: Optional[str]) -> None:
        """Store metadata copied into each accumulator."""
        self.name = name
        self.keys = keys

    def make(self) -> PoissonEstimatorAccumulator:
        """Create an empty Poisson accumulator."""
        return PoissonEstimatorAccumulator(self.name, self.keys)


class PoissonEstimator(ParameterEstimator[int, float, PoissonEncoded, PoissonSuffStat]):
    """Estimate a Poisson rate and update its gamma posterior."""

    def __init__(
        self,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize estimator metadata and prior."""
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self) -> PoissonEstimatorAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return PoissonEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior: Model) -> None:
        """Replace the estimator prior and update prior flags."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, GammaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return self.prior

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> PoissonDistribution:
        """Estimate from ``(observation_count, count_sum)`` statistics."""
        nobs, count_sum = args[-1]
        if self.has_conj_prior:
            assert isinstance(self.prior, GammaDistribution)
            shape, scale = self.prior.get_parameters()
            posterior_shape = shape + count_sum
            posterior_scale = scale / (nobs * scale + 1.0)
            rate = max(
                (posterior_shape - 1.0) * posterior_scale,
                np.finfo(float).tiny,
            )
            return PoissonDistribution(
                rate,
                name=self.name,
                prior=GammaDistribution(posterior_shape, posterior_scale),
                keys=self.keys,
            )
        if self.has_prior:

            def objective(value: float) -> float:
                return float(
                    -count_sum * np.log(value)
                    + nobs * value
                    - self.prior.log_density(value)
                )

            upper = max(count_sum / nobs if nobs else 1.0, 1.0) * 10.0
            solution = minimize_scalar(
                objective,
                bounds=(np.finfo(float).tiny, upper),
                method="bounded",
            )
            return PoissonDistribution(
                float(solution.x),
                name=self.name,
                prior=self.prior,
                keys=self.keys,
            )
        rate = count_sum / nobs if nobs else np.finfo(float).tiny
        return PoissonDistribution(
            max(rate, np.finfo(float).tiny),
            name=self.name,
            prior=null_dist,
            keys=self.keys,
        )


class PoissonDataEncoder(DataSequenceEncoder[int, PoissonEncoded]):
    """Encode Poisson counts and cached log-factorials."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "PoissonDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has Poisson semantics."""
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(self, x: Iterable[int]) -> "PoissonEncodedData":
        """Encode observations in a typed container."""
        values = np.asarray(tuple(x), dtype=np.float64)
        return PoissonEncodedData((values, gammaln(values + 1.0)))


class PoissonEncodedData(EncodedDataSequence[PoissonEncoded]):
    """Contain Poisson values and cached log-factorials."""

    def __init__(self, data: PoissonEncoded) -> None:
        """Store encoded Poisson data."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"PoissonEncodedData(data={self.data!r})"
