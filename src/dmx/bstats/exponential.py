"""Bayesian exponential likelihoods for nonnegative real observations.

``ExponentialDistribution(lam)`` uses the rate parameterization, so its mean is
``1 / lam`` and its support is ``x >= 0``. The default gamma prior has shape
``1.0001`` and scale ``1e6``. Accumulators store weighted ``(count, sum)``
statistics. Under a gamma prior, expected log-density integrates ``log(lam)``
and ``lam``; with another prior it falls back to fixed-parameter scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np

from dmx.bstats.gamma import GammaDistribution
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
ExponentialSuffStat = tuple[float, float]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = GammaDistribution(1.0001, 1.0e6)


class ExponentialDistribution(ProbabilityDistribution[float, float, Array]):
    """Exponential likelihood parameterized by a positive rate."""

    def __init__(
        self,
        lam: float,
        name: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize a positive rate, optional name, and parameter prior."""
        super().__init__()
        self.name = name
        self.set_parameters(lam)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"ExponentialDistribution({self.lam!r}, name={self.name!r}, "
            f"prior={self.prior})"
        )

    def get_parameters(self) -> float:
        """Return the exponential rate."""
        return self.lam

    def set_parameters(self, value: float) -> None:
        """Replace the positive finite rate and refresh its logarithm."""
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Exponential rate must be finite and positive.")
        self.lam = float(value)
        self.log_lam = float(np.log(self.lam))
        self.params = self.lam

    def get_prior(self) -> Model:
        """Return the current gamma or alternate prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache gamma shape-rate parameters."""
        self.prior = prior
        if isinstance(prior, GammaDistribution):
            shape, scale = prior.get_parameters()
            self.conj_prior_params: Optional[tuple[float, float]] = (
                shape,
                1.0 / scale,
            )
        else:
            self.conj_prior_params = None

    def log_density(self, x: float) -> float:
        """Evaluate log-density, returning ``-inf`` outside ``x >= 0``."""
        if not np.isfinite(x) or x < 0.0:
            return float(-np.inf)
        return float(self.log_lam - self.lam * x)

    def expected_log_density(self, x: float) -> float:
        """Evaluate prior-averaged log-density when the prior is gamma."""
        if not np.isfinite(x) or x < 0.0:
            return float(-np.inf)
        if self.conj_prior_params is None:
            return self.log_density(x)
        shape, rate = self.conj_prior_params
        return float(digamma(shape) - np.log(rate) - shape * x / rate)

    def seq_log_density(self, x: Array) -> Array:
        """Evaluate fixed-parameter log-densities for encoded observations."""
        result = self.log_lam - self.lam * x
        return np.where(np.isfinite(x) & (x >= 0.0), result, -np.inf).astype(float)

    def seq_expected_log_density(self, x: Array) -> Array:
        """Evaluate prior-averaged log-densities for encoded observations."""
        if self.conj_prior_params is None:
            return self.seq_log_density(x)
        shape, rate = self.conj_prior_params
        result = digamma(shape) - np.log(rate) - shape * x / rate
        return np.where(np.isfinite(x) & (x >= 0.0), result, -np.inf).astype(float)

    def seq_encode(self, x: Iterable[float]) -> Array:
        """Encode observations as a floating-point array."""
        return np.asarray(tuple(x), dtype=np.float64)

    def value(self) -> list[float]:
        """Return the legacy one-element parameter list."""
        return [self.lam]

    def sampler(self, seed: Optional[int] = None) -> "ExponentialSampler":
        """Create a repeatable exponential sampler."""
        return ExponentialSampler(self, seed)

    def estimator(self) -> "ExponentialEstimator":
        """Create an estimator retaining the current prior and name."""
        return ExponentialEstimator(prior=self.prior, name=self.name)

    def dist_to_encoder(self) -> "ExponentialDataEncoder":
        """Create the exponential sequence encoder."""
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler[float]):
    """Draw independent exponential observations."""

    def __init__(
        self, dist: ExponentialDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one float or an array of ``size`` observations."""
        value = self.rng.exponential(scale=1.0 / self.dist.lam, size=size)
        return float(value) if size is None else value


class ExponentialAccumulator(StatisticAccumulator[float, ExponentialSuffStat, Array]):
    """Accumulate weighted observation count and sum."""

    def __init__(self, keys: tuple[Optional[str], ...] = (None,)) -> None:
        """Initialize empty statistics and an optional shared key."""
        self.sum = 0.0
        self.count = 0.0
        self.key = keys[0]

    def update(self, x: float, weight: float, estimate: Optional[Model]) -> None:
        """Accumulate one weighted observation on the support."""
        del estimate
        if np.isfinite(x) and x >= 0.0:
            self.sum += x * weight
            self.count += weight

    def seq_update(
        self, x: Array, weights: np.ndarray[Any, Any], estimate: Optional[Model]
    ) -> None:
        """Accumulate encoded observations and corresponding weights."""
        del estimate
        valid = np.isfinite(x) & (x >= 0.0)
        self.sum += float(np.dot(x[valid], weights[valid]))
        self.count += float(np.sum(weights[valid]))

    def initialize(self, x: float, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self, x: Array, weights: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> None:
        """Accumulate a sequence during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: ExponentialSuffStat) -> "ExponentialAccumulator":
        """Merge ``(count, sum)`` sufficient statistics."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> ExponentialSuffStat:
        """Return ``(count, sum)`` sufficient statistics."""
        return self.count, self.sum

    def from_value(self, x: ExponentialSuffStat) -> "ExponentialAccumulator":
        """Restore ``(count, sum)`` sufficient statistics."""
        self.count, self.sum = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics through the configured shared key."""
        if self.key is not None:
            if self.key in stats_dict:
                count, total = stats_dict[self.key]
                stats_dict[self.key] = (count + self.count, total + self.sum)
            else:
                stats_dict[self.key] = self.value()

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace statistics through the configured shared key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "ExponentialDataEncoder":
        """Create the compatible exponential encoder."""
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(
    StatisticAccumulatorFactory[float, ExponentialSuffStat, Array]
):
    """Create exponential accumulators with a shared key."""

    def __init__(self, keys: tuple[Optional[str], ...] = (None,)) -> None:
        """Store the keys copied into each accumulator."""
        self.keys = keys

    def make(self) -> ExponentialAccumulator:
        """Create an empty exponential accumulator."""
        return ExponentialAccumulator(self.keys)


class ExponentialEstimator(
    ParameterEstimator[float, float, Array, ExponentialSuffStat]
):
    """Estimate an exponential rate and update its gamma posterior."""

    def __init__(
        self,
        prior: Model | list[float] = default_prior,
        name: Optional[str] = None,
        keys: tuple[Optional[str], ...] = (None,),
    ) -> None:
        """Initialize estimator metadata and prior."""
        self.keys = keys
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self) -> ExponentialAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return ExponentialAccumulatorFactory(self.keys)

    def get_prior(self) -> Any:
        """Return the estimator prior or legacy hyperparameter list."""
        return self.prior

    def set_prior(self, prior: Any) -> None:
        """Replace the prior and cache shape-rate hyperparameters."""
        self.prior = prior
        if isinstance(prior, GammaDistribution):
            shape, scale = prior.get_parameters()
            self.conj_prior_params: Optional[tuple[float, float]] = (
                shape,
                1.0 / scale,
            )
        elif isinstance(prior, list) and len(prior) == 2:
            self.conj_prior_params = (float(prior[0]), float(prior[1]))
        else:
            self.conj_prior_params = None

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> ExponentialDistribution:
        """Estimate from weighted ``(count, sum)`` statistics."""
        count, total = args[-1]
        if self.conj_prior_params is not None:
            shape, rate = self.conj_prior_params
            posterior_shape = count + shape
            posterior_rate = total + rate
            return ExponentialDistribution(
                (posterior_shape - 1.0) / posterior_rate,
                name=self.name,
                prior=GammaDistribution(posterior_shape, 1.0 / posterior_rate),
            )
        return ExponentialDistribution(count / total, name=self.name)


class ExponentialDataEncoder(DataSequenceEncoder[float, Array]):
    """Encode exponential observations as floating-point arrays."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "ExponentialDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has exponential semantics."""
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: Iterable[float]) -> "ExponentialEncodedData":
        """Encode observations in a typed container."""
        return ExponentialEncodedData(np.asarray(tuple(x), dtype=np.float64))


class ExponentialEncodedData(EncodedDataSequence[Array]):
    """Contain an encoded exponential sequence."""

    def __init__(self, data: Array) -> None:
        """Store the floating-point array."""
        super().__init__(data)
