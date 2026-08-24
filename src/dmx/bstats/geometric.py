"""Bayesian geometric likelihoods for positive integer trial counts.

``GeometricDistribution(p)`` models the number of trials through the first
success, so its observation support is integers ``x >= 1``. The default prior
is ``Beta(1, 1)``. Accumulators store weighted ``(count, sum)`` statistics;
the beta posterior adds ``count`` successes and ``sum - count`` failures.
Expected log-density integrates both log-probability terms under a beta prior
and otherwise falls back to fixed-parameter scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import geom

from dmx.bstats.beta import BetaDistribution
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
from dmx.utils.special import digamma

# Legacy bstats implementations are intentionally concrete protocol classes.
# pylint: disable=abstract-method

GeometricEncoded = np.ndarray[Any, np.dtype[np.float64]]
GeometricSuffStat = tuple[float, float]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = BetaDistribution(1.0, 1.0)


class GeometricDistribution(ProbabilityDistribution[int, float, GeometricEncoded]):
    """Geometric likelihood on positive integer trial counts."""

    def __init__(
        self,
        p: float,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a success probability, metadata, and prior."""
        super().__init__()
        self.name = name
        self.keys = keys
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"GeometricDistribution({self.p!r}, name={self.name!r}, "
            f"prior={self.prior}, keys={self.keys!r})"
        )

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache beta expected-log parameters."""
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            self.conj_prior_params: Optional[tuple[float, float, float]] = (
                float(digamma(prior.a)),
                float(digamma(prior.b)),
                float(digamma(prior.a + prior.b)),
            )
        else:
            self.conj_prior_params = None

    def get_prior(self) -> Model:
        """Return the current beta or alternate prior."""
        return self.prior

    def get_parameters(self) -> float:
        """Return the success probability."""
        return self.p

    def set_parameters(self, value: float) -> None:
        """Replace the success probability and cached logarithms."""
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Geometric probability must be between zero and one.")
        self.p = float(value)
        with np.errstate(divide="ignore"):
            self.log_p = float(np.log(self.p))
            self.log_1p = float(np.log1p(-self.p))

    @staticmethod
    def _in_support(x: Any) -> bool:
        """Return whether ``x`` is a positive integer trial count."""
        return bool(
            isinstance(x, (int, np.integer))
            and not isinstance(x, (bool, np.bool_))
            and x >= 1
        )

    def entropy(self) -> float:
        """Return the geometric Shannon entropy."""
        if self.p == 0.0:
            return float(np.inf)
        if self.p == 1.0:
            return 0.0
        return float(-(np.log(self.p) + ((1.0 - self.p) / self.p) * self.log_1p))

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` for another geometric likelihood."""
        if not isinstance(dist, GeometricDistribution):
            return super().cross_entropy(dist)
        if self.p == 0.0:
            return float(np.inf)
        return float(-dist.log_p - ((1.0 - self.p) / self.p) * dist.log_1p)

    def moment(self, order: int) -> float:
        """Return a raw moment of the positive-integer trial count."""
        return float(geom.moment(order, self.p))

    def log_density(self, x: int) -> float:
        """Evaluate log mass, returning ``-inf`` outside positive integers."""
        if not self._in_support(x) or self.p == 0.0:
            return float(-np.inf)
        if self.p == 1.0:
            return 0.0 if x == 1 else float(-np.inf)
        return float((x - 1) * self.log_1p + self.log_p)

    def expected_log_density(self, x: int) -> float:
        """Evaluate prior-averaged log mass when the prior is beta."""
        if not self._in_support(x):
            return float(-np.inf)
        if self.conj_prior_params is None:
            return self.log_density(x)
        expected_p, expected_failure, expected_total = self.conj_prior_params
        return float(
            (expected_failure - expected_total) * (x - 1) + expected_p - expected_total
        )

    def seq_log_density(self, x: GeometricEncoded) -> np.ndarray[Any, Any]:
        """Evaluate log masses for encoded trial counts."""
        valid = np.isfinite(x) & (x >= 1.0) & (x == np.floor(x))
        if self.p == 0.0:
            return np.full(len(x), -np.inf)
        if self.p == 1.0:
            return np.where(valid & (x == 1.0), 0.0, -np.inf)
        result = (x - 1.0) * self.log_1p + self.log_p
        return np.where(valid, result, -np.inf).astype(float)

    def seq_expected_log_density(self, x: GeometricEncoded) -> np.ndarray[Any, Any]:
        """Evaluate expected log masses for encoded trial counts."""
        if self.conj_prior_params is None:
            return self.seq_log_density(x)
        expected_p, expected_failure, expected_total = self.conj_prior_params
        result = (
            (x - 1.0) * (expected_failure - expected_total)
            + expected_p
            - expected_total
        )
        valid = np.isfinite(x) & (x >= 1.0) & (x == np.floor(x))
        return np.where(valid, result, -np.inf).astype(float)

    def seq_encode(self, x: Iterable[int]) -> GeometricEncoded:
        """Encode trial counts as a floating-point NumPy array."""
        return np.asarray(tuple(x), dtype=np.float64)

    def sampler(self, seed: Optional[int] = None) -> "GeometricSampler":
        """Create a repeatable geometric sampler."""
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "GeometricEstimator":
        """Create an estimator retaining metadata and the current prior."""
        del pseudo_count
        return GeometricEstimator(name=self.name, keys=self.keys, prior=self.prior)

    def dist_to_encoder(self) -> "GeometricDataEncoder":
        """Create the geometric sequence encoder."""
        return GeometricDataEncoder()


class GeometricSampler(DistributionSampler[int]):
    """Draw independent positive integer geometric observations."""

    def __init__(self, dist: GeometricDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one integer or an array of ``size`` integers."""
        if self.dist.p <= 0.0:
            raise ValueError("Cannot sample a geometric distribution with p=0.")
        value = self.rng.geometric(p=self.dist.p, size=size)
        return int(value) if size is None else value


class GeometricAccumulator(
    StatisticAccumulator[int, GeometricSuffStat, GeometricEncoded]
):
    """Accumulate weighted observation count and trial-count sum."""

    def __init__(self, keys: Optional[str], name: Optional[str]) -> None:
        """Initialize empty statistics and sharing metadata."""
        self.sum = 0.0
        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional[Model]) -> None:
        """Add one weighted positive integer trial count."""
        del estimate
        if not GeometricDistribution._in_support(x):  # pylint: disable=protected-access
            raise ValueError("Geometric statistics require positive integers.")
        self.sum += x * weight
        self.count += weight

    def seq_update(
        self,
        x: GeometricEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Add encoded trial counts with corresponding weights."""
        del estimate
        if np.any((x < 1.0) | (x != np.floor(x))):
            raise ValueError("Geometric statistics require positive integers.")
        self.sum += float(np.dot(x, weights))
        self.count += float(np.sum(weights))

    def initialize(self, x: int, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: GeometricEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: GeometricSuffStat) -> "GeometricAccumulator":
        """Merge ``(observation_count, trial_count_sum)`` statistics."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> GeometricSuffStat:
        """Return ``(observation_count, trial_count_sum)``."""
        return self.count, self.sum

    def from_value(self, x: GeometricSuffStat) -> "GeometricAccumulator":
        """Restore ``(observation_count, trial_count_sum)``."""
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

    def acc_to_encoder(self) -> "GeometricDataEncoder":
        """Create the compatible geometric encoder."""
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(
    StatisticAccumulatorFactory[int, GeometricSuffStat, GeometricEncoded]
):
    """Create geometric accumulators with shared metadata."""

    def __init__(self, keys: Optional[str], name: Optional[str]) -> None:
        """Store metadata copied into each accumulator."""
        self.keys = keys
        self.name = name

    def make(self) -> GeometricAccumulator:
        """Create an empty geometric accumulator."""
        return GeometricAccumulator(self.keys, self.name)


class GeometricEstimator(
    ParameterEstimator[int, float, GeometricEncoded, GeometricSuffStat]
):
    """Estimate a geometric probability and update its beta posterior."""

    def __init__(
        self,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize estimator metadata and prior."""
        self.keys = keys
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self) -> GeometricAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return GeometricAccumulatorFactory(self.keys, self.name)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the estimator prior and update prior flags."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> GeometricDistribution:
        """Estimate from ``(observation_count, trial_count_sum)`` statistics."""
        count, trial_sum = args[-1]
        if self.has_conj_prior:
            assert isinstance(self.prior, BetaDistribution)
            posterior_a = self.prior.a + count
            posterior_b = self.prior.b + trial_sum - count
            if posterior_a > 1.0 and posterior_b > 1.0:
                probability = (posterior_a - 1.0) / (posterior_a + posterior_b - 2.0)
            elif posterior_a <= 1.0 < posterior_b:
                probability = 0.0
            elif posterior_b <= 1.0 < posterior_a:
                probability = 1.0
            else:
                probability = count / trial_sum if trial_sum else 0.5
            return GeometricDistribution(
                probability,
                name=self.name,
                prior=BetaDistribution(posterior_a, posterior_b),
                keys=self.keys,
            )
        if self.has_prior:
            failures = trial_sum - count

            def objective(value: float) -> float:
                return float(
                    -count * np.log(value)
                    - failures * np.log1p(-value)
                    - self.prior.log_density(value)
                )

            solution = minimize_scalar(
                objective,
                bounds=(np.finfo(float).eps, 1.0 - np.finfo(float).eps),
                method="bounded",
            )
            return GeometricDistribution(
                float(solution.x),
                name=self.name,
                prior=self.prior,
                keys=self.keys,
            )
        probability = count / trial_sum if trial_sum else 0.5
        return GeometricDistribution(
            probability, name=self.name, prior=null_dist, keys=self.keys
        )


class GeometricDataEncoder(DataSequenceEncoder[int, GeometricEncoded]):
    """Encode positive integer geometric observations."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "GeometricDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has geometric semantics."""
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(self, x: Iterable[int]) -> "GeometricEncodedData":
        """Encode observations in a typed container."""
        return GeometricEncodedData(np.asarray(tuple(x), dtype=np.float64))


class GeometricEncodedData(EncodedDataSequence[GeometricEncoded]):
    """Contain an encoded geometric observation sequence."""

    def __init__(self, data: GeometricEncoded) -> None:
        """Store the trial-count array."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"GeometricEncodedData(data={self.data!r})"
