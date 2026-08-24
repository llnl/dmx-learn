"""Bayesian wrapper for observations that may be missing.

Missingness keeps the legacy identity policy: the marker matches with ``is``;
when it is NaN, any scalar NaN also matches. The composite prior contains the
missingness prior followed by the child prior. Sufficient statistics are
``(missing_weight, present_weight, child_statistics)``; only present values
reach the child distribution and accumulator.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.beta import BetaDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.nulldist import NullDistribution
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma

# Legacy bstats implementations are concrete protocol classes.
# pylint: disable=abstract-method,arguments-differ

Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Array = np.ndarray[Any, Any]
OptionalEncoded = tuple[int, Array, Array, Any]
OptionalSuffStat = tuple[float, float, Any]

default_prior = BetaDistribution(1.0001, 1.0001)


def _is_nan_scalar(value: Any) -> bool:
    """Return whether a value is a scalar NaN."""
    if not np.isscalar(value):
        return False
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


class OptionalDistribution(
    ProbabilityDistribution[Any, tuple[float, Any], OptionalEncoded]
):
    """Add an explicit missing outcome to a child distribution.

    ``p`` is the probability of the missing outcome; a present observation has
    probability ``1 - p`` times its child density. Missing values match the
    configured marker by identity, except that a scalar NaN marker matches any
    scalar NaN. The parameter tuple is ``(p, child_parameters)`` and the prior
    is ``(missingness_prior, child_prior)``.

    Args:
        dist: Distribution for present observations.
        p: Probability that an observation is missing.
        missing_value: Marker representing the missing outcome.
        name: Optional identifier for the wrapper.
        prior: Prior for the missing probability.
        keys: Optional key sharing the complete optional sufficient statistic.

    Raises:
        ValueError: If ``p`` is not finite or is outside ``[0, 1]``.
    """

    def __init__(
        self,
        dist: Model,
        p: float = 0.5,
        missing_value: Any = None,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize the child, missing marker, probability, and prior."""
        super().__init__()
        self.dist = dist
        self.missing_value = missing_value
        self.mv_is_nan = _is_nan_scalar(missing_value)
        self.name = name
        self.keys = keys
        self.set_parameters((p, dist.get_parameters()))
        self._set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"OptionalDistribution({self.dist}, p={self.p!r}, "
            f"missing_value={self.missing_value!r}, name={self.name!r}, "
            f"prior={self.prior}, keys={self.keys!r})"
        )

    def _is_missing(self, value: Any) -> bool:
        return value is self.missing_value or (self.mv_is_nan and _is_nan_scalar(value))

    def get_parameters(self) -> tuple[float, Any]:
        """Return missing probability and child parameters."""
        return self.p, self.dist.get_parameters()

    def set_parameters(self, value: tuple[float, Any]) -> None:
        """Set missing probability and child parameters."""
        probability = float(value[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Optional probability must be between zero and one.")
        self.p = probability
        with np.errstate(divide="ignore"):
            self.log_p0 = float(np.log(probability))
            self.log_p1 = float(np.log1p(-probability))
        self.dist.set_parameters(value[1])

    def get_prior(self) -> CompositeDistribution:
        """Return missingness and child priors in that order."""
        return CompositeDistribution((self.prior, self.dist.get_prior()))

    def set_prior(self, prior: Model) -> None:
        """Propagate a two-part composite prior to this wrapper and child."""
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("OptionalDistribution requires a composite prior.")
        self._set_prior(prior.dists[0])
        self.dist.set_prior(prior.dists[1])

    def _set_prior(self, prior: Model) -> None:
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)
        if isinstance(prior, BetaDistribution):
            alpha, beta = prior.get_parameters()
            self.conj_prior_params: Optional[tuple[float, float, float]] = (
                float(digamma(alpha)),
                float(digamma(beta)),
                float(digamma(alpha + beta)),
            )
        else:
            self.conj_prior_params = None

    def get_data_type(self) -> Any:
        """Return the permissive legacy observation type."""
        return Any

    def log_density(self, x: Any) -> float:
        """Score a missing marker or present child observation."""
        if self._is_missing(x):
            return self.log_p0
        return self.log_p1 + float(self.dist.log_density(x))

    def expected_log_density(self, x: Any) -> float:
        """Score under beta and child priors when available."""
        if self.conj_prior_params is None:
            return self.log_density(x)
        missing, present, total = self.conj_prior_params
        if self._is_missing(x):
            return missing - total
        return present - total + float(self.dist.expected_log_density(x))

    def cross_entropy(self, dist: Model) -> float:
        """Return cross-entropy over missing and present outcomes."""
        if isinstance(dist, OptionalDistribution):
            return float(
                -self.p * dist.log_p0
                + (1.0 - self.p) * (-dist.log_p1 + self.dist.cross_entropy(dist.dist))
            )
        return float(
            -self.p * dist.log_density(self.missing_value)
            + (1.0 - self.p) * self.dist.cross_entropy(dist)
        )

    def entropy(self) -> float:
        """Return binary missingness entropy plus weighted child entropy."""
        probabilities = np.asarray([self.p, 1.0 - self.p])
        positive = probabilities > 0.0
        binary = float(
            -np.dot(probabilities[positive], np.log(probabilities[positive]))
        )
        return binary + (1.0 - self.p) * self.dist.entropy()

    def seq_log_density(self, x: OptionalEncoded) -> Array:
        """Score encoded observations, evaluating only present values."""
        count, present, _missing, child_data = x
        values = np.full(count, self.log_p0, dtype=float)
        values[present] = (
            np.asarray(self.dist.seq_log_density(child_data)) + self.log_p1
        )
        return values

    def seq_expected_log_density(self, x: OptionalEncoded) -> Array:
        """Score encoded observations under configured priors."""
        if self.conj_prior_params is None:
            return self.seq_log_density(x)
        count, present_indices, _missing, child_data = x
        missing, present, total = self.conj_prior_params
        values = np.full(count, missing - total, dtype=float)
        values[present_indices] = (
            np.asarray(self.dist.seq_expected_log_density(child_data)) + present - total
        )
        return values

    def seq_encode(self, x: Iterable[Any]) -> OptionalEncoded:
        """Partition observations and encode only present child values."""
        observations = tuple(x)
        present: list[int] = []
        missing: list[int] = []
        child_values: list[Any] = []
        for index, value in enumerate(observations):
            if self._is_missing(value):
                missing.append(index)
            else:
                present.append(index)
                child_values.append(value)
        return (
            len(observations),
            np.asarray(present, dtype=np.int32),
            np.asarray(missing, dtype=np.int32),
            self.dist.seq_encode(child_values),
        )

    def dist_to_encoder(self) -> "OptionalDataEncoder":
        """Create an encoder retaining marker and child encoding."""
        return OptionalDataEncoder(self.dist.dist_to_encoder(), self.missing_value)

    def sampler(self, seed: Optional[int] = None) -> "OptionalSampler":
        """Create a repeatable optional sampler."""
        return OptionalSampler(self, seed)

    def estimator(self) -> "OptionalEstimator":
        """Create an estimator retaining the child estimator and prior."""
        return OptionalEstimator(
            self.dist.estimator(),
            self.missing_value,
            name=self.name,
            keys=self.keys,
            prior=self.prior,
        )


class OptionalSampler(DistributionSampler[Any]):
    """Draw missing markers or child observations."""

    def __init__(self, dist: OptionalDistribution, seed: Optional[int] = None) -> None:
        """Initialize child and missingness samplers from independent seeds."""
        super().__init__(dist, seed)
        self.obs_sampler = dist.dist.sampler(self.new_seed())
        self.mis_sampler = np.random.RandomState(self.new_seed())

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one optional observation or a list."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        if self.mis_sampler.rand() <= self.dist.p:
            return self.dist.missing_value
        return self.obs_sampler.sample()


class OptionalEstimatorAccumulator(
    SequenceEncodableAccumulator[Any, OptionalSuffStat, OptionalEncoded]
):
    """Accumulate missing/present weights and child statistics.

    The sufficient statistic is ``(missing_weight, present_weight,
    child_statistics)``. Only present observations reach the child
    accumulator. The optional wrapper key shares this complete tuple, after
    which sharing configured by the child accumulator is also applied.
    """

    def __init__(
        self,
        accumulator: Accumulator,
        missing_value: Any,
        name: Optional[str],
        keys: Optional[str],
    ) -> None:
        """Initialize wrapper metadata and an empty sufficient statistic."""
        self.acc = accumulator
        self.name = name
        self.key = keys
        self.psum = 0.0
        self.nsum = 0.0
        self.missing_value = missing_value
        self.mv_is_nan = _is_nan_scalar(missing_value)

    def _is_missing(self, value: Any) -> bool:
        return value is self.missing_value or (self.mv_is_nan and _is_nan_scalar(value))

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Initialize from one observation."""
        if self._is_missing(x):
            self.psum += weight
        else:
            self.nsum += weight
            self.acc.initialize(x, weight, rng)

    def seq_initialize(
        self, x: OptionalEncoded, weights: Array, rng: np.random.RandomState
    ) -> None:
        """Initialize from encoded observations."""
        _count, present, missing, child_data = x
        self.psum += float(weights[missing].sum())
        self.nsum += float(weights[present].sum())
        child = cast(SequenceEncodableAccumulator[Any, Any, Any], self.acc)
        child.seq_initialize(child_data, weights[present], rng)

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Update from one observation."""
        if self._is_missing(x):
            self.psum += weight
            return
        self.nsum += weight
        child = estimate.dist if isinstance(estimate, OptionalDistribution) else None
        self.acc.update(x, weight, child)

    def seq_update(
        self, x: OptionalEncoded, weights: Array, estimate: Optional[Model]
    ) -> None:
        """Update from encoded observations."""
        _count, present, missing, child_data = x
        self.psum += float(weights[missing].sum())
        self.nsum += float(weights[present].sum())
        child_estimate = (
            estimate.dist if isinstance(estimate, OptionalDistribution) else None
        )
        child = cast(SequenceEncodableAccumulator[Any, Any, Any], self.acc)
        child.seq_update(child_data, weights[present], child_estimate)

    def combine(self, suff_stat: OptionalSuffStat) -> "OptionalEstimatorAccumulator":
        """Merge wrapper and child statistics."""
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        self.acc.combine(suff_stat[2])
        return self

    def value(self) -> OptionalSuffStat:
        """Return missing, present, and child statistics."""
        return self.psum, self.nsum, self.acc.value()

    def from_value(self, x: OptionalSuffStat) -> "OptionalEstimatorAccumulator":
        """Restore missing, present, and child statistics."""
        self.psum, self.nsum = x[:2]
        self.acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge configured wrapper and child keys."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self
        self.acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace configured wrapper and child keys."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())
        self.acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "OptionalDataEncoder":
        """Create the corresponding optional encoder."""
        return OptionalDataEncoder(self.acc.acc_to_encoder(), self.missing_value)


class OptionalEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[Any, OptionalSuffStat, OptionalEncoded]
):
    """Create optional accumulators around child accumulators."""

    def __init__(
        self,
        acc_factory: AccumulatorFactory,
        missing_value: Any,
        name: Optional[str],
        keys: Optional[str],
    ) -> None:
        """Initialize the child accumulator factory and wrapper metadata."""
        self.acc_factory = acc_factory
        self.missing_value = missing_value
        self.name = name
        self.keys = keys

    def make(self) -> OptionalEstimatorAccumulator:
        """Create an empty optional accumulator."""
        return OptionalEstimatorAccumulator(
            self.acc_factory.make(), self.missing_value, self.name, self.keys
        )


class OptionalEstimator(
    ParameterEstimator[Any, tuple[float, Any], OptionalEncoded, OptionalSuffStat]
):
    """Estimate missingness and child parameters.

    A beta prior produces a posterior-mode missing probability and is updated
    to the beta posterior stored on the result. Other priors use the empirical
    missing fraction. ``fixed_prob`` overrides either probability calculation
    without preventing child estimation.

    Args:
        estimator: Estimator for present observations.
        missing_value: Marker representing the missing outcome.
        fixed_prob: Optional fixed probability of missingness.
        name: Optional identifier copied to estimated distributions.
        keys: Optional key sharing the complete optional statistic.
        prior: Prior for the missing probability.
    """

    def __init__(
        self,
        estimator: Estimator,
        missing_value: Any = None,
        fixed_prob: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize child estimation and missingness configuration."""
        self.estimator = estimator
        self.missing_value = missing_value
        self.fixed_prob = fixed_prob
        self.name = name
        self.keys = keys
        self._set_prior(prior)

    def _set_prior(self, prior: Model) -> None:
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)

    def accumulator_factory(self) -> OptionalEstimatorAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return OptionalEstimatorAccumulatorFactory(
            self.estimator.accumulator_factory(),
            self.missing_value,
            self.name,
            self.keys,
        )

    def set_prior(self, prior: Model) -> None:
        """Propagate missingness and child priors."""
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("OptionalEstimator requires a composite prior.")
        self._set_prior(prior.dists[0])
        self.estimator.set_prior(prior.dists[1])

    def get_prior(self) -> CompositeDistribution:
        """Return missingness and child priors."""
        return CompositeDistribution((self.prior, self.estimator.get_prior()))

    def estimate(
        self, *args: Any
    ) -> OptionalDistribution:  # pylint: disable=arguments-differ
        """Estimate from optional sufficient statistics."""
        missing, present, child_stats = args[-1]
        child = self.estimator.estimate(child_stats)
        if isinstance(self.prior, BetaDistribution):
            alpha, beta = self.prior.get_parameters()
            post_alpha, post_beta = alpha + missing, beta + present
            denominator = post_alpha + post_beta - 2.0
            probability = (post_alpha - 1.0) / denominator if denominator > 0 else 0.5
            updated_prior: Model = BetaDistribution(post_alpha, post_beta)
        else:
            probability = missing / (missing + present) if missing + present else 0.5
            updated_prior = self.prior
        if self.fixed_prob is not None:
            probability = self.fixed_prob
        return OptionalDistribution(
            child, probability, self.missing_value, self.name, updated_prior, self.keys
        )


class OptionalDataEncoder(DataSequenceEncoder[Any, OptionalEncoded]):
    """Encode optional observations with a child encoder.

    For ``n`` observations the payload is ``(n, present_indices,
    missing_indices, encoded_present_values)``. Both index arrays are
    one-dimensional, and the child encoder sees only present values.

    Args:
        encoder: Encoder for present observations.
        missing_value: Marker representing the missing outcome.
    """

    def __init__(
        self, encoder: DataSequenceEncoder[Any, Any], missing_value: Any
    ) -> None:
        """Initialize the child encoder and missing-value marker."""
        self.encoder = encoder
        self.missing_value = missing_value
        self.mv_is_nan = _is_nan_scalar(missing_value)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return f"OptionalDataEncoder({self.encoder}, {self.missing_value!r})"

    def __eq__(self, other: object) -> bool:
        """Return whether marker semantics and child encoders match."""
        if not isinstance(other, OptionalDataEncoder) or self.encoder != other.encoder:
            return False
        return (
            self.mv_is_nan and other.mv_is_nan
        ) or self.missing_value is other.missing_value

    def seq_encode(self, x: Iterable[Any]) -> "OptionalEncodedData":
        """Partition observations and encode present values."""
        observations = tuple(x)
        present = [
            i
            for i, value in enumerate(observations)
            if not (
                value is self.missing_value
                or (self.mv_is_nan and _is_nan_scalar(value))
            )
        ]
        missing = [i for i in range(len(observations)) if i not in present]
        encoded = self.encoder.seq_encode(observations[i] for i in present)
        return OptionalEncodedData(
            (
                len(observations),
                np.asarray(present, dtype=np.int32),
                np.asarray(missing, dtype=np.int32),
                encoded.data,
            )
        )


class OptionalEncodedData(  # pylint: disable=too-few-public-methods
    EncodedDataSequence[OptionalEncoded]
):
    """Contain the stable four-part optional encoding."""
