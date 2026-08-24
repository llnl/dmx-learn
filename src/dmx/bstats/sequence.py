"""Bayesian distribution for variable-length iid sequences.

An observation is a list whose elements are scored independently by ``dist``;
its length is scored by ``len_dist``. Empty and unequal-length observations are
valid. With ``len_normalized=True`` child log scores and child sufficient
statistics are divided by the nonzero sequence length, while the length factor
keeps its original weight. The composite prior and sufficient statistic both
store child information first and length information second.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.nulldist import null_dist, null_estimator
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

# Legacy bstats implementations are concrete protocol classes.
# pylint: disable=abstract-method,arguments-differ

Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Array = np.ndarray[Any, Any]
SequenceEncoded = tuple[Array, Array, Array, Any, Any]
SequenceSuffStat = tuple[Any, Any]


class SequenceDistribution(
    ProbabilityDistribution[list[Any], tuple[Any, Any], SequenceEncoded]
):
    """Model a variable-length list of iid child observations."""

    def __init__(
        self,
        dist: Model,
        len_dist: Optional[Model] = null_dist,
        name: Optional[str] = None,
        len_normalized: bool = False,
    ) -> None:
        """Initialize child, length, normalization, and naming configuration."""
        super().__init__()
        self.dist = dist
        self.len_dist = len_dist
        self.len_normalized = len_normalized
        self.name = name
        dist.add_parent(self)
        if len_dist is not None:
            len_dist.add_parent(self)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"SequenceDistribution({self.dist}, len_dist={self.len_dist}, "
            f"name={self.name!r})"
        )

    def get_parameters(self) -> tuple[Any, Any]:
        """Return child and length parameters."""
        return (
            self.dist.get_parameters(),
            None if self.len_dist is None else self.len_dist.get_parameters(),
        )

    def set_parameters(self, value: tuple[Any, Any]) -> None:
        """Set child and length parameters."""
        self.dist.set_parameters(value[0])
        if self.len_dist is not None:
            self.len_dist.set_parameters(value[1])

    def get_prior(self) -> CompositeDistribution:
        """Return child and length priors in that order."""
        length_prior = null_dist if self.len_dist is None else self.len_dist.get_prior()
        return CompositeDistribution((self.dist.get_prior(), length_prior))

    def set_prior(self, prior: Model) -> None:
        """Propagate a two-part composite prior to both child models."""
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("SequenceDistribution requires a composite prior.")
        self.dist.set_prior(prior.dists[0])
        if self.len_dist is not None:
            self.len_dist.set_prior(prior.dists[1])

    def cross_entropy(self, dist: Model) -> float:
        """Return sequence cross-entropy when the mean length is available."""
        if not isinstance(dist, SequenceDistribution) or self.len_dist is None:
            raise TypeError("Sequence cross-entropy requires another sequence model.")
        mean_length = float(getattr(self.len_dist, "moment")(1))
        length_value = (
            0.0 if dist.len_dist is None else self.len_dist.cross_entropy(dist.len_dist)
        )
        return mean_length * self.dist.cross_entropy(dist.dist) + length_value

    def entropy(self) -> float:
        """Return sequence entropy when the mean length is available."""
        if self.len_dist is None:
            return self.dist.entropy()
        mean_length = float(getattr(self.len_dist, "moment")(1))
        return mean_length * self.dist.entropy() + self.len_dist.entropy()

    def log_density(self, x: list[Any]) -> float:
        """Sum child scores and the score of the observed length."""
        child_score = sum(self.dist.log_density(value) for value in x)
        if self.len_normalized and x:
            child_score /= len(x)
        length_score = (
            0.0 if self.len_dist is None else self.len_dist.log_density(len(x))
        )
        return float(child_score + length_score)

    def expected_log_density(self, x: list[Any]) -> float:
        """Sum prior-expected child and length scores."""
        child_score = sum(self.dist.expected_log_density(value) for value in x)
        if self.len_normalized and x:
            child_score /= len(x)
        length_score = (
            0.0 if self.len_dist is None else self.len_dist.expected_log_density(len(x))
        )
        return float(child_score + length_score)

    def seq_log_density(self, x: SequenceEncoded) -> Array:
        """Score flattened encoded sequences and regroup by observation."""
        indices, inverse_lengths, _nonempty, child_data, length_data = x
        count = len(inverse_lengths)
        if indices.size:
            child_scores = np.asarray(
                self.dist.seq_log_density(child_data), dtype=float
            )
            values = np.bincount(indices, weights=child_scores, minlength=count)
            if self.len_normalized:
                values *= inverse_lengths
        else:
            values = np.zeros(count, dtype=float)
        if self.len_dist is not None and length_data is not None:
            values += np.asarray(
                self.len_dist.seq_log_density(length_data), dtype=float
            )
        return values

    def seq_expected_log_density(self, x: SequenceEncoded) -> Array:
        """Score encoded sequences under child and length priors."""
        indices, inverse_lengths, _nonempty, child_data, length_data = x
        count = len(inverse_lengths)
        if indices.size:
            child_scores = np.asarray(
                self.dist.seq_expected_log_density(child_data), dtype=float
            )
            values = np.bincount(indices, weights=child_scores, minlength=count)
            if self.len_normalized:
                values *= inverse_lengths
        else:
            values = np.zeros(count, dtype=float)
        if self.len_dist is not None and length_data is not None:
            values += np.asarray(
                self.len_dist.seq_expected_log_density(length_data), dtype=float
            )
        return values

    def seq_encode(self, x: Iterable[list[Any]]) -> SequenceEncoded:
        """Flatten variable-length lists while retaining grouping and lengths."""
        observations = tuple(x)
        flat: list[Any] = []
        indices: list[int] = []
        lengths: list[int] = []
        for index, observation in enumerate(observations):
            lengths.append(len(observation))
            flat.extend(observation)
            indices.extend([index] * len(observation))
        inverse_lengths = np.asarray(lengths, dtype=float)
        nonempty = inverse_lengths != 0
        inverse_lengths[nonempty] = 1.0 / inverse_lengths[nonempty]
        return (
            np.asarray(indices, dtype=int),
            inverse_lengths,
            nonempty,
            self.dist.seq_encode(flat),
            None if self.len_dist is None else self.len_dist.seq_encode(lengths),
        )

    def sampler(self, seed: Optional[int] = None) -> "SequenceSampler":
        """Create a repeatable sequence sampler."""
        return SequenceSampler(self, seed)

    def estimator(self) -> "SequenceEstimator":
        """Create an estimator retaining normalization and child estimators."""
        return SequenceEstimator(
            self.dist.estimator(),
            None if self.len_dist is None else self.len_dist.estimator(),
            self.len_normalized,
            self.name,
        )

    def dist_to_encoder(self) -> "SequenceDataEncoder":
        """Create an encoder combining child and length encoders."""
        length_encoder = (
            None if self.len_dist is None else self.len_dist.dist_to_encoder()
        )
        return SequenceDataEncoder(self.dist.dist_to_encoder(), length_encoder)


class SequenceSampler(DistributionSampler[list[Any]]):
    """Draw a length followed by that many iid child observations."""

    def __init__(self, dist: SequenceDistribution, seed: Optional[int] = None) -> None:
        """Initialize independently seeded child and length samplers."""
        super().__init__(dist, seed)
        self.distSampler = dist.dist.sampler(self.new_seed())
        if dist.len_dist is None:
            raise ValueError("A sequence sampler requires a length distribution.")
        self.lenSampler = dist.len_dist.sampler(self.new_seed())

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one variable-length list or a list of lists."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        length = int(self.lenSampler.sample())
        return [self.distSampler.sample() for _ in range(length)]


class SequenceEstimatorAccumulator(
    SequenceEncodableAccumulator[list[Any], SequenceSuffStat, SequenceEncoded]
):
    """Accumulate flattened child statistics and length statistics."""

    def __init__(
        self,
        accumulator: Accumulator,
        len_normalized: bool,
        len_accumulator: Optional[Accumulator],
        keys: tuple[Optional[str], Optional[str]],
    ) -> None:
        """Initialize child accumulators and sequence metadata."""
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.dist_key, self.len_key = keys
        self.len_normalized = len_normalized

    def update(self, x: list[Any], weight: float, estimate: Optional[Model]) -> None:
        """Update child elements and sequence length."""
        child_estimate = (
            estimate.dist if isinstance(estimate, SequenceDistribution) else None
        )
        length_estimate = (
            estimate.len_dist if isinstance(estimate, SequenceDistribution) else None
        )
        child_weight = weight / len(x) if self.len_normalized and x else weight
        for value in x:
            self.accumulator.update(value, child_weight, child_estimate)
        if self.len_accumulator is not None:
            self.len_accumulator.update(len(x), weight, length_estimate)

    def initialize(
        self, x: list[Any], weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize child elements and sequence length."""
        child_weight = weight / len(x) if self.len_normalized and x else weight
        for value in x:
            self.accumulator.initialize(value, child_weight, rng)
        if self.len_accumulator is not None:
            self.len_accumulator.initialize(len(x), weight, rng)

    def combine(self, suff_stat: SequenceSuffStat) -> "SequenceEstimatorAccumulator":
        """Merge child and length sufficient statistics."""
        self.accumulator.combine(suff_stat[0])
        if self.len_accumulator is not None:
            self.len_accumulator.combine(suff_stat[1])
        return self

    def value(self) -> SequenceSuffStat:
        """Return child and length sufficient statistics."""
        length_value = (
            None if self.len_accumulator is None else self.len_accumulator.value()
        )
        return self.accumulator.value(), length_value

    def from_value(self, x: SequenceSuffStat) -> "SequenceEstimatorAccumulator":
        """Restore child and length sufficient statistics."""
        self.accumulator.from_value(x[0])
        if self.len_accumulator is not None:
            self.len_accumulator.from_value(x[1])
        return self

    def seq_initialize(
        self, x: SequenceEncoded, weights: Array, rng: np.random.RandomState
    ) -> None:
        """Initialize from flattened encoded sequences."""
        indices, inverse_lengths, _nonempty, child_data, length_data = x
        child_weights = weights[indices]
        if self.len_normalized:
            child_weights = child_weights * inverse_lengths[indices]
        child = cast(SequenceEncodableAccumulator[Any, Any, Any], self.accumulator)
        child.seq_initialize(child_data, child_weights, rng)
        if self.len_accumulator is not None:
            length = cast(
                SequenceEncodableAccumulator[Any, Any, Any], self.len_accumulator
            )
            length.seq_initialize(length_data, weights, rng)

    def seq_update(
        self, x: SequenceEncoded, weights: Array, estimate: Optional[Model]
    ) -> None:
        """Update from flattened encoded sequences."""
        indices, inverse_lengths, _nonempty, child_data, length_data = x
        child_weights = weights[indices]
        if self.len_normalized:
            child_weights = child_weights * inverse_lengths[indices]
        child_estimate = (
            estimate.dist if isinstance(estimate, SequenceDistribution) else None
        )
        length_estimate = (
            estimate.len_dist if isinstance(estimate, SequenceDistribution) else None
        )
        child = cast(SequenceEncodableAccumulator[Any, Any, Any], self.accumulator)
        child.seq_update(child_data, child_weights, child_estimate)
        if self.len_accumulator is not None:
            length = cast(
                SequenceEncodableAccumulator[Any, Any, Any], self.len_accumulator
            )
            length.seq_update(length_data, weights, length_estimate)

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge whole sequence statistics and configured child keys."""
        for key in (self.dist_key, self.len_key):
            if key is not None:
                if key in stats_dict:
                    stats_dict[key].combine(self.value())
                else:
                    stats_dict[key] = self
        self.accumulator.key_merge(stats_dict)
        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace whole sequence statistics and configured child keys."""
        for key in (self.dist_key, self.len_key):
            if key is not None and key in stats_dict:
                self.from_value(stats_dict[key].value())
        self.accumulator.key_replace(stats_dict)
        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SequenceDataEncoder":
        """Create the corresponding child and length encoder."""
        length_encoder = (
            None
            if self.len_accumulator is None
            else self.len_accumulator.acc_to_encoder()
        )
        return SequenceDataEncoder(self.accumulator.acc_to_encoder(), length_encoder)


class SequenceEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[list[Any], SequenceSuffStat, SequenceEncoded]
):
    """Create sequence accumulators from child factories."""

    def __init__(
        self,
        factory: AccumulatorFactory,
        len_normalized: bool,
        len_factory: Optional[AccumulatorFactory],
        keys: tuple[Optional[str], Optional[str]],
    ) -> None:
        """Initialize child factories and sequence metadata."""
        self.factory = factory
        self.len_normalized = len_normalized
        self.len_factory = len_factory
        self.keys = keys

    def make(self) -> SequenceEstimatorAccumulator:
        """Create an empty sequence accumulator."""
        length = None if self.len_factory is None else self.len_factory.make()
        return SequenceEstimatorAccumulator(
            self.factory.make(), self.len_normalized, length, self.keys
        )


class SequenceEstimator(
    ParameterEstimator[list[Any], tuple[Any, Any], SequenceEncoded, SequenceSuffStat]
):
    """Estimate child and optional length models."""

    def __init__(
        self,
        estimator: Estimator,
        len_estimator: Optional[Estimator] = null_estimator,
        len_normalized: bool = False,
        name: Optional[str] = None,
        keys: tuple[Optional[str], Optional[str]] = (None, None),
    ) -> None:
        """Initialize child estimators and sequence metadata."""
        self.name = name
        self.estimator = estimator
        self.len_estimator = len_estimator
        self.keys = keys
        self.len_normalized = len_normalized

    def get_prior(self) -> CompositeDistribution:
        """Return child and length priors."""
        length_prior = (
            null_dist if self.len_estimator is None else self.len_estimator.get_prior()
        )
        return CompositeDistribution((self.estimator.get_prior(), length_prior))

    def set_prior(self, prior: Model) -> None:
        """Propagate a two-part prior to child and length estimators."""
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("SequenceEstimator requires a composite prior.")
        self.estimator.set_prior(prior.dists[0])
        if self.len_estimator is not None:
            self.len_estimator.set_prior(prior.dists[1])

    def model_log_density(self, model: CompositeDistribution) -> float:
        """Score model parameters under this estimator's composite prior."""
        return float(self.get_prior().log_density(model.get_parameters()))

    def accumulator_factory(self) -> SequenceEstimatorAccumulatorFactory:
        """Create a factory combining child and length factories."""
        length_factory = (
            None
            if self.len_estimator is None
            else self.len_estimator.accumulator_factory()
        )
        return SequenceEstimatorAccumulatorFactory(
            self.estimator.accumulator_factory(),
            self.len_normalized,
            length_factory,
            self.keys,
        )

    def estimate(
        self, *args: Any
    ) -> SequenceDistribution:  # pylint: disable=arguments-differ
        """Estimate child and optional length models from paired statistics."""
        child_stats, length_stats = args[-1]
        child = self.estimator.estimate(child_stats)
        length = (
            None
            if self.len_estimator is None
            else self.len_estimator.estimate(length_stats)
        )
        return SequenceDistribution(child, length, self.name, self.len_normalized)


class SequenceDataEncoder(DataSequenceEncoder[list[Any], SequenceEncoded]):
    """Encode flattened child values and sequence lengths."""

    def __init__(
        self,
        encoder: DataSequenceEncoder[Any, Any],
        len_encoder: Optional[DataSequenceEncoder[Any, Any]],
    ) -> None:
        """Initialize child and optional length encoders."""
        self.encoder = encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return f"SequenceDataEncoder({self.encoder}, {self.len_encoder})"

    def __eq__(self, other: object) -> bool:
        """Return whether child and length encoders match."""
        return (
            isinstance(other, SequenceDataEncoder)
            and self.encoder == other.encoder
            and self.len_encoder == other.len_encoder
        )

    def seq_encode(self, x: Iterable[list[Any]]) -> "SequenceEncodedData":
        """Flatten lists and encode child values and lengths."""
        observations = tuple(x)
        lengths = [len(value) for value in observations]
        indices = np.asarray(
            [index for index, value in enumerate(observations) for _ in value],
            dtype=int,
        )
        inverse = np.asarray(lengths, dtype=float)
        nonempty = inverse != 0
        inverse[nonempty] = 1.0 / inverse[nonempty]
        child = self.encoder.seq_encode(
            value for observation in observations for value in observation
        )
        length_data = (
            None
            if self.len_encoder is None
            else self.len_encoder.seq_encode(lengths).data
        )
        return SequenceEncodedData(
            (indices, inverse, nonempty, child.data, length_data)
        )


class SequenceEncodedData(  # pylint: disable=too-few-public-methods
    EncodedDataSequence[SequenceEncoded]
):
    """Contain the stable five-part variable-sequence encoding."""
