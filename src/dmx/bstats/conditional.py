"""Bayesian distributions selected by a hashable condition value.

An observation is ``(condition, value)``. The condition selects a child from
``dmap``; when it is absent, ``default_dist`` is used. The shared ``null_dist``
means that no fallback is configured: such observations have log-density
``-inf`` and are ignored by sufficient-statistic accumulators.

The condition distribution supplies keys for sampling. It is not another
factor in the observation log-density and is retained, rather than re-fitted,
by the conditional estimator. This preserves the legacy bstats model.

Encoded data has the stable five-part shape
``(size, conditions, encoded_values, indices, encoded_conditions)``. The middle
three tuples are parallel and contain one entry per distinct condition, in
first-observed order. ``encoded_values`` contains ``None`` for a missing
condition when no default distribution exists. ``indices`` maps each group
back to positions in the original sequence.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.nulldist import NullDataEncoder, NullDistribution, null_dist
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
# pylint: disable=abstract-method

# Conditional children are heterogeneous, so their individual type arguments
# are erased at the mapping boundary while condition keys stay explicitly hashable.
ConditionKey = Hashable
ConditionalObservation = tuple[ConditionKey, Any]
ConditionalEncoded = tuple[
    int,
    tuple[ConditionKey, ...],
    tuple[Optional[Any], ...],
    tuple[np.ndarray[Any, Any], ...],
    Any,
]
ConditionalSuffStat = tuple[dict[ConditionKey, Any], Optional[Any]]
Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Encoder = DataSequenceEncoder[Any, Any]
Array = np.ndarray[Any, Any]


def _unwrap_encoded(
    x: ConditionalEncoded | ConditionalEncodedData,
) -> ConditionalEncoded:
    """Return the payload stored in an optional encoded-data container."""
    return x.data if isinstance(x, ConditionalEncodedData) else x


class ConditionalDistribution(
    ProbabilityDistribution[ConditionalObservation, Any, ConditionalEncoded]
):
    """Select a child distribution using the first observation element.

    If ``pass_value`` is false, the selected child receives only the second
    observation element. If true, it receives the complete
    ``(condition, value)`` pair.

    The condition model is used to sample conditions and encode them, but its
    density is not included in the conditional observation score. An unknown
    condition is handled by ``default_dist``; the default ``null_dist`` means
    the observation is assigned log density ``-inf`` and contributes no child
    sufficient statistics.

    Args:
        dmap: Mapping from condition values to child distributions.
        cond_dist: Fixed distribution used to sample and encode conditions.
        default_dist: Child used for conditions absent from ``dmap``.
        pass_value: Whether children receive the complete observation instead
            of only its value element.
    """

    def __init__(
        self,
        dmap: Mapping[ConditionKey, Model],
        cond_dist: Model,
        default_dist: Model = null_dist,
        pass_value: bool = False,
    ) -> None:
        """Initialize explicit children, the condition model, and fallback."""
        super().__init__()
        self.dmap = dict(dmap)
        self.cond_dist = cond_dist
        self.default_dist = default_dist
        self.pass_value = pass_value
        self.has_default = not isinstance(default_dist, NullDistribution)

    def __str__(self) -> str:
        """Return a constructor-like representation of the conditional model."""
        return (
            f"ConditionalDistribution({self.dmap!r}, {self.cond_dist!r}, "
            f"default_dist={self.default_dist!r}, pass_value={self.pass_value!r})"
        )

    def _child_value(self, x: ConditionalObservation) -> Any:
        return x if self.pass_value else x[1]

    def log_density(self, x: ConditionalObservation) -> float:
        """Score with the selected child or the configured default.

        A condition absent from ``dmap`` has log-density ``-inf`` when the
        default is ``null_dist``.
        """
        child = self.dmap.get(x[0])
        if child is None:
            if not self.has_default:
                return float(-np.inf)
            child = self.default_dist
        return float(child.log_density(self._child_value(x)))

    def seq_log_density(
        self, x: ConditionalEncoded | "ConditionalEncodedData"
    ) -> Array:
        """Score the documented five-part conditional encoding."""
        size, conditions, encoded_values, indices, _encoded_conditions = (
            _unwrap_encoded(x)
        )
        result = np.full(size, -np.inf, dtype=float)
        for condition, encoded, group_indices in zip(
            conditions, encoded_values, indices
        ):
            child = self.dmap.get(condition)
            if child is None:
                if not self.has_default:
                    continue
                child = self.default_dist
            result[group_indices] = child.seq_log_density(encoded)
        return result

    def seq_encode(self, x: Iterable[ConditionalObservation]) -> ConditionalEncoded:
        """Encode observations directly into the stable five-part payload."""
        observations = tuple(x)
        return self._encode(observations, use_encoders=False)

    def _encode(
        self,
        observations: tuple[ConditionalObservation, ...],
        use_encoders: bool,
    ) -> ConditionalEncoded:
        grouped: dict[ConditionKey, tuple[list[Any], list[int]]] = {}
        for index, observation in enumerate(observations):
            values, positions = grouped.setdefault(observation[0], ([], []))
            values.append(self._child_value(observation))
            positions.append(index)

        conditions: list[ConditionKey] = []
        encoded_values: list[Optional[Any]] = []
        indices: list[Array] = []
        for condition, (values, positions) in grouped.items():
            conditions.append(condition)
            child = self.dmap.get(condition)
            if child is None and self.has_default:
                child = self.default_dist
            if child is None:
                encoded_values.append(None)
            elif use_encoders:
                encoded_values.append(child.dist_to_encoder().seq_encode(values).data)
            else:
                encoded_values.append(child.seq_encode(values))
            indices.append(np.asarray(positions, dtype=int))

        condition_values = (observation[0] for observation in observations)
        if use_encoders:
            encoded_conditions = (
                self.cond_dist.dist_to_encoder().seq_encode(condition_values).data
            )
        else:
            encoded_conditions = self.cond_dist.seq_encode(condition_values)
        return (
            len(observations),
            tuple(conditions),
            tuple(encoded_values),
            tuple(indices),
            encoded_conditions,
        )

    def sampler(self, seed: Optional[int] = None) -> "ConditionalDistributionSampler":
        """Create a sampler that first draws a condition, then its child value."""
        return ConditionalDistributionSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "ConditionalDistributionEstimator":
        """Create an estimator for children while retaining the condition model."""
        del pseudo_count
        return ConditionalDistributionEstimator(
            {key: child.estimator() for key, child in self.dmap.items()},
            self.default_dist.estimator() if self.has_default else None,
            cond_dist=self.cond_dist,
            pass_value=self.pass_value,
        )

    def dist_to_encoder(self) -> "ConditionalDataEncoder":
        """Create an encoder with matching explicit, default, and condition paths."""
        return ConditionalDataEncoder(
            encoder_map={
                key: child.dist_to_encoder() for key, child in self.dmap.items()
            },
            given_encoder=self.cond_dist.dist_to_encoder(),
            default_encoder=self.default_dist.dist_to_encoder(),
            pass_value=self.pass_value,
        )


class ConditionalDistributionSampler(DistributionSampler[ConditionalObservation]):
    """Draw a condition and then sample its explicit or default child."""

    def __init__(
        self, dist: ConditionalDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize independently seeded condition and child samplers."""
        super().__init__(dist, seed)
        self.condition_sampler = dist.cond_dist.sampler(self.new_seed())
        self.samplers = {
            key: child.sampler(self.new_seed()) for key, child in dist.dmap.items()
        }
        self.default_sampler = (
            dist.default_dist.sampler(self.new_seed()) if dist.has_default else None
        )

    def _sample_one(self) -> ConditionalObservation:
        condition = cast(ConditionKey, self.condition_sampler.sample())
        sampler = self.samplers.get(condition, self.default_sampler)
        if sampler is None:
            raise RuntimeError(
                f"No conditional distribution is configured for {condition!r}."
            )
        value = sampler.sample()
        if self.dist.pass_value:
            return cast(ConditionalObservation, value)
        return condition, value

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one conditional observation or a list of ``size`` observations."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class ConditionalDistributionEstimatorAccumulator(
    SequenceEncodableAccumulator[
        ConditionalObservation, ConditionalSuffStat, ConditionalEncoded
    ]
):
    """Accumulate explicit child statistics and optional default statistics.

    The public sufficient statistic is ``(explicit, default)``, where
    ``explicit`` maps each configured condition to its child statistic and
    ``default`` is the fallback statistic or ``None``. Unknown conditions are
    ignored when no default accumulator is configured. Key sharing is
    delegated to the explicit and default children.
    """

    def __init__(
        self,
        accumulator_map: Mapping[ConditionKey, Accumulator],
        default_accumulator: Optional[Accumulator],
        keys: Optional[str] = None,
        given_encoder: Optional[Encoder] = None,
        pass_value: bool = False,
    ) -> None:
        """Initialize keyed child accumulators and encoding metadata."""
        self.accumulator_map = dict(accumulator_map)
        self.default_accumulator = default_accumulator
        self.key = keys
        self.given_encoder = given_encoder or NullDataEncoder()
        self.pass_value = pass_value

    def _child_value(self, x: ConditionalObservation) -> Any:
        return x if self.pass_value else x[1]

    def update(
        self, x: ConditionalObservation, weight: float, estimate: Optional[Model]
    ) -> None:
        """Update the selected explicit or default accumulator."""
        accumulator = self.accumulator_map.get(x[0])
        child_estimate: Optional[Model] = None
        if isinstance(estimate, ConditionalDistribution):
            child_estimate = estimate.dmap.get(x[0])
            if child_estimate is None and estimate.has_default:
                child_estimate = estimate.default_dist
        if accumulator is not None:
            accumulator.update(self._child_value(x), weight, child_estimate)
        elif self.default_accumulator is not None:
            self.default_accumulator.update(
                self._child_value(x), weight, child_estimate
            )

    def initialize(
        self, x: ConditionalObservation, weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize the selected explicit or default accumulator."""
        accumulator = self.accumulator_map.get(x[0], self.default_accumulator)
        if accumulator is not None:
            accumulator.initialize(self._child_value(x), weight, rng)

    def seq_initialize(
        self,
        x: ConditionalEncoded | ConditionalEncodedData,
        weights: Array,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize accumulators from grouped encoded observations."""
        _size, conditions, encoded_values, indices, _conditions = _unwrap_encoded(x)
        for condition, encoded, group_indices in zip(
            conditions, encoded_values, indices
        ):
            accumulator = self.accumulator_map.get(condition, self.default_accumulator)
            if accumulator is not None and encoded is not None:
                sequence = cast(
                    SequenceEncodableAccumulator[Any, Any, Any], accumulator
                )
                sequence.seq_initialize(encoded, weights[group_indices], rng)

    def seq_update(
        self,
        x: ConditionalEncoded | ConditionalEncodedData,
        weights: Array,
        estimate: Optional[Model],
    ) -> None:
        """Update accumulators from grouped encoded observations."""
        _size, conditions, encoded_values, indices, _conditions = _unwrap_encoded(x)
        conditional = (
            estimate if isinstance(estimate, ConditionalDistribution) else None
        )
        for condition, encoded, group_indices in zip(
            conditions, encoded_values, indices
        ):
            accumulator = self.accumulator_map.get(condition, self.default_accumulator)
            if accumulator is None or encoded is None:
                continue
            child_estimate = None
            if conditional is not None:
                child_estimate = conditional.dmap.get(condition)
                if child_estimate is None and conditional.has_default:
                    child_estimate = conditional.default_dist
            sequence = cast(SequenceEncodableAccumulator[Any, Any, Any], accumulator)
            sequence.seq_update(encoded, weights[group_indices], child_estimate)

    def combine(
        self, suff_stat: ConditionalSuffStat
    ) -> "ConditionalDistributionEstimatorAccumulator":
        """Merge explicit and default sufficient statistics."""
        for key, value in suff_stat[0].items():
            if key in self.accumulator_map:
                self.accumulator_map[key].combine(value)
        if self.default_accumulator is not None and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])
        return self

    def value(self) -> ConditionalSuffStat:
        """Return explicit statistics followed by optional default statistics."""
        default_value = (
            None
            if self.default_accumulator is None
            else self.default_accumulator.value()
        )
        return (
            {
                key: accumulator.value()
                for key, accumulator in self.accumulator_map.items()
            },
            default_value,
        )

    def from_value(
        self, x: ConditionalSuffStat
    ) -> "ConditionalDistributionEstimatorAccumulator":
        """Restore explicit and default sufficient statistics."""
        for key, value in x[0].items():
            if key in self.accumulator_map:
                self.accumulator_map[key].from_value(value)
        if self.default_accumulator is not None and x[1] is not None:
            self.default_accumulator.from_value(x[1])
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Delegate keyed statistic merging to every child accumulator."""
        for accumulator in self.accumulator_map.values():
            accumulator.key_merge(stats_dict)
        if self.default_accumulator is not None:
            self.default_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Delegate keyed statistic replacement to every child accumulator."""
        for accumulator in self.accumulator_map.values():
            accumulator.key_replace(stats_dict)
        if self.default_accumulator is not None:
            self.default_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "ConditionalDataEncoder":
        """Create an encoder matching accumulator child paths."""
        default_encoder = (
            NullDataEncoder()
            if self.default_accumulator is None
            else self.default_accumulator.acc_to_encoder()
        )
        return ConditionalDataEncoder(
            encoder_map={
                key: accumulator.acc_to_encoder()
                for key, accumulator in self.accumulator_map.items()
            },
            given_encoder=self.given_encoder,
            default_encoder=default_encoder,
            pass_value=self.pass_value,
        )


class ConditionalDistributionAccumulatorFactory(
    StatisticAccumulatorFactory[
        ConditionalObservation, ConditionalSuffStat, ConditionalEncoded
    ]
):
    """Create fresh keyed conditional accumulators."""

    def __init__(
        self,
        factory_map: Mapping[ConditionKey, AccumulatorFactory],
        default_factory: Optional[AccumulatorFactory],
        keys: Optional[str],
        given_encoder: Encoder,
        pass_value: bool,
    ) -> None:
        """Store child factories and conditional encoding metadata."""
        self.factory_map = dict(factory_map)
        self.default_factory = default_factory
        self.keys = keys
        self.given_encoder = given_encoder
        self.pass_value = pass_value

    def make(self) -> ConditionalDistributionEstimatorAccumulator:
        """Create an empty accumulator for every configured child."""
        default = None if self.default_factory is None else self.default_factory.make()
        return ConditionalDistributionEstimatorAccumulator(
            {key: factory.make() for key, factory in self.factory_map.items()},
            default,
            self.keys,
            self.given_encoder,
            self.pass_value,
        )


class ConditionalDistributionEstimator(
    ParameterEstimator[
        ConditionalObservation, Any, ConditionalEncoded, ConditionalSuffStat
    ]
):
    """Estimate explicit and default children while retaining the condition model.

    The condition distribution is fixed: it supplies sampling keys and an
    encoder but receives no sufficient statistics. Each explicit child is
    estimated from the matching entry in the statistic mapping, and the
    optional default child is estimated from the second statistic item.

    Args:
        estimator_map: Mapping from condition values to child estimators.
        default_estimator: Estimator for conditions absent from the mapping,
            or ``None`` to ignore those observations during fitting.
        keys: Compatibility metadata retained by the accumulator factory.
        cond_dist: Fixed distribution used to sample and encode conditions.
        pass_value: Whether child estimators receive complete observations.
    """

    def __init__(
        self,
        estimator_map: Mapping[ConditionKey, Estimator],
        default_estimator: Optional[Estimator] = None,
        keys: Optional[str] = None,
        cond_dist: Model = null_dist,
        pass_value: bool = False,
    ) -> None:
        """Initialize child estimators and fixed conditional metadata."""
        self.estimator_map = dict(estimator_map)
        self.default_estimator = default_estimator
        self.keys = keys
        self.cond_dist = cond_dist
        self.pass_value = pass_value

    def accumulator_factory(self) -> ConditionalDistributionAccumulatorFactory:
        """Create a concrete factory for explicit and default accumulators."""
        default_factory = (
            None
            if self.default_estimator is None
            else self.default_estimator.accumulator_factory()
        )
        return ConditionalDistributionAccumulatorFactory(
            {
                key: estimator.accumulator_factory()
                for key, estimator in self.estimator_map.items()
            },
            default_factory,
            self.keys,
            self.cond_dist.dist_to_encoder(),
            self.pass_value,
        )

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> ConditionalDistribution:
        """Estimate every configured child from matching sufficient statistics."""
        suff_stat = cast(ConditionalSuffStat, args[-1])
        default_dist = (
            null_dist
            if self.default_estimator is None or suff_stat[1] is None
            else self.default_estimator.estimate(suff_stat[1])
        )
        dist_map = {
            key: estimator.estimate(suff_stat[0][key])
            for key, estimator in self.estimator_map.items()
        }
        return ConditionalDistribution(
            dist_map,
            self.cond_dist,
            default_dist=default_dist,
            pass_value=self.pass_value,
        )


class ConditionalDataEncoder(
    DataSequenceEncoder[ConditionalObservation, ConditionalEncoded]
):
    """Encode conditional observations into aligned per-condition groups.

    The payload is ``(n, conditions, encoded_values, indices,
    encoded_conditions)``. The three middle tuples have one entry per distinct
    condition in first-observed order. Each index vector restores that group's
    positions among the ``n`` observations. An unknown condition without a
    default encoder has ``None`` as its encoded value.

    Args:
        encoder_map: Mapping from condition values to child encoders.
        given_encoder: Encoder for the condition elements themselves.
        default_encoder: Encoder for conditions absent from ``encoder_map``.
        pass_value: Whether child encoders receive complete observations.
    """

    def __init__(
        self,
        encoder_map: Mapping[ConditionKey, Encoder],
        given_encoder: Encoder,
        default_encoder: Encoder,
        pass_value: bool = False,
    ) -> None:
        """Initialize explicit, condition, and default encoders."""
        self.encoder_map = dict(encoder_map)
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder
        self.pass_value = pass_value
        self.null_default_encoder = isinstance(default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(given_encoder, NullDataEncoder)

    def __eq__(self, other: object) -> bool:
        """Return whether all child encoders and pass-through behavior match."""
        return (
            isinstance(other, ConditionalDataEncoder)
            and self.encoder_map == other.encoder_map
            and self.default_encoder == other.default_encoder
            and self.given_encoder == other.given_encoder
            and self.pass_value == other.pass_value
        )

    def __str__(self) -> str:
        """Return a stable constructor-like encoder representation."""
        entries = ",".join(
            f"{key!r}:{encoder}" for key, encoder in self.encoder_map.items()
        )
        default = None if self.null_default_encoder else self.default_encoder
        given = None if self.null_given_encoder else self.given_encoder
        return (
            f"ConditionalDataEncoder({{{entries}}}, default={default}, "
            f"given={given}, pass_value={self.pass_value!r})"
        )

    def seq_encode(
        self, x: Iterable[ConditionalObservation]
    ) -> "ConditionalEncodedData":
        """Group values by condition and encode all five payload components."""
        observations = tuple(x)
        grouped: dict[ConditionKey, tuple[list[Any], list[int]]] = {}
        for index, observation in enumerate(observations):
            values, positions = grouped.setdefault(observation[0], ([], []))
            values.append(observation if self.pass_value else observation[1])
            positions.append(index)

        conditions: list[ConditionKey] = []
        encoded_values: list[Optional[Any]] = []
        indices: list[Array] = []
        for condition, (values, positions) in grouped.items():
            conditions.append(condition)
            encoder = self.encoder_map.get(condition)
            if encoder is None and not self.null_default_encoder:
                encoder = self.default_encoder
            encoded_values.append(
                None if encoder is None else encoder.seq_encode(values).data
            )
            indices.append(np.asarray(positions, dtype=int))

        encoded_conditions = self.given_encoder.seq_encode(
            observation[0] for observation in observations
        ).data
        return ConditionalEncodedData(
            (
                len(observations),
                tuple(conditions),
                tuple(encoded_values),
                tuple(indices),
                encoded_conditions,
            )
        )


class ConditionalEncodedData(EncodedDataSequence[ConditionalEncoded]):
    """Contain the stable five-part conditional sequence encoding.

    ``data`` is ``(size, conditions, encoded_values, indices,
    encoded_conditions)``. The condition, value, and index tuples always have
    equal lengths, including when an unknown condition has no default encoder.
    """

    def __init__(self, data: ConditionalEncoded) -> None:
        """Store a five-part conditional encoding."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise representation containing the encoded payload."""
        return f"ConditionalEncodedData(data={self.data!r})"
