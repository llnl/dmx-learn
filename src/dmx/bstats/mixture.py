"""Finite Bayesian mixtures with ordered, homogeneous components.

Mixture weights are finite, nonnegative, normalized on input, and positionally
paired with components. Scalar and sequence scoring use stable log-sum-exp. Posterior
responsibilities have shape ``(K,)`` for one observation and ``(N, K)`` for a
sequence. Priors are an ordered composite of the weight prior and component
priors; conjugate weight priors supply expected log weights, while alternate
priors use plug-in log weights.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional, cast

import numpy as np

from dmx.arithmetic import maxint
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.dirichlet import DirichletDistribution
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
from dmx.bstats.symdirichlet import SymmetricDirichletDistribution
from dmx.utils.special import digamma

# Bayesian distributions intentionally implement the supported legacy protocol.
# pylint: disable=abstract-method

Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Encoder = DataSequenceEncoder[Any, Any]
Array = np.ndarray[Any, np.dtype[np.float64]]
MixtureParameters = tuple[Array, list[Any]]
MixtureSuffStat = tuple[Array, tuple[Any, ...]]
MixtureKeys = tuple[Optional[str], Optional[str]]

default_prior = SymmetricDirichletDistribution(1.0)


def _logsumexp(values: Array, axis: int, keepdims: bool = False) -> Array:
    """Reduce log values stably along one axis."""
    return cast(Array, np.logaddexp.reduce(values, axis=axis, keepdims=keepdims))


def _normalized_weights(values: Sequence[float] | np.ndarray[Any, Any]) -> Array:
    """Return finite nonnegative weights normalized to sum to one."""
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Mixture weights must be a nonempty vector.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Mixture weights must be finite and nonnegative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one mixture weight must be positive.")
    return cast(Array, weights / total)


def _encoded_value(x: Any) -> Any:
    """Unwrap the dedicated encoder while accepting legacy raw encodings."""
    return x.data if isinstance(x, MixtureEncodedData) else x


class MixtureDistribution(ProbabilityDistribution[Any, MixtureParameters, Any]):
    """Model observations with an ordered finite mixture.

    All components must accept the same observation and encoded-sequence
    representation. The length-``K`` weight vector is normalized on input and
    paired positionally with ``K`` components. ``posterior`` returns a
    length-``K`` responsibility vector, while ``seq_posterior`` returns an
    ``(N, K)`` matrix for ``N`` encoded observations.

    A complete prior is ``(weight_prior, component_priors)`` represented by a
    nested :class:`CompositeDistribution`. Passing only a weight prior leaves
    existing component priors unchanged.

    Args:
        components: Ordered homogeneous component distributions.
        w: Finite nonnegative component weights with at least one positive
            entry.
        name: Optional identifier for the mixture.
        prior: Weight prior, or a complete composite prior when set later.

    Raises:
        ValueError: If there are no components, counts differ, or weights are
            invalid.
    """

    def __init__(
        self,
        components: Sequence[Model],
        w: Sequence[float] | np.ndarray[Any, Any],
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize homogeneous components, normalized weights, and a prior."""
        super().__init__()
        self.components = tuple(components)
        self.num_components = len(self.components)
        if not self.components:
            raise ValueError("A mixture requires at least one component.")
        if len(w) != self.num_components:
            raise ValueError("Mixture weights and components must have equal length.")
        self.set_name(name)
        self._set_weights(w)
        self.conj_prior_params: Optional[Array] = None
        self.expected_nparams = self.log_w.copy()
        self.set_prior(
            prior
            if prior is not None
            else DirichletDistribution(np.ones(self.num_components))
        )

    def __str__(self) -> str:
        """Return the legacy constructor-like representation."""
        components = ",".join(map(str, self.components))
        weights = ",".join(map(str, self.w))
        return (
            f"MixtureDistribution([{components}], [{weights}], "
            f"name={self.name!r}, prior={self.prior})"
        )

    def _set_weights(self, values: Sequence[float] | np.ndarray[Any, Any]) -> None:
        """Validate weights and refresh cached log weights."""
        self.w = _normalized_weights(values)
        self.zw = self.w == 0.0
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)

    def _set_expected_weights(self) -> None:
        """Cache expected log weights for the configured prior."""
        concentrations: Optional[Array]
        if isinstance(self.prior, DirichletDistribution):
            parameters = self.prior.get_parameters()
            concentrations = (
                np.full(self.num_components, parameters, dtype=np.float64)
                if isinstance(parameters, float)
                else np.asarray(parameters, dtype=np.float64)
            )
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            concentrations = np.full(
                self.num_components, self.prior.get_parameters(), dtype=np.float64
            )
        else:
            concentrations = None
        if concentrations is not None and len(concentrations) != self.num_components:
            raise ValueError("Weight prior dimension must match mixture components.")
        self.conj_prior_params = concentrations
        self.expected_nparams = (
            self.log_w.copy()
            if concentrations is None
            else np.asarray(
                digamma(concentrations) - digamma(concentrations.sum()),
                dtype=np.float64,
            )
        )

    def get_prior(self) -> CompositeDistribution:
        """Return ordered weight and component priors."""
        return CompositeDistribution(
            (
                self.prior,
                CompositeDistribution(tuple(x.get_prior() for x in self.components)),
            )
        )

    def set_prior(self, prior: Model) -> None:
        """Replace the weight prior or propagate a complete composite prior."""
        if isinstance(prior, CompositeDistribution):
            if len(prior.dists) != 2 or not isinstance(
                prior.dists[1], CompositeDistribution
            ):
                raise ValueError("Mixture prior must contain weights and components.")
            component_priors = prior.dists[1].dists
            if len(component_priors) != self.num_components:
                raise ValueError("Component prior count must match the mixture.")
            self.prior = prior.dists[0]
            for component, component_prior in zip(self.components, component_priors):
                component.set_prior(component_prior)
        else:
            self.prior = prior
        self._set_expected_weights()

    def get_parameters(self) -> MixtureParameters:
        """Return weights and component parameters in component order."""
        return self.w.copy(), [x.get_parameters() for x in self.components]

    def set_parameters(self, value: MixtureParameters) -> None:
        """Replace weights and ordered component parameters."""
        weights, parameters = value
        if len(parameters) != self.num_components:
            raise ValueError("Component parameter count must match the mixture.")
        self._set_weights(weights)
        for component, component_parameters in zip(self.components, parameters):
            component.set_parameters(component_parameters)
        self._set_expected_weights()

    def log_density(self, x: Any) -> float:
        """Score one observation with stable weighted log-sum-exp."""
        values = np.asarray([u.log_density(x) for u in self.components], dtype=float)
        return float(_logsumexp(values + self.log_w, axis=0))

    def expected_log_density(self, x: Any) -> float:
        """Score using component expectations and expected log weights."""
        values = np.asarray(
            [u.expected_log_density(x) for u in self.components], dtype=float
        )
        return float(_logsumexp(values + self.expected_nparams, axis=0))

    def seq_component_log_density(self, x: Any) -> Array:
        """Return an ``(N, K)`` matrix of component log scores."""
        encoded = _encoded_value(x)
        return cast(
            Array,
            np.asarray(
                [u.seq_log_density(encoded) for u in self.components],
                dtype=np.float64,
            ).T,
        )

    def seq_log_density(self, x: Any) -> Array:
        """Return stable mixture scores for an encoded sequence."""
        return np.asarray(
            _logsumexp(self.seq_component_log_density(x) + self.log_w, axis=1),
            dtype=np.float64,
        )

    def seq_expected_log_density(self, x: Any) -> Array:
        """Return prior-expected scores for an encoded sequence."""
        encoded = _encoded_value(x)
        values = np.asarray(
            [u.seq_expected_log_density(encoded) for u in self.components],
            dtype=np.float64,
        ).T
        return np.asarray(
            _logsumexp(values + self.expected_nparams, axis=1), dtype=np.float64
        )

    def _responsibilities(self, component_values: Array) -> Array:
        """Normalize weighted component log scores row by row."""
        weighted = component_values + self.log_w
        normalizers = _logsumexp(weighted, axis=1, keepdims=True)
        result = np.empty_like(weighted, dtype=np.float64)
        valid = np.isfinite(normalizers[:, 0])
        result[valid] = np.exp(weighted[valid] - normalizers[valid])
        result[~valid] = self.w
        return result

    def posterior(self, x: Any) -> Array:
        """Return the length-``K`` responsibility vector for one observation."""
        values = np.asarray([u.log_density(x) for u in self.components], dtype=float)
        return cast(Array, self._responsibilities(values[None, :])[0])

    def seq_posterior(self, x: Any) -> Array:
        """Return posterior responsibilities shaped ``(N, K)``."""
        return self._responsibilities(self.seq_component_log_density(x))

    def seq_encode(self, x: Iterable[Any]) -> Any:
        """Encode observations with the first homogeneous component."""
        return self.components[0].seq_encode(x)

    def sampler(self, seed: Optional[int] = None) -> "MixtureSampler":
        """Create a repeatable mixture sampler."""
        return MixtureSampler(self, seed)

    def estimator(self) -> "MixtureEstimator":
        """Create an estimator preserving component order and priors."""
        return MixtureEstimator(
            tuple(x.estimator() for x in self.components),
            name=self.name,
            prior=self.prior,
        )

    def dist_to_encoder(self) -> "MixtureDataEncoder":
        """Create the common component encoder."""
        return MixtureDataEncoder(self.components[0].dist_to_encoder())


class MixtureSampler(DistributionSampler[Any]):
    """Draw observations after sampling an ordered component index."""

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None) -> None:
        """Initialize independent selection and component random states."""
        seed_source = np.random.RandomState(seed)
        super().__init__(dist, int(seed_source.randint(maxint)))
        self.dist = dist
        self.comp_samplers = tuple(
            x.sampler(seed=int(seed_source.randint(maxint))) for x in dist.components
        )
        self.compSamplers = self.comp_samplers

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one observation or a list of ``size`` observations."""
        states = self.rng.choice(
            self.dist.num_components, size=size, replace=True, p=self.dist.w
        )
        if size is None:
            return self.comp_samplers[int(states)].sample()
        return [self.comp_samplers[int(index)].sample() for index in states]


class MixtureEstimatorAccumulator(
    SequenceEncodableAccumulator[Any, MixtureSuffStat, Any]
):
    """Accumulate counts and ordered child sufficient statistics.

    The sufficient statistic is ``(component_counts, child_statistics)``.
    ``component_counts`` has shape ``(K,)`` and the second item is a
    length-``K`` tuple in component order. Updates distribute each observation
    weight according to posterior responsibilities.

    The two optional keys independently share the component-count vector and
    the complete collection of child statistics. Child accumulator keys are
    also honored.
    """

    def __init__(
        self, accumulators: Sequence[Accumulator], keys: MixtureKeys = (None, None)
    ) -> None:
        """Initialize child accumulators and optional sharing keys."""
        self.accumulators = list(accumulators)
        self.num_components = len(self.accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=np.float64)
        self.weight_key, self.comp_key = keys

    @staticmethod
    def _estimate(value: Optional[Model]) -> MixtureDistribution:
        """Require the current mixture used for responsibilities."""
        if not isinstance(value, MixtureDistribution):
            raise TypeError("Mixture accumulation requires a mixture estimate.")
        return value

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Allocate one observation by posterior responsibility."""
        mixture = self._estimate(estimate)
        responsibilities = mixture.posterior(x) * weight
        self.comp_counts += responsibilities
        for index, accumulator in enumerate(self.accumulators):
            accumulator.update(
                x, float(responsibilities[index]), mixture.components[index]
            )

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Randomly split one weight with the legacy uniform Dirichlet."""
        allocations = (
            np.zeros(self.num_components)
            if weight == 0.0
            else weight * rng.dirichlet(np.ones(self.num_components))
        )
        for index, accumulator in enumerate(self.accumulators):
            allocation = float(allocations[index])
            accumulator.initialize(x, allocation, rng)
            self.comp_counts[index] += allocation

    def seq_initialize(
        self, x: Any, weights: Array, rng: np.random.RandomState
    ) -> None:
        """Randomly split sequence weights over ordered components."""
        encoded = _encoded_value(x)
        allocations = np.zeros((len(weights), self.num_components), dtype=float)
        active = weights != 0.0
        allocations[active] = rng.dirichlet(
            np.ones(self.num_components), size=int(np.count_nonzero(active))
        )
        allocations *= weights[:, None]
        for index, accumulator in enumerate(self.accumulators):
            sequence_accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any], accumulator
            )
            sequence_accumulator.seq_initialize(encoded, allocations[:, index], rng)
            self.comp_counts[index] += float(allocations[:, index].sum())

    def seq_update(self, x: Any, weights: Array, estimate: Optional[Model]) -> None:
        """Allocate a sequence by posterior responsibility."""
        mixture = self._estimate(estimate)
        encoded = _encoded_value(x)
        responsibilities = mixture.seq_posterior(encoded) * weights[:, None]
        for index, accumulator in enumerate(self.accumulators):
            component_weights = responsibilities[:, index]
            self.comp_counts[index] += float(component_weights.sum())
            sequence_accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any], accumulator
            )
            sequence_accumulator.seq_update(
                encoded, component_weights, mixture.components[index]
            )

    def combine(self, suff_stat: MixtureSuffStat) -> "MixtureEstimatorAccumulator":
        """Merge counts and child sufficient statistics."""
        self.comp_counts += suff_stat[0]
        for accumulator, value in zip(self.accumulators, suff_stat[1]):
            accumulator.combine(value)
        return self

    def value(self) -> MixtureSuffStat:
        """Return copied counts and ordered child statistics."""
        return self.comp_counts.copy(), tuple(x.value() for x in self.accumulators)

    def from_value(self, x: MixtureSuffStat) -> "MixtureEstimatorAccumulator":
        """Restore counts and ordered child statistics."""
        self.comp_counts = np.asarray(x[0], dtype=np.float64).copy()
        for accumulator, value in zip(self.accumulators, x[1]):
            accumulator.from_value(value)
        return self

    @staticmethod
    def _supports_key_merge(accumulator: Accumulator) -> bool:
        """Return whether a child implements optional keyed merging."""
        return type(accumulator).key_merge is not StatisticAccumulator.key_merge

    @staticmethod
    def _supports_key_replace(accumulator: Accumulator) -> bool:
        """Return whether a child implements optional keyed replacement."""
        return type(accumulator).key_replace is not StatisticAccumulator.key_replace

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge mixture blocks and supported child keys."""
        if self.weight_key is not None:
            counts = stats_dict.get(self.weight_key)
            if isinstance(counts, np.ndarray):
                counts += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts.copy()
        if self.comp_key is not None:
            existing = stats_dict.get(self.comp_key)
            if isinstance(existing, MixtureEstimatorAccumulator):
                for accumulator, value in zip(existing.accumulators, self.value()[1]):
                    accumulator.combine(value)
            else:
                stats_dict[self.comp_key] = copy.deepcopy(self)
        for accumulator in self.accumulators:
            if self._supports_key_merge(accumulator):
                accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace mixture blocks and supported child keys."""
        if self.weight_key is not None:
            counts = stats_dict.get(self.weight_key)
            if isinstance(counts, np.ndarray):
                self.comp_counts = counts.copy()
        if self.comp_key is not None:
            existing = stats_dict.get(self.comp_key)
            if isinstance(existing, MixtureEstimatorAccumulator):
                for accumulator, source in zip(
                    self.accumulators, existing.accumulators
                ):
                    accumulator.from_value(source.value())
        for accumulator in self.accumulators:
            if self._supports_key_replace(accumulator):
                accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "MixtureDataEncoder":
        """Create the common child accumulator encoder."""
        return MixtureDataEncoder(self.accumulators[0].acc_to_encoder())


class MixtureEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[Any, MixtureSuffStat, Any]
):
    """Create mixture accumulators from ordered child factories."""

    def __init__(
        self,
        factories: Sequence[AccumulatorFactory],
        dim: int,
        keys: MixtureKeys,
    ) -> None:
        """Store child factories, legacy dimension, and keys."""
        self.factories = tuple(factories)
        self.dim = dim
        self.keys = keys

    def make(self) -> MixtureEstimatorAccumulator:
        """Create one fresh accumulator per component."""
        return MixtureEstimatorAccumulator(
            tuple(self.factories[index].make() for index in range(self.dim)),
            self.keys,
        )


class MixtureEstimator(
    ParameterEstimator[Any, MixtureParameters, Any, MixtureSuffStat]
):
    """Estimate ordered components and normalized finite-mixture weights.

    Each child estimator consumes the statistic at its component position.
    With a Dirichlet weight prior, weights are posterior modes when that mode
    has positive mass and posterior means otherwise. Without a conjugate
    prior, normalized effective counts are used. ``fixed_w`` overrides both
    rules while child components are still estimated.

    Args:
        estimators: Ordered homogeneous component estimators.
        fixed_w: Optional fixed component weights, normalized on input.
        name: Optional identifier copied to estimated mixtures.
        prior: Prior for the component weights.
        keys: Pair ``(weight_key, component_key)`` controlling independent
            sufficient-statistic sharing.
    """

    def __init__(
        self,
        estimators: Sequence[Estimator],
        fixed_w: Optional[Sequence[float] | np.ndarray[Any, Any]] = None,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: MixtureKeys = (None, None),
    ) -> None:
        """Initialize child estimators, optional fixed weights, and priors."""
        self.estimators = tuple(estimators)
        self.num_components = len(self.estimators)
        if not self.estimators:
            raise ValueError("A mixture estimator requires at least one component.")
        self.prior = prior
        self.keys = keys
        self.name = name
        self.fixed_w = None if fixed_w is None else _normalized_weights(fixed_w)
        if self.fixed_w is not None and len(self.fixed_w) != self.num_components:
            raise ValueError("Fixed weights and estimators must have equal length.")

    def accumulator_factory(self) -> MixtureEstimatorAccumulatorFactory:
        """Create a factory preserving estimator order and sharing keys."""
        factories = tuple(x.accumulator_factory() for x in self.estimators)
        return MixtureEstimatorAccumulatorFactory(
            factories, self.num_components, self.keys
        )

    def get_prior(self) -> CompositeDistribution:
        """Return ordered weight and component estimator priors."""
        return CompositeDistribution(
            (
                self.prior,
                CompositeDistribution(
                    tuple(x.get_prior() for x in self.estimators),
                    name=self.keys[1],
                ),
            )
        )

    def set_prior(self, prior: Model) -> None:
        """Replace weights or propagate a complete ordered composite prior."""
        if not isinstance(prior, CompositeDistribution):
            self.prior = prior
            return
        if len(prior.dists) != 2 or not isinstance(
            prior.dists[1], CompositeDistribution
        ):
            raise ValueError("Mixture prior must contain weights and components.")
        component_priors = prior.dists[1].dists
        if len(component_priors) != self.num_components:
            raise ValueError("Component prior count must match mixture estimators.")
        self.prior = prior.dists[0]
        for estimator, component_prior in zip(self.estimators, component_priors):
            estimator.set_prior(component_prior)

    def model_log_density(self, model: MixtureDistribution) -> float:
        """Score mixture and component parameters under the complete prior."""
        return float(self.get_prior().log_density(model.get_parameters()))

    def _weight_concentrations(self) -> Optional[Array]:
        """Return conjugate concentrations for the mixture dimension."""
        if isinstance(self.prior, DirichletDistribution):
            parameters = self.prior.get_parameters()
            values = (
                np.full(self.num_components, parameters, dtype=np.float64)
                if isinstance(parameters, float)
                else np.asarray(parameters, dtype=np.float64)
            )
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            values = np.full(
                self.num_components, self.prior.get_parameters(), dtype=np.float64
            )
        else:
            return None
        if len(values) != self.num_components:
            raise ValueError("Weight prior dimension must match mixture estimators.")
        return values

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> MixtureDistribution:
        """Estimate components and normalized MAP or empirical weights."""
        counts, component_stats = cast(MixtureSuffStat, args[-1])
        if len(counts) != self.num_components:
            raise ValueError("Component counts must match mixture estimators.")
        components = tuple(
            estimator.estimate(component_stat)
            for estimator, component_stat in zip(self.estimators, component_stats)
        )
        if self.fixed_w is not None:
            return MixtureDistribution(
                components, self.fixed_w, name=self.name, prior=self.prior
            )
        concentrations = self._weight_concentrations()
        if concentrations is not None:
            posterior = np.asarray(counts, dtype=np.float64) + concentrations
            map_mass = np.maximum(posterior - 1.0, 0.0)
            weights = (
                map_mass / map_mass.sum()
                if map_mass.sum() > 0.0
                else posterior / posterior.sum()
            )
            return MixtureDistribution(
                components,
                weights,
                name=self.name,
                prior=DirichletDistribution(posterior),
            )
        total = float(np.sum(counts))
        weights = (
            np.full(self.num_components, 1.0 / self.num_components)
            if total <= 0.0
            else np.asarray(counts, dtype=np.float64) / total
        )
        return MixtureDistribution(
            components, weights, name=self.name, prior=self.prior
        )


class MixtureDataEncoder(DataSequenceEncoder[Any, Any]):
    """Encode homogeneous mixture observations with one child encoder.

    Every component must consume this common payload; component-specific
    encodings are not stored separately.

    Args:
        encoder: Encoder shared by all mixture components.
    """

    def __init__(self, encoder: Encoder) -> None:
        """Store the common component encoder."""
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a stable constructor-like representation."""
        return f"MixtureDataEncoder({self.encoder})"

    def __eq__(self, other: object) -> bool:
        """Return whether wrapped component encoders are equal."""
        return isinstance(other, MixtureDataEncoder) and self.encoder == other.encoder

    def seq_encode(self, x: Iterable[Any]) -> "MixtureEncodedData":
        """Encode observations and unwrap the child encoder container."""
        return MixtureEncodedData(self.encoder.seq_encode(x).data)


class MixtureEncodedData(EncodedDataSequence[Any]):
    """Contain the common component encoding for a mixture sequence."""

    def __init__(self, data: Any) -> None:
        """Store component-encoded observations."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return f"MixtureEncodedData(data={self.data!r})"


MixtureEncodedDataSequence = MixtureEncodedData
