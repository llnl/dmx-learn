"""Bayesian product distributions for heterogeneous tuple observations.

``CompositeDistribution`` preserves component order: observation element
``x[i]`` is scored, encoded, sampled, and estimated by ``dists[i]``. The joint
log-density and prior-expected log-density are sums because components are
conditionally independent. A composite name labels the product while child
names continue to control component DataFrame operations.

The optional composite key shares the complete ordered tuple of child
sufficient statistics between accumulators. Child accumulator keys are also
merged and replaced independently, so composite and child keys should be
distinct. Priors have the same ordered product structure as parameters.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from dmx.bstats.pdist import (
    DataFrameEncodableAccumulator,
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)

# Components erase their individual generic arguments at the heterogeneous product
# boundary. Their positions retain the runtime type relationship.
Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Encoder = DataSequenceEncoder[Any, Any]
Observation = tuple[Any, ...]
Parameters = tuple[Any, ...]
CompositeEncoded = tuple[Any, ...]
CompositeSuffStat = tuple[Any, ...]
Array = np.ndarray[Any, Any]


class CompositeDistribution(
    ProbabilityDistribution[Observation, Parameters, CompositeEncoded]
):
    """Model an ordered tuple as a product of heterogeneous distributions.

    ``name`` identifies the product itself. ``keys`` is copied into the
    estimator and controls whole-product sufficient-statistic sharing; it does
    not replace or reorder component-level keys.
    """

    def __init__(
        self,
        dists: Sequence[Model],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize ordered component distributions and metadata."""
        super().__init__()
        self.dists: tuple[Model, ...] = tuple(dists)
        self.count = len(self.dists)
        self.keys = keys
        self.set_name(name)

    def __str__(self) -> str:
        """Return the legacy constructor-like representation."""
        components = ",".join(map(str, self.dists))
        if self.count == 1:
            components += ","
        return f"CompositeDistribution(({components}))"

    def get_parameters(self) -> Parameters:
        """Return child parameters in component order."""
        return tuple(dist.get_parameters() for dist in self.dists)

    def set_parameters(self, value: Parameters) -> None:
        """Set child parameters position by position."""
        for dist, parameters in zip(self.dists, value):
            dist.set_parameters(parameters)

    def get_prior(self) -> "CompositeDistribution":
        """Return the ordered product of component priors."""
        return CompositeDistribution(
            tuple(dist.get_prior() for dist in self.dists),
            name=self.name,
            keys=self.keys,
        )

    def set_prior(self, prior: Model) -> None:
        """Propagate an ordered composite prior to every component.

        Args:
            prior: Composite distribution whose children are component priors.

        Raises:
            TypeError: If ``prior`` is not a composite distribution.
        """
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("CompositeDistribution requires a composite prior.")
        for dist, component_prior in zip(self.dists, prior.dists):
            dist.set_prior(component_prior)

    def cross_entropy(self, dist: Model) -> float:
        """Sum component cross-entropies in their fixed order."""
        if isinstance(dist, CompositeDistribution):
            return float(
                sum(
                    component.cross_entropy(other)
                    for component, other in zip(self.dists, dist.dists)
                )
            )
        return float(sum(component.cross_entropy(dist) for component in self.dists))

    def entropy(self) -> float:
        """Return the sum of independent component entropies."""
        return float(sum(component.entropy() for component in self.dists))

    def log_density(self, x: Observation) -> float:
        """Sum child log-densities for one ordered tuple observation."""
        return float(sum(dist.log_density(value) for dist, value in zip(self.dists, x)))

    def expected_log_density(self, x: Observation) -> float:
        """Sum child prior-expected log-densities for one observation."""
        return float(
            sum(dist.expected_log_density(value) for dist, value in zip(self.dists, x))
        )

    def seq_encode(self, x: Iterable[Observation]) -> CompositeEncoded:
        """Encode each tuple position with its corresponding component."""
        observations = tuple(x)
        return tuple(
            dist.seq_encode(observation[index] for observation in observations)
            for index, dist in enumerate(self.dists)
        )

    @staticmethod
    def _encoded_values(
        x: CompositeEncoded | "CompositeEncodedData",
    ) -> CompositeEncoded:
        """Unwrap encoder containers while retaining legacy tuple inputs."""
        return x.data if isinstance(x, CompositeEncodedData) else x

    def seq_log_density(self, x: CompositeEncoded | "CompositeEncodedData") -> Array:
        """Sum vectorized child log-densities elementwise."""
        encoded = self._encoded_values(x)
        values = [
            np.asarray(dist.seq_log_density(encoded[index]), dtype=float)
            for index, dist in enumerate(self.dists)
        ]
        return np.asarray(np.sum(values, axis=0, dtype=float), dtype=float)

    def seq_expected_log_density(
        self, x: CompositeEncoded | "CompositeEncodedData"
    ) -> Array:
        """Sum vectorized child prior-expected log-densities elementwise."""
        encoded = self._encoded_values(x)
        values = [
            np.asarray(dist.seq_expected_log_density(encoded[index]), dtype=float)
            for index, dist in enumerate(self.dists)
        ]
        return np.asarray(np.sum(values, axis=0, dtype=float), dtype=float)

    def df_log_density(self, df: pd.DataFrame) -> pd.Series:
        """Sum child DataFrame log-densities using component names."""
        values = [dist.df_log_density(df) for dist in self.dists]
        return sum(values[1:], start=values[0].copy())

    def sampler(self, seed: Optional[int] = None) -> "CompositeSampler":
        """Create a sampler with one independently seeded child sampler."""
        return CompositeSampler(self, seed)

    def estimator(self) -> "CompositeEstimator":
        """Create an estimator preserving component order and metadata."""
        return CompositeEstimator(
            tuple(dist.estimator() for dist in self.dists),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "CompositeDataEncoder":
        """Create an ordered product of component encoders."""
        return CompositeDataEncoder(
            tuple(dist.dist_to_encoder() for dist in self.dists)
        )


class CompositeSampler(DistributionSampler[Observation]):
    """Draw ordered tuples from the component distributions."""

    def __init__(self, dist: CompositeDistribution, seed: Optional[int] = None) -> None:
        """Initialize child samplers from deterministic child seeds."""
        super().__init__(dist, seed)
        self.dist_samplers: tuple[DistributionSampler[Any], ...] = tuple(
            component.sampler(seed=self.new_seed()) for component in dist.dists
        )
        # Keep the legacy public attribute spelling used by downstream code.
        self.distSamplers = self.dist_samplers

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one tuple or a list of ``size`` tuples."""
        if size is None:
            return tuple(sampler.sample() for sampler in self.dist_samplers)
        return list(zip(*(sampler.sample(size=size) for sampler in self.dist_samplers)))


class CompositeEstimatorAccumulator(
    SequenceEncodableAccumulator[Observation, CompositeSuffStat, CompositeEncoded],
    DataFrameEncodableAccumulator[Observation, CompositeSuffStat, CompositeEncoded],
):
    """Accumulate one ordered sufficient-statistic value per component.

    A non-``None`` composite key shares the entire tuple returned by
    :meth:`value`. After that whole-product operation, supported child keys are
    merged or replaced as well.
    """

    def __init__(
        self, accumulators: Sequence[Accumulator], keys: Optional[str] = None
    ) -> None:
        """Initialize ordered child accumulators and the sharing key."""
        self.accumulators: list[Accumulator] = list(accumulators)
        self.count = len(self.accumulators)
        self.key = keys

    def update(self, x: Observation, weight: float, estimate: Optional[Model]) -> None:
        """Update every child from the matching observation position."""
        component_estimates = (
            estimate.dists
            if isinstance(estimate, CompositeDistribution)
            else (None,) * self.count
        )
        for index, accumulator in enumerate(self.accumulators):
            accumulator.update(x[index], weight, component_estimates[index])

    def initialize(
        self, x: Observation, weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize every child from the matching observation position."""
        for index, accumulator in enumerate(self.accumulators):
            accumulator.initialize(x[index], weight, rng)

    def seq_initialize(
        self,
        x: CompositeEncoded,
        weights: Array,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize every child from its encoded sequence position."""
        for index, accumulator in enumerate(self.accumulators):
            sequence_accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any], accumulator
            )
            sequence_accumulator.seq_initialize(x[index], weights, rng)

    def seq_update(
        self,
        x: CompositeEncoded,
        weights: Array,
        estimate: Optional[Model],
    ) -> None:
        """Update every child from its encoded sequence position."""
        component_estimates = (
            estimate.dists
            if isinstance(estimate, CompositeDistribution)
            else (None,) * self.count
        )
        for index, accumulator in enumerate(self.accumulators):
            sequence_accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any], accumulator
            )
            sequence_accumulator.seq_update(
                x[index], weights, component_estimates[index]
            )

    def df_initialize(
        self,
        df: pd.DataFrame,
        weights: Iterable[float],
        rng: np.random.RandomState,
    ) -> None:
        """Initialize child accumulators from their named DataFrame columns."""
        for accumulator in self.accumulators:
            dataframe_accumulator = cast(
                DataFrameEncodableAccumulator[Any, Any, Any], accumulator
            )
            dataframe_accumulator.df_initialize(df, weights, rng)

    def df_update(
        self,
        df: pd.DataFrame,
        weights: Iterable[float],
        estimate: Optional[Model],
    ) -> None:
        """Update child accumulators from their named DataFrame columns."""
        component_estimates = (
            estimate.dists
            if isinstance(estimate, CompositeDistribution)
            else (None,) * self.count
        )
        for index, accumulator in enumerate(self.accumulators):
            dataframe_accumulator = cast(
                DataFrameEncodableAccumulator[Any, Any, Any], accumulator
            )
            dataframe_accumulator.df_update(df, weights, component_estimates[index])

    def combine(self, suff_stat: CompositeSuffStat) -> "CompositeEstimatorAccumulator":
        """Merge ordered child sufficient statistics."""
        for accumulator, value in zip(self.accumulators, suff_stat):
            accumulator.combine(value)
        return self

    def value(self) -> CompositeSuffStat:
        """Return child sufficient statistics in component order."""
        return tuple(accumulator.value() for accumulator in self.accumulators)

    def from_value(self, x: CompositeSuffStat) -> "CompositeEstimatorAccumulator":
        """Restore child sufficient statistics in component order."""
        for accumulator, value in zip(self.accumulators, x):
            accumulator.from_value(value)
        return self

    @staticmethod
    def _supports_key_merge(accumulator: Accumulator) -> bool:
        """Return whether a child implements optional key merging."""
        return type(accumulator).key_merge is not StatisticAccumulator.key_merge

    @staticmethod
    def _supports_key_replace(accumulator: Accumulator) -> bool:
        """Return whether a child implements optional key replacement."""
        return type(accumulator).key_replace is not StatisticAccumulator.key_replace

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge the whole product key, then independent supported child keys."""
        if self.key is not None:
            existing = stats_dict.get(self.key)
            if isinstance(existing, CompositeEstimatorAccumulator):
                existing.combine(self.value())
            else:
                # Keep the product-level entry independent from child-key entries;
                # otherwise both keys can alias one accumulator and double-count.
                stats_dict[self.key] = copy.deepcopy(self)
        for accumulator in self.accumulators:
            if self._supports_key_merge(accumulator):
                accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace from the whole product key, then supported child keys."""
        if self.key is not None:
            existing = stats_dict.get(self.key)
            if isinstance(existing, CompositeEstimatorAccumulator):
                self.from_value(existing.value())
        for accumulator in self.accumulators:
            if self._supports_key_replace(accumulator):
                accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "CompositeDataEncoder":
        """Create an ordered product of child accumulator encoders."""
        return CompositeDataEncoder(
            tuple(accumulator.acc_to_encoder() for accumulator in self.accumulators)
        )


class CompositeAccumulatorFactory(
    StatisticAccumulatorFactory[Observation, CompositeSuffStat, CompositeEncoded]
):
    """Create composite accumulators from ordered child factories."""

    def __init__(
        self, factories: Sequence[AccumulatorFactory], keys: Optional[str]
    ) -> None:
        """Store ordered child factories and the whole-product key."""
        self.factories = tuple(factories)
        self.keys = keys

    def make(self) -> CompositeEstimatorAccumulator:
        """Create a fresh accumulator for every component."""
        return CompositeEstimatorAccumulator(
            tuple(factory.make() for factory in self.factories), keys=self.keys
        )


class CompositeEstimator(
    ParameterEstimator[Observation, Parameters, CompositeEncoded, CompositeSuffStat]
):
    """Estimate an ordered product using one estimator per component."""

    def __init__(
        self,
        estimators: Sequence[Estimator],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize ordered child estimators and product metadata."""
        self.estimators = tuple(estimators)
        self.count = len(self.estimators)
        self.keys = keys
        self.name = name

    def get_prior(self) -> CompositeDistribution:
        """Return the ordered product of child estimator priors."""
        return CompositeDistribution(
            tuple(estimator.get_prior() for estimator in self.estimators),
            name=self.name,
            keys=self.keys,
        )

    def set_prior(self, prior: Model) -> None:
        """Propagate an ordered composite prior to child estimators."""
        if not isinstance(prior, CompositeDistribution):
            raise TypeError("CompositeEstimator requires a composite prior.")
        for estimator, component_prior in zip(self.estimators, prior.dists):
            estimator.set_prior(component_prior)

    def accumulator_factory(self) -> CompositeAccumulatorFactory:
        """Create a concrete factory preserving the whole-product key."""
        return CompositeAccumulatorFactory(
            tuple(estimator.accumulator_factory() for estimator in self.estimators),
            self.keys,
        )

    def model_log_density(self, model: CompositeDistribution) -> float:
        """Score ordered model parameters under the ordered product prior."""
        return self.get_prior().log_density(model.get_parameters())

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> CompositeDistribution:
        """Estimate each component from its corresponding sufficient statistic."""
        suff_stat = cast(CompositeSuffStat, args[-1])
        return CompositeDistribution(
            tuple(
                estimator.estimate(component_stat)
                for estimator, component_stat in zip(self.estimators, suff_stat)
            ),
            name=self.name,
            keys=self.keys,
        )


class CompositeDataEncoder(DataSequenceEncoder[Observation, CompositeEncoded]):
    """Encode each tuple position with its ordered child encoder."""

    def __init__(self, encoders: Sequence[Encoder]) -> None:
        """Store child encoders in component order."""
        self.encoders = tuple(encoders)

    def __eq__(self, other: object) -> bool:
        """Return whether child encoder sequences are equal and equally long."""
        return (
            isinstance(other, CompositeDataEncoder) and self.encoders == other.encoders
        )

    def __str__(self) -> str:
        """Return a stable constructor-like encoder representation."""
        return f"CompositeDataEncoder([{','.join(map(str, self.encoders))}])"

    def seq_encode(self, x: Iterable[Observation]) -> "CompositeEncodedData":
        """Encode each observation position and unwrap child containers."""
        observations = tuple(x)
        encoded = tuple(
            encoder.seq_encode(observation[index] for observation in observations).data
            for index, encoder in enumerate(self.encoders)
        )
        return CompositeEncodedData(data=encoded)


class CompositeEncodedData(EncodedDataSequence[CompositeEncoded]):
    """Contain component encodings in the same order as the distributions."""

    def __init__(self, data: CompositeEncoded) -> None:
        """Store the ordered tuple of child encoded data."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"CompositeEncodedDataSequence(data={self.data!r})"
