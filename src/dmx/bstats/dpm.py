"""Truncated variational Dirichlet-process mixture models.

The truncation level is the number of supplied component estimators. Each
variational stick has beta parameters ``g[k] = (gamma_1, gamma_2)``. Expected
log stick lengths and expected log remaining lengths are combined into ordered
component log weights, then exponentiated and normalized for the finite
predictive mixture stored in ``w``. The legacy ``v`` attribute aliases these
normalized predictive weights; it does not contain beta-stick means.

Component distributions carry posterior priors, while ``component_priors``
records the corresponding priors from before the update. Components and their
priors are sorted together by decreasing effective count after every update.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional, cast

import numpy as np

from dmx.arithmetic import maxint
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.nulldist import null_dist
from dmx.bstats.pdist import (
    DistributionSampler,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.bstats.sequence import SequenceDistribution
from dmx.utils.special import betaln, digamma

# Bayesian distributions intentionally implement the supported legacy protocol.
# pylint: disable=abstract-method

Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
Accumulator = StatisticAccumulator[Any, Any, Any]
AccumulatorFactory = StatisticAccumulatorFactory[Any, Any, Any]
Array = np.ndarray[Any, np.dtype[np.float64]]
DPMParameters = tuple[float, Array, list[Any]]
DPMSuffStat = tuple[Array, Array, float, float, tuple[Any, ...]]
DPMKeys = tuple[Optional[str], Optional[str]]

default_prior = GammaDistribution(2.0, 1.0)


def cbg(x: float, s1: float, s2: float) -> float:
    """Evaluate the legacy transformed beta-gamma log density helper."""
    return float(
        np.log(s1)
        + s1 * np.log(s2)
        - (s1 + 1.0) * np.log(s2 - np.log1p(-x))
        - np.log1p(-x)
    )


def _normalized_weights(values: Sequence[float] | np.ndarray[Any, Any]) -> Array:
    """Return finite, nonnegative weights normalized to one."""
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("DPM weights must be a nonempty vector.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("DPM weights must be finite and nonnegative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one DPM weight must be positive.")
    return cast(Array, weights / total)


def _logsumexp(values: Array, axis: int) -> Array:
    """Reduce log values stably along one axis."""
    return cast(Array, np.logaddexp.reduce(values, axis=axis))


class DirichletProcessMixtureDistribution(
    ProbabilityDistribution[Any, DPMParameters, Any]
):
    """Finite truncation of a variational Dirichlet-process mixture.

    ``g`` stores beta variational parameters and ``a`` the concentration.
    ``w`` is the normalized finite predictive approximation used for sampling
    and ordinary log-density.

    There are ``K`` components, ``w`` has shape ``(K,)``, and ``g`` has shape
    ``(K, 2)``. The legacy ``v`` attribute is an alias for ``w`` rather than a
    vector of beta-stick means. Components must share an observation encoding.

    Args:
        components: Ordered homogeneous component distributions.
        w: Finite nonnegative predictive weights, normalized on input.
        a: Positive Dirichlet-process concentration parameter.
        g: Positive beta variational parameters shaped ``(K, 2)``.
        component_priors: Pre-update priors paired with the components.
        name: Optional identifier for the mixture.
        prior: Hyperprior for the concentration parameter.

    Raises:
        ValueError: If component counts or shapes disagree, or concentration,
            weights, or beta parameters are invalid.
    """

    # Keep the established public constructor signature.
    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        components: Sequence[Model],
        w: Sequence[float] | np.ndarray[Any, Any],
        a: float,
        g: np.ndarray[Any, Any],
        component_priors: Sequence[Model],
        name: Optional[str] = None,
        prior: Optional[Model] = default_prior,
    ) -> None:
        """Initialize components, predictive weights, and variational state."""
        super().__init__()
        self.set_parameters((a, np.asarray(w, dtype=np.float64), list(components)))
        gamma = np.asarray(g, dtype=np.float64)
        if gamma.shape != (self.max_components, 2):
            raise ValueError("DPM beta parameters must have shape (components, 2).")
        if np.any(~np.isfinite(gamma)) or np.any(gamma <= 0.0):
            raise ValueError("DPM beta parameters must be finite and positive.")
        if len(component_priors) != self.max_components:
            raise ValueError("Component prior count must match DPM components.")
        self.g = gamma
        self.component_priors = tuple(component_priors)
        self.name = name
        self.prior = cast(Model, prior)

    def __str__(self) -> str:
        """Return a concise constructor-like representation."""
        components = ",".join(map(str, self.components))
        weights = ",".join(map(str, self.w))
        return (
            f"DirichletProcessMixtureDistribution([{components}], [{weights}], "
            f"{self.a!r}, name={self.name!r}, prior={self.prior})"
        )

    def get_prior(self) -> CompositeDistribution:
        """Return concentration, stick, and ordered component priors."""
        stick_prior = SequenceDistribution(BetaDistribution(1.0, self.a))
        component_prior = CompositeDistribution(
            tuple(component.get_prior() for component in self.components)
        )
        return CompositeDistribution((self.prior, stick_prior, component_prior))

    def set_prior(self, prior: Model) -> None:
        """Propagate a complete concentration, stick, and component prior."""
        if not isinstance(prior, CompositeDistribution) or len(prior.dists) != 3:
            raise TypeError(
                "A DPM prior must contain concentration, sticks, and components."
            )
        component_prior = prior.dists[2]
        if not isinstance(component_prior, CompositeDistribution):
            raise TypeError("DPM component priors must be a composite distribution.")
        if len(component_prior.dists) != self.max_components:
            raise ValueError("Component prior count must match DPM components.")
        self.prior = prior.dists[0]
        self.component_priors = tuple(component_prior.dists)
        for component, value in zip(self.components, component_prior.dists):
            component.set_prior(value)

    def get_parameters(self) -> DPMParameters:
        """Return concentration, legacy normalized ``v`` weights, and parameters."""
        return self.a, self.v.copy(), [x.get_parameters() for x in self.components]

    def set_parameters(self, value: DPMParameters) -> None:
        """Replace concentration, predictive weights, and component objects."""
        concentration, weights, components = value
        if not np.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("DPM concentration must be finite and positive.")
        self.components = tuple(cast(Sequence[Model], components))
        self.max_components = len(self.components)
        self.num_components = self.max_components
        if not self.components:
            raise ValueError("A DPM requires at least one component.")
        if len(weights) != self.max_components:
            raise ValueError("DPM weights and components must have equal length.")
        self.w = _normalized_weights(weights)
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)
        self.a = float(concentration)
        self.expected_log_nw = float(self.log_w[-1])
        self.v = self.w

    def log_density(self, x: Any) -> float:
        """Evaluate the normalized finite predictive mixture density."""
        values = np.asarray(
            [component.log_density(x) for component in self.components], dtype=float
        )
        return float(_logsumexp(values + self.log_w, axis=0))

    def expected_log_density(self, x: Any) -> float:
        """Combine component posterior expectations with predictive log weights."""
        values = np.asarray(
            [component.expected_log_density(x) for component in self.components],
            dtype=float,
        )
        return float(_logsumexp(values + self.log_w, axis=0))

    def seq_log_density(self, x: Any) -> Array:
        """Return the variational objective for one encoded data block.

        The one-element result is the block evidence lower bound used by the
        legacy optimizer for convergence, not one predictive score per row.
        """
        expected = np.asarray(
            [component.seq_expected_log_density(x) for component in self.components],
            dtype=np.float64,
        ).T
        weighted = expected + self.log_w
        maximum = weighted.max(axis=1, keepdims=True)
        responsibilities = np.exp(weighted - maximum)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        remaining = 1.0 - np.cumsum(responsibilities, axis=1)

        gamma_sum = self.g.sum(axis=1)
        prior_cross_entropy = np.sum(
            -betaln(1.0, self.a)
            + (digamma(self.g[:, 1]) - digamma(gamma_sum)) * (self.a - 1.0)
        )
        component_cross_entropy = sum(
            -component.get_prior().cross_entropy(component_prior)
            for component, component_prior in zip(
                self.components, self.component_priors
            )
        )
        expected_log_stick = digamma(self.g[:, 0]) - digamma(gamma_sum)
        expected_log_remainder = digamma(self.g[:, 1]) - digamma(gamma_sum)
        assignment_term = float(
            (remaining * expected_log_remainder).sum()
            + (responsibilities * expected_log_stick).sum()
            + np.sum(responsibilities * expected)
        )
        beta_entropy = -(
            betaln(self.g[:, 0], self.g[:, 1]).sum()
            - ((self.g - 1.0) * digamma(self.g)).sum()
            + ((gamma_sum - 2.0) * digamma(gamma_sum)).sum()
        )
        component_entropy = sum(
            -component.get_prior().entropy() for component in self.components
        )
        positive = responsibilities > 0.0
        assignment_entropy = float(
            np.sum(np.log(responsibilities[positive]) * responsibilities[positive])
        )
        objective = (
            float(prior_cross_entropy)
            + float(component_cross_entropy)
            + assignment_term
            - float(beta_entropy + component_entropy + assignment_entropy)
        )
        return np.asarray([objective], dtype=np.float64)

    def seq_expected_log_density(self, x: Any) -> Array:
        """Return predictive expected scores for encoded observations."""
        values = np.asarray(
            [component.seq_expected_log_density(x) for component in self.components],
            dtype=np.float64,
        ).T
        return np.asarray(_logsumexp(values + self.log_w, axis=1), dtype=np.float64)

    def seq_encode(self, x: Iterable[Any]) -> Any:
        """Encode homogeneous observations with the first component."""
        return self.components[0].seq_encode(x)

    def sampler(self, seed: Optional[int] = None) -> "DirichletProcessMixtureSampler":
        """Create a repeatable sampler from normalized predictive weights."""
        return DirichletProcessMixtureSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "DirichletProcessMixtureEstimator":
        """Create a compatible estimator for the current truncation level."""
        if pseudo_count is None:
            estimators = tuple(component.estimator() for component in self.components)
        else:
            component_count = float(self.num_components)
            estimators = tuple(
                cast(Any, component.estimator)(
                    pseudo_count=pseudo_count / component_count
                )
                for component in self.components
            )
        return DirichletProcessMixtureEstimator(
            estimators, name=self.name, prior=self.prior
        )


class DirichletProcessMixtureSampler(DistributionSampler[Any]):
    """Sample component indices and observations from a truncated DPM."""

    def __init__(
        self, dist: DirichletProcessMixtureDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize independent selection and component random states."""
        seed_source = np.random.RandomState(seed)
        super().__init__(dist, int(seed_source.randint(maxint)))
        self.dist = dist
        self.comp_samplers = tuple(
            component.sampler(seed=int(seed_source.randint(maxint)))
            for component in dist.components
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


class DirichletProcessMixtureAccumulator(
    SequenceEncodableAccumulator[Any, DPMSuffStat, Any]
):
    """Accumulate variational assignments and component statistics.

    The sufficient statistic is ``(component_counts, beta_counts, alpha,
    previous_log_remainder, child_statistics)``. For truncation level ``K``,
    the first two arrays have shapes ``(K,)`` and ``(K, 2)`` and the last item
    is a length-``K`` tuple in component order. Updates use expected component
    scores and current predictive log weights to form responsibilities.

    The optional keys independently share beta counts and the complete child
    accumulator collection. Child-level sharing remains active.
    """

    def __init__(
        self, accumulators: Sequence[Accumulator], keys: DPMKeys = (None, None)
    ) -> None:
        """Initialize empty statistics for a fixed truncation level."""
        self.accumulators = list(accumulators)
        self.num_components = len(self.accumulators)
        if not self.accumulators:
            raise ValueError("A DPM accumulator requires at least one component.")
        self.comp_counts = np.zeros(self.num_components, dtype=np.float64)
        self.beta_counts = np.zeros((self.num_components, 2), dtype=np.float64)
        self.prev_nw = float(np.log(0.5) * (self.num_components - 1))
        self.a = 1.0
        self.weight_key, self.comp_key = keys

    @staticmethod
    def _estimate(value: Optional[Model]) -> DirichletProcessMixtureDistribution:
        """Require the current DPM estimate used for responsibilities."""
        if not isinstance(value, DirichletProcessMixtureDistribution):
            raise TypeError("DPM accumulation requires a DPM estimate.")
        return value

    @staticmethod
    def _normalize(log_values: Array) -> Array:
        """Convert component log scores into stable responsibilities."""
        shifted = log_values - np.max(log_values, axis=-1, keepdims=True)
        values = np.exp(shifted)
        return cast(Array, values / values.sum(axis=-1, keepdims=True))

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Allocate one observation by variational responsibility."""
        mixture = self._estimate(estimate)
        log_values = (
            np.asarray(
                [component.expected_log_density(x) for component in mixture.components],
                dtype=np.float64,
            )
            + mixture.log_w
        )
        responsibilities = self._normalize(log_values)
        allocations = responsibilities * weight
        self.comp_counts += allocations
        self.beta_counts[:, 0] += allocations
        self.beta_counts[:, 1] += (1.0 - np.cumsum(responsibilities)) * weight
        for index, accumulator in enumerate(self.accumulators):
            accumulator.update(x, float(allocations[index]), mixture.components[index])

    def seq_update(self, x: Any, weights: Array, estimate: Optional[Model]) -> None:
        """Allocate encoded observations by variational responsibility."""
        mixture = self._estimate(estimate)
        log_values = (
            np.asarray(
                [
                    component.seq_expected_log_density(x)
                    for component in mixture.components
                ],
                dtype=np.float64,
            ).T
            + mixture.log_w
        )
        responsibilities = self._normalize(log_values)
        allocations = responsibilities * weights[:, None]
        component_counts = allocations.sum(axis=0)
        remaining_counts = (
            (1.0 - np.cumsum(responsibilities, axis=1)) * weights[:, None]
        ).sum(axis=0)
        self.comp_counts += component_counts
        self.beta_counts[:, 0] += component_counts
        self.beta_counts[:, 1] += remaining_counts
        for index, accumulator in enumerate(self.accumulators):
            sequence_accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any], accumulator
            )
            sequence_accumulator.seq_update(
                x, allocations[:, index], mixture.components[index]
            )

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Randomly split one weight with the established uniform Dirichlet."""
        responsibilities = (
            np.zeros(self.num_components, dtype=np.float64)
            if weight == 0.0
            else rng.dirichlet(np.ones(self.num_components))
        )
        allocations = responsibilities * weight
        self.comp_counts += allocations
        self.beta_counts[:, 0] += allocations
        self.beta_counts[:, 1] += (1.0 - np.cumsum(responsibilities)) * weight
        for index, accumulator in enumerate(self.accumulators):
            accumulator.initialize(x, float(allocations[index]), rng)

    def combine(self, suff_stat: DPMSuffStat) -> "DirichletProcessMixtureAccumulator":
        """Merge variational and ordered component sufficient statistics."""
        self.comp_counts += suff_stat[0]
        self.beta_counts += suff_stat[1]
        self.a = suff_stat[2]
        self.prev_nw = suff_stat[3]
        for accumulator, value in zip(self.accumulators, suff_stat[4]):
            accumulator.combine(value)
        return self

    def value(self) -> DPMSuffStat:
        """Return copied variational and component sufficient statistics."""
        return (
            self.comp_counts.copy(),
            self.beta_counts.copy(),
            self.a,
            self.prev_nw,
            tuple(accumulator.value() for accumulator in self.accumulators),
        )

    def from_value(self, x: DPMSuffStat) -> "DirichletProcessMixtureAccumulator":
        """Restore variational and component sufficient statistics."""
        self.comp_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.beta_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.a = float(x[2])
        self.prev_nw = float(x[3])
        for accumulator, value in zip(self.accumulators, x[4]):
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
        """Merge DPM blocks and supported child statistics by key."""
        if self.weight_key is not None:
            existing = stats_dict.get(self.weight_key)
            stats_dict[self.weight_key] = (
                existing + self.beta_counts
                if isinstance(existing, np.ndarray)
                else self.beta_counts.copy()
            )
        if self.comp_key is not None:
            existing = stats_dict.get(self.comp_key)
            if isinstance(existing, DirichletProcessMixtureAccumulator):
                for accumulator, value in zip(existing.accumulators, self.value()[4]):
                    accumulator.combine(value)
            else:
                stats_dict[self.comp_key] = copy.deepcopy(self)
        for accumulator in self.accumulators:
            if self._supports_key_merge(accumulator):
                try:
                    accumulator.key_merge(stats_dict)
                except NotImplementedError:
                    # Nested legacy accumulators may expose a container hook
                    # while one of their children has no keyed-stat protocol.
                    pass

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace DPM blocks and supported child statistics by key."""
        if self.weight_key is not None:
            counts = stats_dict.get(self.weight_key)
            if isinstance(counts, np.ndarray):
                self.beta_counts = counts.copy()
        if self.comp_key is not None:
            existing = stats_dict.get(self.comp_key)
            if isinstance(existing, DirichletProcessMixtureAccumulator):
                for accumulator, source in zip(
                    self.accumulators, existing.accumulators
                ):
                    accumulator.from_value(source.value())
        for accumulator in self.accumulators:
            if self._supports_key_replace(accumulator):
                try:
                    accumulator.key_replace(stats_dict)
                except NotImplementedError:
                    # Key sharing is optional throughout the legacy protocol.
                    pass

    def acc_to_encoder(self) -> Any:
        """Return the homogeneous encoder exposed by the first component."""
        return self.accumulators[0].acc_to_encoder()


class DirichletProcessMixtureAccumulatorFactory(
    StatisticAccumulatorFactory[Any, DPMSuffStat, Any]
):
    """Create DPM accumulators for a fixed truncation level."""

    def __init__(
        self, factories: Sequence[AccumulatorFactory], dim: int, keys: DPMKeys
    ) -> None:
        """Store ordered child factories, dimension, and sharing keys."""
        self.factories = tuple(factories)
        self.dim = dim
        self.keys = keys

    def make(self) -> DirichletProcessMixtureAccumulator:
        """Create one fresh accumulator per truncated component."""
        return DirichletProcessMixtureAccumulator(
            tuple(self.factories[index].make() for index in range(self.dim)),
            self.keys,
        )


class DirichletProcessMixtureEstimator(
    ParameterEstimator[Any, DPMParameters, Any, DPMSuffStat]
):
    """Perform the established coordinate update for a truncated DPM.

    The number of component estimators fixes the truncation level. Estimation
    updates beta-stick parameters and the concentration, estimates every
    component, then stably sorts components and their pre-update priors by
    decreasing effective count. The resulting normalized weights are the
    finite predictive approximation, not beta-stick means.

    Args:
        estimators: Component estimators defining the truncation level.
        name: Optional identifier copied to the estimated mixture.
        prior: Hyperprior for the concentration parameter.
        keys: Pair ``(weight_key, component_key)`` controlling independent
            sufficient-statistic sharing.
    """

    def __init__(
        self,
        estimators: Sequence[Estimator],
        name: Optional[str] = None,
        prior: Optional[Model] = default_prior,
        keys: DPMKeys = (None, None),
    ) -> None:
        """Initialize ordered component estimators and concentration prior."""
        self.estimators = tuple(estimators)
        self.num_components = len(self.estimators)
        if not self.estimators:
            raise ValueError("A DPM estimator requires at least one component.")
        self.name = name
        self.keys = keys
        self.prior = cast(Model, prior)

    def accumulator_factory(self) -> DirichletProcessMixtureAccumulatorFactory:
        """Create a factory retaining estimator order and sharing keys."""
        factories = tuple(
            estimator.accumulator_factory() for estimator in self.estimators
        )
        return DirichletProcessMixtureAccumulatorFactory(
            factories, self.num_components, self.keys
        )

    def get_prior(self) -> CompositeDistribution:
        """Return concentration, stick, and ordered component priors."""
        concentration = (
            (self.prior.k - 1.0) * self.prior.theta
            if isinstance(self.prior, GammaDistribution)
            else 1.0
        )
        stick_prior = SequenceDistribution(
            BetaDistribution(1.0, concentration), null_dist
        )
        component_prior = CompositeDistribution(
            tuple(estimator.get_prior() for estimator in self.estimators)
        )
        return CompositeDistribution((self.prior, stick_prior, component_prior))

    def set_prior(self, prior: Model) -> None:
        """Propagate a complete concentration, stick, and component prior."""
        if not isinstance(prior, CompositeDistribution) or len(prior.dists) != 3:
            raise TypeError(
                "A DPM prior must contain concentration, sticks, and components."
            )
        component_prior = prior.dists[2]
        if not isinstance(component_prior, CompositeDistribution):
            raise TypeError("DPM component priors must be a composite distribution.")
        if len(component_prior.dists) != self.num_components:
            raise ValueError("Component prior count must match DPM estimators.")
        self.prior = prior.dists[0]
        for estimator, value in zip(self.estimators, component_prior.dists):
            estimator.set_prior(value)

    def model_log_density(self, model: DirichletProcessMixtureDistribution) -> float:
        """Return prior and entropy terms for optimizer convergence output."""
        gamma_sum = model.g.sum(axis=1)
        stick_cross_entropy = np.sum(
            -betaln(1.0, model.a)
            + (digamma(model.g[:, 1]) - digamma(gamma_sum)) * (model.a - 1.0)
        )
        component_cross_entropy = sum(
            -component.get_prior().cross_entropy(component_prior)
            for component, component_prior in zip(
                model.components, model.component_priors
            )
        )
        beta_entropy = -(
            betaln(model.g[:, 0], model.g[:, 1]).sum()
            - ((model.g - 1.0) * digamma(model.g)).sum()
            + ((gamma_sum - 2.0) * digamma(gamma_sum)).sum()
        )
        component_entropy = sum(
            -component.get_prior().entropy() for component in model.components
        )
        return float(
            stick_cross_entropy
            + component_cross_entropy
            - beta_entropy
            - component_entropy
        )

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> DirichletProcessMixtureDistribution:
        """Apply one variational update and sort components by effective count."""
        comp_counts, raw_beta_counts, alpha, _prev_nw, component_stats = cast(
            DPMSuffStat, args[-1]
        )
        component_priors = tuple(estimator.get_prior() for estimator in self.estimators)
        components = tuple(
            estimator.estimate(value)
            for estimator, value in zip(self.estimators, component_stats)
        )

        order = np.argsort(-comp_counts, kind="stable")
        sorted_counts = np.asarray(comp_counts[order], dtype=np.float64)
        beta_counts = np.asarray(raw_beta_counts[order], dtype=np.float64).copy()
        components = tuple(components[index] for index in order)
        component_priors = tuple(component_priors[index] for index in order)
        beta_counts[:, 0] = sorted_counts
        beta_counts[:, 1] = sorted_counts.sum() - np.cumsum(sorted_counts)

        digamma_sum = digamma(beta_counts.sum(axis=1) + 1.0 + alpha)
        expected_log_stick = digamma(beta_counts[:, 0] + 1.0) - digamma_sum
        expected_log_remainder = digamma(beta_counts[:, 1] + alpha) - digamma_sum
        expected_log_weights = np.asarray(expected_log_stick, dtype=np.float64)
        if self.num_components > 1:
            expected_log_weights[1:] += np.cumsum(expected_log_remainder)[:-1]
        weights = _normalized_weights(
            np.exp(expected_log_weights - np.max(expected_log_weights))
        )

        if isinstance(self.prior, GammaDistribution):
            prior_shape = self.prior.k
            prior_rate = 1.0 / self.prior.theta
            posterior_shape = prior_shape + self.num_components
            posterior_rate = prior_rate - float(np.cumsum(expected_log_remainder)[-1])
            hyper_posterior: Optional[Model] = GammaDistribution(
                posterior_shape, 1.0 / posterior_rate
            )
        else:
            prior_shape = 0.0
            prior_rate = 0.0
            hyper_posterior = None

        remaining_index = -2 if self.num_components > 1 else -1
        concentration_shape = prior_shape + self.num_components - 1.0
        concentration_rate = prior_rate - float(
            np.cumsum(expected_log_remainder)[remaining_index]
        )
        new_alpha = concentration_shape / concentration_rate
        if not np.isfinite(new_alpha) or new_alpha <= 0.0:
            new_alpha = float(alpha)

        gammas = beta_counts.copy()
        gammas[:, 0] += 1.0
        gammas[:, 1] += new_alpha
        return DirichletProcessMixtureDistribution(
            components,
            weights,
            new_alpha,
            gammas,
            component_priors,
            name=self.name,
            prior=hyper_posterior,
        )
