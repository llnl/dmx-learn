"""Bayesian categorical likelihoods on arbitrary hashable observations.

The probability mapping defines the sampled support. Observations missing from
that mapping receive ``default_value / (1 + default_value)`` mass, matching the
legacy escape-mass convention. The default dimension-free dictionary Dirichlet
prior uses concentration ``1 + 1e-12`` per observed category. Accumulators
store a weighted category-count mapping and its total. Expected log-density
uses Dirichlet expectations on known categories and the fixed escape mass for
unknown categories.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.pdist import (
    DataFrameEncodableAccumulator,
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma

# Legacy bstats implementations are intentionally concrete protocol classes.
# pylint: disable=abstract-method

Key = Any
CategoricalParameters = dict[Key, float]
CategoricalEncoded = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
CategoricalSuffStat = tuple[dict[Key, float], float]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = DictDirichletDistribution(1.0 + 1.0e-12)


class CategoricalDistribution(
    ProbabilityDistribution[Key, CategoricalParameters, CategoricalEncoded]
):
    """Categorical likelihood over arbitrary hashable category labels."""

    def __init__(
        self,
        prob_map: dict[Any, float],
        default_value: float = 0.0,
        name: Optional[str] = None,
        prior: Optional[Model] = default_prior,
    ) -> None:
        """Initialize category probabilities, escape mass, and a prior."""
        super().__init__()
        self.name = name
        self.default_value = float(default_value)
        if not np.isfinite(self.default_value) or self.default_value < 0.0:
            raise ValueError(
                "Categorical default value must be finite and nonnegative."
            )
        with np.errstate(divide="ignore"):
            self.log_default_value = float(np.log(self.default_value))
        self.log1p_default_value = float(np.log1p(self.default_value))
        self.set_parameters(prob_map)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"CategoricalDistribution({self.prob_map!r}, "
            f"default_value={self.default_value!r}, name={self.name!r}, "
            f"prior={self.prior})"
        )

    def get_parameters(self) -> CategoricalParameters:
        """Return the category probability mapping."""
        return self.prob_map

    def set_parameters(self, value: dict[Any, float]) -> None:
        """Replace the category probability mapping."""
        values = np.asarray(list(value.values()), dtype=float)
        if not value or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Categorical probabilities must be finite and nonnegative."
            )
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("Categorical probabilities must sum to one.")
        self.prob_map = {key: float(item) for key, item in value.items()}

    def get_prior(self) -> Model:
        """Return the current dictionary Dirichlet or alternate prior."""
        return self.prior

    def set_prior(self, prior: Optional[Model]) -> None:
        """Replace the prior and cache expected log probabilities."""
        self.prior = cast(Model, prior)
        self.expected_nparams: Optional[dict[Key, float]] = None
        if not isinstance(prior, DictDirichletDistribution):
            return
        parameters = prior.get_parameters()
        if isinstance(parameters, float):
            normalizer = float(digamma(len(self.prob_map) * parameters))
            self.expected_nparams = {
                key: float(digamma(parameters) - normalizer) for key in self.prob_map
            }
        else:
            if set(parameters) != set(self.prob_map):
                raise ValueError("Categorical prior keys must match probability keys.")
            normalizer = float(digamma(sum(parameters.values())))
            self.expected_nparams = {
                key: float(digamma(value) - normalizer)
                for key, value in parameters.items()
            }

    def entropy(self) -> float:
        """Return Shannon entropy over the explicit sampled support."""
        probabilities = np.asarray(list(self.prob_map.values()), dtype=float)
        positive = probabilities > 0.0
        return float(-np.dot(probabilities[positive], np.log(probabilities[positive])))

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` over explicit categories."""
        return float(
            -sum(
                probability * dist.log_density(key)
                for key, probability in self.prob_map.items()
            )
        )

    def log_density(self, x: Key) -> float:
        """Evaluate known-category or escape log mass."""
        with np.errstate(divide="ignore"):
            value = np.log(self.prob_map.get(x, self.default_value))
        return float(value - self.log1p_default_value)

    def expected_log_density(self, x: Key) -> float:
        """Evaluate the prior-averaged log mass when conjugate."""
        if self.expected_nparams is None:
            return self.log_density(x)
        if x not in self.prob_map:
            return self.log_default_value - self.log1p_default_value
        return self.expected_nparams[x] - self.log1p_default_value

    def seq_log_density(self, x: CategoricalEncoded) -> np.ndarray[Any, Any]:
        """Evaluate log masses for encoded category indices."""
        indices, levels = x
        mapped = np.asarray(
            [self.prob_map.get(level, self.default_value) for level in levels],
            dtype=float,
        )
        with np.errstate(divide="ignore"):
            mapped = np.log(mapped) - self.log1p_default_value
        return np.asarray(mapped[indices], dtype=float)

    def seq_expected_log_density(self, x: CategoricalEncoded) -> np.ndarray[Any, Any]:
        """Evaluate expected log masses for encoded category indices."""
        indices, levels = x
        mapped = np.asarray(
            [self.expected_log_density(level) for level in levels], dtype=float
        )
        return mapped[indices]

    def seq_encode(self, x: Iterable[Key]) -> CategoricalEncoded:
        """Encode values as unique levels and inverse indices."""
        levels, indices = np.unique(tuple(x), return_inverse=True)
        return indices, levels

    def sampler(self, seed: Optional[int] = None) -> "CategoricalSampler":
        """Create a repeatable categorical sampler."""
        return CategoricalSampler(self, seed)

    def estimator(self) -> "CategoricalEstimator":
        """Create an estimator retaining the current prior."""
        return CategoricalEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "CategoricalDataEncoder":
        """Create the categorical sequence encoder."""
        return CategoricalDataEncoder()


class CategoricalSampler(DistributionSampler[Key]):
    """Draw independent values from an explicit categorical support."""

    def __init__(
        self, dist: CategoricalDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize levels and probabilities from ``dist``."""
        super().__init__(dist, seed)
        self.levels = list(dist.prob_map)
        self.probs = list(dist.prob_map.values())

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one category or a list of ``size`` categories."""
        if size is None:
            index = int(self.rng.choice(len(self.levels), p=self.probs))
            return self.levels[index]
        indices = self.rng.choice(len(self.levels), p=self.probs, size=size)
        return [self.levels[index] for index in indices]


class CategoricalEstimatorAccumulator(
    SequenceEncodableAccumulator[Key, CategoricalSuffStat, CategoricalEncoded],
    DataFrameEncodableAccumulator[Key, CategoricalSuffStat, CategoricalEncoded],
):
    """Accumulate weighted category counts and their total."""

    def __init__(self, name: Optional[str], keys: tuple[Optional[str], ...]) -> None:
        """Initialize empty counts and optional sharing metadata."""
        self.name = name
        self.key = keys[0]
        self.count_map: defaultdict[Key, float] = defaultdict(float)
        self.count_sum = 0.0

    def update(self, x: Key, weight: float, estimate: Optional[Model]) -> None:
        """Add one weighted category observation."""
        del estimate
        self.count_map[x] += weight
        self.count_sum += weight

    def initialize(self, x: Key, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: CategoricalEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: CategoricalEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Add encoded category observations with weights."""
        del estimate
        indices, levels = x
        counts = np.bincount(indices, weights=weights, minlength=len(levels))
        self.count_sum += float(counts.sum())
        for level, count in zip(levels, counts):
            self.count_map[level] += float(count)

    def combine(
        self, suff_stat: CategoricalSuffStat
    ) -> "CategoricalEstimatorAccumulator":
        """Merge a category-count mapping and total."""
        for key, value in suff_stat[0].items():
            self.count_map[key] += value
        self.count_sum += suff_stat[1]
        return self

    def value(self) -> CategoricalSuffStat:
        """Return a copied count mapping and total count."""
        return dict(self.count_map), self.count_sum

    def from_value(self, x: CategoricalSuffStat) -> "CategoricalEstimatorAccumulator":
        """Restore a category-count mapping and total count."""
        self.count_map = defaultdict(float, x[0])
        self.count_sum = x[1]
        return self

    def acc_to_encoder(self) -> "CategoricalDataEncoder":
        """Create the compatible categorical encoder."""
        return CategoricalDataEncoder()


class CategoricalEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[Key, CategoricalSuffStat, CategoricalEncoded]
):
    """Create categorical accumulators with shared metadata."""

    def __init__(self, name: Optional[str], keys: tuple[Optional[str], ...]) -> None:
        """Store metadata copied into each accumulator."""
        self.name = name
        self.keys = keys

    def make(self) -> CategoricalEstimatorAccumulator:
        """Create an empty categorical accumulator."""
        return CategoricalEstimatorAccumulator(self.name, self.keys)


class CategoricalEstimator(
    ParameterEstimator[
        Key, CategoricalParameters, CategoricalEncoded, CategoricalSuffStat
    ]
):
    """Estimate categorical probabilities and update a Dirichlet posterior."""

    def __init__(
        self,
        default_value: float = 0.0,
        name: Optional[str] = None,
        prior: Optional[Model] = default_prior,
        keys: tuple[Optional[str], ...] = (None,),
    ) -> None:
        """Initialize escape mass, metadata, and prior."""
        self.keys = keys
        self.name = name
        self.prior = prior
        self.default_value = default_value

    def accumulator_factory(self) -> CategoricalEstimatorAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return CategoricalEstimatorAccumulatorFactory(self.name, self.keys)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return cast(Model, self.prior)

    def set_prior(self, prior: Optional[Model]) -> None:
        """Replace the estimator prior."""
        self.prior = cast(Model, prior)

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> CategoricalDistribution:
        """Estimate probabilities from a weighted category-count mapping."""
        count_map, _ = args[-1]
        stats_sum = float(sum(count_map.values()))
        if isinstance(self.prior, DictDirichletDistribution):
            parameters = self.prior.get_parameters()
            keys = list(count_map)
            if isinstance(parameters, float):
                concentrations = {key: parameters for key in keys}
            else:
                keys = list(dict.fromkeys([*parameters, *keys]))
                concentrations = {key: parameters.get(key, 1.0) for key in keys}
            posterior = {
                key: concentrations[key] + count_map.get(key, 0.0) for key in keys
            }
            map_weights = {
                key: max(value - 1.0, 0.0) for key, value in posterior.items()
            }
            normalizer = sum(map_weights.values())
            if normalizer <= 0.0:
                probabilities = {
                    key: value / sum(posterior.values())
                    for key, value in posterior.items()
                }
            else:
                probabilities = {
                    key: value / normalizer for key, value in map_weights.items()
                }
            return CategoricalDistribution(
                probabilities,
                default_value=self.default_value,
                name=self.name,
                prior=DictDirichletDistribution(posterior),
            )
        if stats_sum <= 0.0:
            probability = 1.0 / len(count_map)
            probabilities = {key: probability for key in count_map}
        else:
            probabilities = {key: value / stats_sum for key, value in count_map.items()}
        return CategoricalDistribution(
            probabilities,
            default_value=self.default_value,
            name=self.name,
            prior=self.prior,
        )


class CategoricalDataEncoder(DataSequenceEncoder[Key, CategoricalEncoded]):
    """Encode arbitrary categorical observations."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "CategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has categorical semantics."""
        return isinstance(other, CategoricalDataEncoder)

    def seq_encode(self, x: Iterable[Key]) -> "CategoricalEncodedData":
        """Encode observations in a typed container."""
        levels, indices = np.unique(tuple(x), return_inverse=True)
        return CategoricalEncodedData(data=(indices, levels))


class CategoricalEncodedData(EncodedDataSequence[CategoricalEncoded]):
    """Contain inverse indices and unique categorical levels."""

    def __init__(self, data: CategoricalEncoded) -> None:
        """Store encoded categorical data."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"CategoricalEncodedData(data={self.data!r})"
