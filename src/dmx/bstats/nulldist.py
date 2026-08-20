"""Provide the neutral distribution used by Bayesian models.

``NullDistribution`` represents the absence of a probabilistic factor or prior. It
assigns unit density to every observation, contributes no sufficient statistics,
and estimates back to the shared null distribution. Setting its prior or parameters
is intentionally a no-op so it can safely fill optional prior and model slots.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional

import numpy as np

from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)

NullSequence = Sequence[Any]
Model = ProbabilityDistribution[Any, Any, Any]


class NullDistribution(ProbabilityDistribution[Any, None, NullSequence]):
    """Neutral distribution representing no likelihood or prior contribution."""

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return "NullDistribution()"

    def get_prior(self) -> "NullDistribution":
        """Return this distribution as its own neutral prior."""
        return self

    def set_prior(self, prior: Model) -> None:
        """Ignore a prior because a null distribution has no parameters."""
        del prior

    def get_parameters(self) -> None:
        """Return the sole null parameter value."""
        return None

    def set_parameters(self, value: None) -> None:
        """Accept the sole null parameter value without changing state."""
        del value

    def moments(self, p: Any, o: Any) -> float:
        """Return the neutral moment used by legacy Bayesian callers."""
        del p, o
        return 1.0

    def cross_entropy(self, dist: Model) -> float:
        """Return zero because the null factor has no information content."""
        del dist
        return 0.0

    def entropy(self) -> float:
        """Return zero because the null factor has no information content."""
        return 0.0

    def density(self, x: Any) -> float:
        """Return unit density for every observation."""
        del x
        return 1.0

    def log_density(self, x: Any) -> float:
        """Return zero log-density for every observation."""
        del x
        return 0.0

    def seq_log_density(
        self, x: NullSequence | "NullEncodedData"
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return one zero log-density per encoded observation."""
        observations = x.data if isinstance(x, NullEncodedData) else x
        return np.zeros(len(observations), dtype=np.float64)

    def seq_encode(self, x: Iterable[Any]) -> NullSequence:
        """Materialize observations for length-preserving neutral scoring."""
        return tuple(x)

    def expected_log_density(self, x: Any) -> float:
        """Return zero expected log-density for every observation."""
        del x
        return 0.0

    def seq_expected_log_density(
        self, x: NullSequence | "NullEncodedData"
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return one zero expected log-density per encoded observation."""
        return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None) -> "NullSampler":
        """Create a sampler that emits ``None`` placeholders."""
        return NullSampler(self, seed)

    def estimator(self) -> "NullEstimator":
        """Create an estimator that always returns the shared null instance."""
        return NullEstimator()

    def dist_to_encoder(self) -> "NullDataEncoder":
        """Create the length-preserving null encoder."""
        return NullDataEncoder()


null_dist = NullDistribution()


class NullSampler(DistributionSampler[None]):
    """Sampler returning ``None`` because a null factor has no observation model."""

    def __init__(
        self, dist: Optional[NullDistribution] = None, seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler; the seed is retained only for API consistency."""
        super().__init__(dist if dist is not None else null_dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Return one or more ``None`` placeholders."""
        if size is None:
            return None
        return [None] * size


class NullAccumulator(StatisticAccumulator[Any, None, NullSequence]):
    """Accumulator that deliberately discards every observation."""

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Discard one weighted observation."""
        del x, weight, estimate

    def seq_update(
        self,
        x: NullSequence,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Discard a sequence of weighted observations."""
        del x, weights, estimate

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Discard one observation during randomized initialization."""
        del x, weight, rng

    def seq_initialize(
        self,
        x: NullSequence,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Discard encoded observations during randomized initialization."""
        del x, weights, rng

    def combine(self, suff_stat: None) -> "NullAccumulator":
        """Combine the sole null statistic value."""
        del suff_stat
        return self

    def value(self) -> None:
        """Return the sole null sufficient statistic."""
        return None

    def from_value(self, x: None) -> "NullAccumulator":
        """Restore the sole null sufficient statistic."""
        del x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def acc_to_encoder(self) -> "NullDataEncoder":
        """Create the encoder corresponding to this accumulator."""
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory[Any, None, NullSequence]):
    """Create fresh null accumulators."""

    def make(self) -> NullAccumulator:
        """Create a null accumulator."""
        return NullAccumulator()


class NullEstimator(ParameterEstimator[Any, None, NullSequence, None]):
    """Estimator that always returns the shared neutral distribution."""

    def __init__(self, prior: Optional[Model] = None, keys: Any = None) -> None:
        """Initialize compatibility arguments that have no statistical effect."""
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self) -> NullAccumulatorFactory:
        """Create a factory for no-op accumulators."""
        return NullAccumulatorFactory()

    def get_prior(self) -> NullDistribution:
        """Return the shared neutral prior."""
        return null_dist

    def set_prior(self, prior: Model) -> None:
        """Ignore prior replacement because this estimator is parameter-free."""
        del prior

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> NullDistribution:
        """Ignore either supported statistic call form and return the null model."""
        del args
        return null_dist


class NullDataEncoder(DataSequenceEncoder[Any, NullSequence]):
    """Encode observations while retaining only what neutral scoring needs."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "NullDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has null semantics."""
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Iterable[Any]) -> "NullEncodedData":
        """Materialize observations so sequence result length remains available."""
        return NullEncodedData(data=tuple(x))


class NullEncodedData(EncodedDataSequence[NullSequence]):
    """Contain observations used only to preserve null sequence length."""

    def __init__(self, data: NullSequence) -> None:
        """Store length-bearing null sequence data."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise representation of encoded null data."""
        return f"NullEncodedData(data={self.data!r})"


null_estimator = NullEstimator()
