"""Fixed point-mass likelihoods.

``DiracDistribution(value)`` has singleton support at ``value``: matching
observations have log-density zero and all others have log-density ``-inf``.
It has no learned parameter or nontrivial sufficient statistic. Sampling and
estimation always reproduce the constructor value, and expected log-density
is identical to fixed log-density.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np

from dmx.bstats.nulldist import null_dist
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)

# A fixed point mass intentionally does not implement information measures.
# pylint: disable=abstract-method,too-few-public-methods

DiracEncoded = np.ndarray[Any, np.dtype[np.object_]]
Model = ProbabilityDistribution[Any, Any, Any]


def _matches(left: Any, right: Any) -> bool:
    """Return a scalar equality result, including for array-valued points."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


class DiracDistribution(ProbabilityDistribution[Any, Any, DiracEncoded]):
    """Point mass supported only at one fixed value."""

    def __init__(self, value: Any) -> None:
        """Initialize the fixed support point."""
        super().__init__()
        self.value = value
        self.prior = null_dist

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return f"DiracDistribution({self.value!r})"

    def get_prior(self) -> Model:
        """Return the neutral prior associated with a fixed point mass."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Associate prior metadata without changing the fixed value."""
        self.prior = prior

    def get_parameters(self) -> Any:
        """Return the fixed support point."""
        return self.value

    def set_parameters(self, value: Any) -> None:
        """Replace the fixed support point."""
        self.value = value

    def density(self, x: Any) -> float:
        """Return one at the support point and zero elsewhere."""
        return 1.0 if _matches(x, self.value) else 0.0

    def log_density(self, x: Any) -> float:
        """Return zero at the support point and ``-inf`` elsewhere."""
        return 0.0 if _matches(x, self.value) else float(-np.inf)

    def expected_log_density(self, x: Any) -> float:
        """Return fixed log-density because the point is not random."""
        return self.log_density(x)

    def seq_log_density(self, x: DiracEncoded) -> np.ndarray[Any, Any]:
        """Evaluate point-mass log-densities for encoded observations."""
        return np.asarray([self.log_density(value) for value in x], dtype=float)

    def seq_expected_log_density(self, x: DiracEncoded) -> np.ndarray[Any, Any]:
        """Return vectorized fixed log-densities."""
        return self.seq_log_density(x)

    def seq_encode(self, x: Iterable[Any]) -> DiracEncoded:
        """Encode observations as an object array."""
        return np.asarray(tuple(x), dtype=object)

    def sampler(self, seed: Optional[int] = None) -> "DiracSampler":
        """Create a deterministic sampler for the fixed point."""
        return DiracSampler(self, seed)

    def estimator(self) -> "DiracEstimator":
        """Create an estimator that retains the fixed point."""
        return DiracEstimator(self.value, prior=self.prior)

    def dist_to_encoder(self) -> "DiracDataEncoder":
        """Create the point-mass sequence encoder."""
        return DiracDataEncoder()


class DiracSampler(DistributionSampler[Any]):
    """Return copies of one fixed point."""

    def __init__(self, dist: DiracDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler; the seed is retained for API consistency."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Return one copied point or a list of ``size`` copied points."""
        if size is None:
            return copy.deepcopy(self.dist.value)
        return [copy.deepcopy(self.dist.value) for _ in range(size)]


class DiracAccumulator(StatisticAccumulator[Any, None, DiracEncoded]):
    """Discard observations because a point mass has no learned statistics."""

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Discard one weighted observation."""
        del x, weight, estimate

    def seq_update(
        self,
        x: DiracEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Discard encoded weighted observations."""
        del x, weights, estimate

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Discard one observation during initialization."""
        del x, weight, rng

    def seq_initialize(
        self,
        x: DiracEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Discard encoded observations during initialization."""
        del x, weights, rng

    def combine(self, suff_stat: None) -> "DiracAccumulator":
        """Combine the sole empty statistic."""
        del suff_stat
        return self

    def value(self) -> None:
        """Return the sole empty sufficient statistic."""
        return None

    def from_value(self, x: None) -> "DiracAccumulator":
        """Restore the sole empty sufficient statistic."""
        del x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def acc_to_encoder(self) -> "DiracDataEncoder":
        """Create the compatible point-mass encoder."""
        return DiracDataEncoder()


class DiracAccumulatorFactory(StatisticAccumulatorFactory[Any, None, DiracEncoded]):
    """Create empty point-mass accumulators."""

    def make(self) -> DiracAccumulator:
        """Create a point-mass accumulator."""
        return DiracAccumulator()


class DiracEstimator(ParameterEstimator[Any, Any, DiracEncoded, None]):
    """Return a fixed point-mass distribution for every statistic value."""

    def __init__(
        self, value: Any, prior: Model = null_dist, keys: Optional[str] = None
    ) -> None:
        """Initialize the fixed point and compatibility metadata."""
        self.value = value
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self) -> DiracAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return DiracAccumulatorFactory()

    def get_prior(self) -> Model:
        """Return prior metadata for the fixed point."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace prior metadata without changing the fixed point."""
        self.prior = prior

    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> DiracDistribution:
        """Return the fixed point mass, ignoring empty statistics."""
        del args
        distribution = DiracDistribution(self.value)
        distribution.set_prior(self.prior)
        return distribution


class DiracDataEncoder(DataSequenceEncoder[Any, DiracEncoded]):
    """Encode arbitrary point-mass observations as object arrays."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "DiracDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has point-mass semantics."""
        return isinstance(other, DiracDataEncoder)

    def seq_encode(self, x: Iterable[Any]) -> "DiracEncodedData":
        """Encode observations in a typed container."""
        return DiracEncodedData(np.asarray(tuple(x), dtype=object))


class DiracEncodedData(EncodedDataSequence[DiracEncoded]):
    """Contain an encoded point-mass sequence."""

    def __init__(self, data: DiracEncoded) -> None:
        """Store the object array."""
        super().__init__(data)
