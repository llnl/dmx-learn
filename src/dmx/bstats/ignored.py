"""Keep a Bayesian distribution fixed while retaining its scoring behavior.

An ``IgnoredDistribution`` delegates scoring, sampling, parameters, and priors to a
wrapped distribution. Its accumulator intentionally discards observations and its
estimator returns the same wrapped distribution unchanged. Thus "ignored" refers
only to parameter estimation; observations are still scored normally.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np

from dmx.bstats.nulldist import NullDataEncoder, null_dist
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)

Model = ProbabilityDistribution[Any, Any, Any]


class IgnoredDistribution(ProbabilityDistribution[Any, Any, Any]):
    """Wrap a fixed distribution that participates in scoring but not fitting."""

    def __init__(self, dist: Model = null_dist) -> None:
        """Initialize the wrapper around the distribution to keep fixed."""
        super().__init__()
        self.dist = dist

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return f"IgnoredDistribution({self.dist})"

    def get_prior(self) -> Model:
        """Return the wrapped distribution's prior."""
        return self.dist.get_prior()

    def set_prior(self, prior: Model) -> None:
        """Set the wrapped distribution's prior without changing ignored semantics."""
        self.dist.set_prior(prior)

    def set_parameters(self, value: Any) -> None:
        """Set parameters on the wrapped fixed distribution."""
        self.dist.set_parameters(value)

    def get_parameters(self) -> Any:
        """Return parameters from the wrapped fixed distribution."""
        return self.dist.get_parameters()

    def cross_entropy(self, dist: Model) -> float:
        """Delegate cross-entropy evaluation to the wrapped distribution."""
        return self.dist.cross_entropy(dist)

    def entropy(self) -> float:
        """Delegate entropy evaluation to the wrapped distribution."""
        return self.dist.entropy()

    def density(self, x: Any) -> float:
        """Evaluate the wrapped distribution's density."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Evaluate the wrapped distribution's log-density."""
        return self.dist.log_density(x)

    def expected_log_density(self, x: Any) -> float:
        """Evaluate the wrapped distribution's expected log-density."""
        return self.dist.expected_log_density(x)

    def seq_log_density(self, x: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Evaluate wrapped log-densities for encoded observations."""
        encoded = x.data if isinstance(x, IgnoredEncodedData) else x
        return np.asarray(self.dist.seq_log_density(encoded), dtype=np.float64)

    def seq_expected_log_density(self, x: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Evaluate wrapped expected log-densities for encoded observations."""
        encoded = x.data if isinstance(x, IgnoredEncodedData) else x
        return np.asarray(self.dist.seq_expected_log_density(encoded), dtype=np.float64)

    def seq_encode(self, x: Iterable[Any]) -> Any:
        """Delegate direct sequence encoding to the wrapped distribution."""
        return self.dist.seq_encode(x)

    def sampler(self, seed: Optional[int] = None) -> "IgnoredSampler":
        """Create a sampler backed by the wrapped distribution."""
        return IgnoredSampler(self, seed)

    def estimator(self) -> "IgnoredEstimator":
        """Create an estimator that preserves the wrapped distribution."""
        return IgnoredEstimator(dist=self.dist)

    def dist_to_encoder(self) -> "IgnoredDataEncoder":
        """Create an encoder backed by the wrapped distribution's encoder."""
        return IgnoredDataEncoder(self.dist.dist_to_encoder())


class IgnoredSampler(DistributionSampler[Any]):
    """Sampler delegating all draws to the wrapped fixed distribution."""

    def __init__(self, dist: IgnoredDistribution, seed: Optional[int] = None) -> None:
        """Initialize a sampler for the wrapped distribution."""
        super().__init__(dist, seed)
        self.dist_sampler = dist.dist.sampler(seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw from the wrapped distribution."""
        return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(StatisticAccumulator[Any, None, Any]):
    """Accumulator that deliberately discards all observations and weights."""

    def __init__(self, encoder: Optional[DataSequenceEncoder[Any, Any]] = None) -> None:
        """Initialize with the wrapped distribution's sequence encoder."""
        self.encoder = encoder if encoder is not None else NullDataEncoder()

    def update(self, x: Any, weight: float, estimate: Optional[Model]) -> None:
        """Discard one weighted observation."""
        del x, weight, estimate

    def seq_update(
        self, x: Any, weights: np.ndarray[Any, Any], estimate: Optional[Model]
    ) -> None:
        """Discard a sequence of weighted observations."""
        del x, weights, estimate

    def seq_initialize(
        self, x: Any, weights: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> None:
        """Discard encoded observations during randomized initialization."""
        del x, weights, rng

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Discard one observation during randomized initialization."""
        del x, weight, rng

    def combine(self, suff_stat: None) -> "IgnoredAccumulator":
        """Combine the sole ignored statistic value."""
        del suff_stat
        return self

    def value(self) -> None:
        """Return the sole ignored sufficient statistic."""
        return None

    def from_value(self, x: None) -> "IgnoredAccumulator":
        """Restore the sole ignored sufficient statistic."""
        del x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged."""
        del stats_dict

    def acc_to_encoder(self) -> "IgnoredDataEncoder":
        """Create the encoder corresponding to this accumulator."""
        return IgnoredDataEncoder(self.encoder)


class IgnoredAccumulatorFactory(StatisticAccumulatorFactory[Any, None, Any]):
    """Create fresh accumulators that ignore estimation data."""

    def __init__(self, encoder: DataSequenceEncoder[Any, Any]) -> None:
        """Store the wrapped distribution's encoder."""
        self.encoder = encoder

    def make(self) -> IgnoredAccumulator:
        """Create an ignored accumulator."""
        return IgnoredAccumulator(self.encoder)


class IgnoredEstimator(ParameterEstimator[Any, Any, Any, None]):
    """Estimator preserving a wrapped distribution regardless of observations."""

    def __init__(
        self,
        dist: Model = null_dist,
        prior: Model = null_dist,
        keys: Any = None,
    ) -> None:
        """Initialize the fixed distribution and compatibility metadata."""
        self.dist = dist
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self) -> IgnoredAccumulatorFactory:
        """Create a factory whose accumulators discard all observations."""
        return IgnoredAccumulatorFactory(self.dist.dist_to_encoder())

    def get_prior(self) -> Model:
        """Return the fixed distribution's current prior."""
        return self.dist.get_prior()

    def set_prior(self, prior: Model) -> None:
        """Set the prior of the fixed distribution."""
        self.prior = prior
        self.dist.set_prior(prior)

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> IgnoredDistribution:
        """Ignore either statistic call form and preserve the fixed distribution."""
        del args
        return IgnoredDistribution(self.dist)


class IgnoredDataEncoder(DataSequenceEncoder[Any, Any]):
    """Wrap the sequence encoder belonging to the fixed distribution."""

    def __init__(self, encoder: DataSequenceEncoder[Any, Any]) -> None:
        """Initialize with the wrapped distribution's encoder."""
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a stable description containing the wrapped encoder."""
        return f"IgnoredDataEncoder({self.encoder})"

    def __eq__(self, other: object) -> bool:
        """Return whether both ignored encoders wrap equivalent encoders."""
        return isinstance(other, IgnoredDataEncoder) and self.encoder == other.encoder

    def seq_encode(self, x: Iterable[Any]) -> "IgnoredEncodedData":
        """Encode observations using the fixed distribution's encoder."""
        encoded = self.encoder.seq_encode(x)
        return IgnoredEncodedData(encoded.data)


class IgnoredEncodedData(  # pylint: disable=too-few-public-methods
    EncodedDataSequence[Any]
):
    """Contain encoded data produced by a fixed distribution's encoder."""

    def __init__(self, data: Any) -> None:
        """Store wrapped encoded data."""
        super().__init__(data)
