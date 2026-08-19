"""Define the core interfaces used by Bayesian statistical models.

The classes in this module describe the contracts shared by distributions,
samplers, sufficient-statistic accumulators, estimators, and sequence encoders
in :mod:`dmx.bstats`. Methods raise :class:`NotImplementedError` when no safe
general implementation exists. The classes deliberately avoid abstract base
class enforcement because legacy distributions implement partial protocols.
"""

# pylint: disable=abstract-method
# Pylint infers abstractness from NotImplementedError, but ABC enforcement is
# intentionally avoided so legacy partial protocol implementations still work.

from collections.abc import Callable, Iterable, MutableMapping
from typing import Any, Generic, Optional, TypeVar, cast, overload

import numpy as np
import pandas as pd

from dmx.arithmetic import exp, maxrandint

X = TypeVar("X")
E = TypeVar("E")
P = TypeVar("P")
V = TypeVar("V")
SS = TypeVar("SS")

noname_instance_count: int = 0


class ProbabilityDistribution(Generic[X, P, V]):
    """Base contract for a Bayesian probability distribution."""

    name: Optional[str] = None
    params: P
    parents: list["ProbabilityDistribution[Any, Any, Any]"]
    prior: "ProbabilityDistribution[Any, Any, Any]"

    def __init__(self) -> None:
        """Initialize optional distribution metadata."""
        self.name = None
        self.parents = []

    def get_parameters(self) -> P:
        """Return the distribution parameters."""
        return self.params

    def set_parameters(self, value: P) -> None:
        """Replace the distribution parameters.

        Args:
            value: New parameter value.
        """
        self.params = value

    def get_name(self) -> Optional[str]:
        """Return the optional distribution name."""
        return self.name

    def set_name(self, name: Optional[str]) -> None:
        """Set the optional distribution name.

        Args:
            name: Name used by DataFrame-based helpers.
        """
        self.name = name

    def add_parent(self, dist: "ProbabilityDistribution[Any, Any, Any]") -> None:
        """Register a distribution that contains this distribution.

        Args:
            dist: Parent distribution to register.
        """
        if not hasattr(self, "parents"):
            self.parents = []
        self.parents.append(dist)

    def get_prior(self) -> "ProbabilityDistribution[Any, Any, Any]":
        """Return the prior distribution.

        Raises:
            NotImplementedError: If this distribution does not expose a prior.
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose a prior")

    def set_prior(self, prior: "ProbabilityDistribution[Any, Any, Any]") -> None:
        """Set the prior distribution.

        Args:
            prior: Prior distribution to associate with this distribution.

        Raises:
            NotImplementedError: If this distribution does not support a prior.
        """
        del prior
        raise NotImplementedError(f"{type(self).__name__} does not support a prior")

    def density(self, x: X) -> float:
        """Evaluate the probability density at an observation.

        Args:
            x: Observation to evaluate.

        Returns:
            Probability density at ``x``.
        """
        return float(exp(self.log_density(x)))

    def log_density(self, x: X) -> float:
        """Evaluate the log-density at an observation.

        Args:
            x: Observation to evaluate.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x
        raise NotImplementedError(f"{type(self).__name__} has no log-density")

    def expected_log_density(self, x: X) -> float:
        """Evaluate the posterior expected log-density.

        Args:
            x: Observation to evaluate.

        Returns:
            The plug-in log-density by default.
        """
        return self.log_density(x)

    def entropy(self) -> float:
        """Return the distribution entropy.

        Raises:
            NotImplementedError: If entropy has no generic implementation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define entropy")

    def cross_entropy(self, dist: "ProbabilityDistribution[Any, Any, Any]") -> float:
        """Return the cross-entropy relative to another distribution.

        Args:
            dist: Distribution used for the cross-entropy expectation.

        Raises:
            NotImplementedError: If no generic implementation exists.
        """
        del dist
        raise NotImplementedError(
            f"{type(self).__name__} does not define cross-entropy"
        )

    def seq_log_density(self, x: V) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Evaluate log-densities for an iterable encoded sequence.

        Args:
            x: Encoded observations.

        Returns:
            One log-density per observation.
        """
        observations = cast(Iterable[X], x)
        return np.asarray(
            [self.log_density(value) for value in observations], dtype=np.float64
        )

    def seq_expected_log_density(self, x: V) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Evaluate expected log-densities for an encoded sequence.

        Args:
            x: Encoded observations.

        Returns:
            One expected log-density per observation.
        """
        observations = cast(Iterable[X], x)
        return np.asarray(
            [self.expected_log_density(value) for value in observations],
            dtype=np.float64,
        )

    def seq_encode(self, x: Iterable[X]) -> V:
        """Encode observations, preserving the input sequence by default.

        Args:
            x: Observations to encode.

        Returns:
            Encoded observations.
        """
        return cast(V, x)

    def df_log_density(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate log-density for the named column of a DataFrame.

        Args:
            df: DataFrame containing the named observation column.

        Returns:
            Log-density indexed like ``df``.

        Raises:
            ValueError: If the distribution has no column name.
        """
        if self.name is None:
            raise ValueError("A distribution name is required for DataFrame scoring")
        return df[self.name].map(self.log_density)

    def sampler(self, seed: Optional[int] = None) -> "DistributionSampler[X]":
        """Create a sampler for the distribution.

        Args:
            seed: Optional random seed.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del seed
        raise NotImplementedError(f"{type(self).__name__} has no sampler")

    def estimator(self) -> "ParameterEstimator[X, P, V, Any]":
        """Create a parameter estimator for the distribution.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} has no estimator")

    def dist_to_encoder(self) -> "DataSequenceEncoder[X, V]":
        """Create the sequence encoder associated with this distribution.

        Raises:
            NotImplementedError: If the distribution has no separate encoder.
        """
        raise NotImplementedError(f"{type(self).__name__} has no data encoder")


class ProbabilityDistributionFactory(Generic[X, P, V]):
    """Factory for distributions with a shared parameter type."""

    def make(self, params: P) -> ProbabilityDistribution[X, P, V]:
        """Create a distribution from parameters.

        Args:
            params: Parameters for the new distribution.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del params
        raise NotImplementedError(f"{type(self).__name__} cannot make distributions")


class DistributionSampler(Generic[X]):
    """Base contract for sampling observations from a distribution."""

    def __init__(self, dist: Any, seed: Optional[int] = None) -> None:
        """Initialize the sampler.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.
        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def new_seed(self) -> int:
        """Draw a child seed from this sampler's random state."""
        return int(self.rng.randint(0, maxrandint))

    @overload
    def sample(self, size: None = None) -> X: ...

    @overload
    def sample(self, size: int) -> Any: ...

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one or more observations.

        Args:
            size: Number of observations, or ``None`` for one observation.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del size
        raise NotImplementedError(f"{type(self).__name__} cannot sample")


class StatisticAccumulator(Generic[X, SS, V]):
    """Base contract for accumulating sufficient statistics."""

    def update(
        self,
        x: X,
        weight: float,
        estimate: Optional[ProbabilityDistribution[Any, Any, Any]],
    ) -> None:
        """Accumulate one weighted observation.

        Args:
            x: Observation to accumulate.
            weight: Observation weight.
            estimate: Optional current distribution estimate.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x, weight, estimate
        raise NotImplementedError(f"{type(self).__name__} cannot update statistics")

    def initialize(self, x: X, weight: float, rng: np.random.RandomState) -> None:
        """Initialize statistics from one weighted observation.

        Args:
            x: Observation to accumulate.
            weight: Observation weight.
            rng: Random state available to randomized initializers.
        """
        del rng
        self.update(x, weight, estimate=None)

    def combine(self, suff_stat: SS) -> "StatisticAccumulator[X, SS, V]":
        """Merge sufficient statistics into this accumulator.

        Args:
            suff_stat: Sufficient statistics to merge.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del suff_stat
        raise NotImplementedError(f"{type(self).__name__} cannot combine statistics")

    def value(self) -> SS:
        """Return the accumulated sufficient statistics.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} has no statistic value")

    def from_value(self, x: SS) -> "StatisticAccumulator[X, SS, V]":
        """Restore sufficient statistics from a value.

        Args:
            x: Sufficient statistics to restore.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x
        raise NotImplementedError(f"{type(self).__name__} cannot restore statistics")

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics shared through named keys.

        Args:
            stats_dict: Mutable mapping of shared statistic values.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del stats_dict
        raise NotImplementedError(
            f"{type(self).__name__} cannot merge keyed statistics"
        )

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace statistics from values shared through named keys.

        Args:
            stats_dict: Mutable mapping of shared statistic values.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del stats_dict
        raise NotImplementedError(
            f"{type(self).__name__} cannot replace keyed statistics"
        )

    def acc_to_encoder(self) -> "DataSequenceEncoder[X, V]":
        """Create the sequence encoder associated with this accumulator.

        Raises:
            NotImplementedError: If the accumulator has no separate encoder.
        """
        raise NotImplementedError(f"{type(self).__name__} has no data encoder")


class StatisticAccumulatorFactory(Generic[X, SS, V]):
    """Factory for sufficient-statistic accumulators."""

    def make(self) -> StatisticAccumulator[X, SS, V]:
        """Create a fresh accumulator.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot make accumulators")


class ParameterEstimator(Generic[X, P, V, SS]):
    """Estimate a distribution from sufficient statistics."""

    @overload
    def estimate(self, suff_stat: SS, /) -> ProbabilityDistribution[X, P, V]: ...

    @overload
    def estimate(
        self, nobs: Optional[float], suff_stat: SS, /
    ) -> ProbabilityDistribution[X, P, V]: ...

    def estimate(self, *args: Any) -> ProbabilityDistribution[X, P, V]:
        """Estimate using either supported legacy call form.

        Args:
            *args: Either ``suff_stat`` or ``nobs, suff_stat``.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del args
        raise NotImplementedError(f"{type(self).__name__} cannot estimate parameters")

    def accumulator_factory(self) -> StatisticAccumulatorFactory[X, SS, V]:
        """Create an accumulator factory, including legacy implementations.

        Raises:
            NotImplementedError: If neither method spelling is implemented.
        """
        legacy_method = type(self).accumulatorFactory
        if legacy_method is not ParameterEstimator.accumulatorFactory:
            return legacy_method(self)
        raise NotImplementedError(f"{type(self).__name__} has no accumulator factory")

    def accumulatorFactory(self) -> StatisticAccumulatorFactory[X, SS, V]:
        """Return the factory using the legacy method spelling.

        Raises:
            NotImplementedError: If neither method spelling is implemented.
        """
        modern_method = type(self).accumulator_factory
        if modern_method is not ParameterEstimator.accumulator_factory:
            return modern_method(self)
        raise NotImplementedError(f"{type(self).__name__} has no accumulator factory")

    def get_prior(self) -> ProbabilityDistribution[Any, Any, Any]:
        """Return the estimator prior.

        Raises:
            NotImplementedError: If the estimator does not expose a prior.
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose a prior")

    def set_prior(self, prior: ProbabilityDistribution[Any, Any, Any]) -> None:
        """Set the estimator prior.

        Args:
            prior: Prior distribution for subsequent estimates.

        Raises:
            NotImplementedError: If the estimator does not support a prior.
        """
        del prior
        raise NotImplementedError(f"{type(self).__name__} does not support a prior")


class SequenceEncodableDistribution(ProbabilityDistribution[X, P, V]):
    """Distribution with default sequence-oriented operations."""

    def seq_log_density_lambda(self) -> list[Callable[[V], np.ndarray[Any, Any]]]:
        """Return the sequence log-density callable for legacy helpers."""
        return [self.seq_log_density]

    def seq_encode(self, x: Iterable[X]) -> V:
        """Preserve an already sequence-compatible input."""
        return cast(V, x)


class DataFrameEncodableDistribution(ProbabilityDistribution[X, P, V]):
    """Distribution that scores observations in a named DataFrame column."""


class SequenceEncodableAccumulator(StatisticAccumulator[X, SS, V]):
    """Accumulator with vectorized sequence update operations."""

    def get_seq_lambda(self) -> list[Callable[..., None]]:
        """Return the sequence update callable for legacy helpers."""
        return [self.seq_update]

    def seq_initialize(
        self, x: V, weights: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> None:
        """Initialize statistics from an encoded sequence.

        Args:
            x: Encoded observations.
            weights: Observation weights.
            rng: Random state for randomized initialization.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x, weights, rng
        raise NotImplementedError(
            f"{type(self).__name__} cannot initialize sequence statistics"
        )

    def seq_update(
        self,
        x: V,
        weights: np.ndarray[Any, Any],
        estimate: Optional[ProbabilityDistribution[Any, Any, Any]],
    ) -> None:
        """Update statistics from an encoded sequence.

        Args:
            x: Encoded observations.
            weights: Observation weights.
            estimate: Optional current distribution estimate.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x, weights, estimate
        raise NotImplementedError(
            f"{type(self).__name__} cannot update sequence statistics"
        )


class DataFrameEncodableAccumulator(StatisticAccumulator[X, SS, V]):
    """Accumulator that reads observations from a named DataFrame column."""

    name: Optional[str] = None

    def df_initialize(
        self,
        df: pd.DataFrame,
        weights: Iterable[float],
        rng: np.random.RandomState,
    ) -> None:
        """Initialize statistics from a DataFrame column.

        Args:
            df: DataFrame containing the named observation column.
            weights: Observation weights.
            rng: Random state for randomized initialization.

        Raises:
            ValueError: If the accumulator has no column name.
        """
        if self.name is None:
            raise ValueError("An accumulator name is required for DataFrame updates")
        for value, weight in zip(df[self.name], weights):
            self.initialize(cast(X, value), weight, rng)

    def df_update(
        self,
        df: pd.DataFrame,
        weights: Iterable[float],
        estimate: Optional[ProbabilityDistribution[Any, Any, Any]],
    ) -> None:
        """Update statistics from a DataFrame column.

        Args:
            df: DataFrame containing the named observation column.
            weights: Observation weights.
            estimate: Optional current distribution estimate.

        Raises:
            ValueError: If the accumulator has no column name.
        """
        if self.name is None:
            raise ValueError("An accumulator name is required for DataFrame updates")
        for value, weight in zip(df[self.name], weights):
            self.update(cast(X, value), weight, estimate)


class DataSequenceEncoder(Generic[X, E]):
    """Convert observations to encoded sequence data."""

    def __str__(self) -> str:
        """Return a stable default encoder name."""
        return type(self).__name__

    def seq_encode(self, x: Iterable[X]) -> "EncodedDataSequence[E]":
        """Encode independent observations for vectorized operations.

        Args:
            x: Observations to encode.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        del x
        raise NotImplementedError(f"{type(self).__name__} cannot encode sequences")

    def __eq__(self, other: object) -> bool:
        """Return whether two encoders have the same concrete type."""
        return type(self) is type(other)


class EncodedDataSequence(Generic[E]):
    """Contain data produced by a :class:`DataSequenceEncoder`."""

    def __init__(self, data: E) -> None:
        """Store encoded data.

        Args:
            data: Distribution-specific encoded representation.
        """
        self.data = data

    def __repr__(self) -> str:
        """Return a representation containing the encoded data."""
        return f"{type(self).__name__}(data={self.data!r})"
