"""Create, estimate, and sample from an IgnoredDistribution.

Defines the IgnoredDistribution, IgnoredSampler, IgnoredAccumulatorFactory,
IgnoredAccumulator, IgnoredEstimator,
and the IgnoredDataEncoder classes for use with dmx-learn.

Ignored distribution is simply a distribution that is ignored in estimation and treated
as fixed.

"""

from typing import Any, Dict, Optional, Sequence, TypeVar

import numpy as np
from numpy.random import RandomState

from dmx.stats.null_dist import NullDataEncoder, NullDistribution, NullSampler
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

T = TypeVar("T")
E = TypeVar("E")


class IgnoredDistribution(SequenceEncodableProbabilityDistribution):
    """Preserve a child distribution while excluding it from estimation.

    The wrapper has exactly the child's support, density, normalization, encoding,
    and sampling behavior. Its accumulator deliberately records no sufficient
    statistics, and its estimator always returns a new wrapper around the original
    child without fitting that child. Thus "ignored" applies only to estimation.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution to be ignored.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Keys for distribution (just a place holder).

    """

    def __init__(
        self,
        dist: Optional[SequenceEncodableProbabilityDistribution],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize an estimation-ignored child distribution.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to
                be ignored.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Keys for distribution (just a place holder).

        """
        super().__init__()
        self.dist = dist if dist is not None else NullDistribution()
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable representation of the wrapper."""
        return (
            f"IgnoredDistribution({repr(self.dist)}, name={repr(self.name)}, "
            f"keys={repr(self.keys)})"
        )

    def density(self, x: T) -> float:
        """Evaluate the density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            float: Density of attribute 'dist' at x

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: T) -> float:
        """Evaluate the log-density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            float: log-density of attribute 'dist' at x.

        """
        return self.dist.log_density(x)

    def seq_log_density(self, x: EncodedDataSequence) -> np.ndarray:
        """Delegate vectorized scoring to the child distribution.

        Args:
            x: An ignored-wrapper encoding or the child's encoding directly.

        Returns:
            The child's vector of log densities.

        Raises:
            TypeError: If ``x`` is not an encoded data sequence.
        """
        if isinstance(x, IgnoredEncodedDataSequence):
            rv = self.dist.seq_log_density(x.data)
        elif not isinstance(x, IgnoredEncodedDataSequence) and isinstance(
            x, EncodedDataSequence
        ):
            rv = self.dist.seq_log_density(x)
        else:
            raise TypeError("Wrong EncodedDataSequence passed to seq_log_density().")

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IgnoredSampler":
        """Create a sampler that delegates to the child sampler."""
        return IgnoredSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "IgnoredEstimator":
        """Create an estimator that preserves, rather than fits, the child."""
        return IgnoredEstimator(dist=self.dist, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "IgnoredDataEncoder":
        """Wrap the child's sequence encoder."""
        return IgnoredDataEncoder(encoder=self.dist.dist_to_encoder())


class IgnoredSampler(DistributionSampler):
    """IgnoredSampler object for generating samples from Ignored distribution.

    Attributes:
        dist_sampler (DistributionSampler): DistributionSampler for ignored
            distribution.
        null_sampler (bool): True if IgnoredDistribution is the NullDistribution.

    """

    def __init__(self, dist: IgnoredDistribution, seed: Optional[int] = None) -> None:
        """Initialize a sampler backed by the ignored child.

        Attributes:
            dist (IgnoredDistribution): DistributionSampler for ignored distribution.
            seed (Optional[int]): Set seed for generating random samples.

        """
        super().__init__(dist, seed)
        self.dist_sampler = dist.dist.sampler(seed)
        self.null_sampler = isinstance(self.dist_sampler, NullSampler)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw from the child, preserving its scalar or sized return contract."""
        if self.null_sampler:
            if size is None:
                return None
            return [None] * size
        return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(SequenceEncodableStatisticAccumulator):
    """Implement the accumulator protocol while retaining no statistics.

    All update, initialization, combination, and key-sharing operations are
    intentional no-ops. ``value`` is always ``None``; the encoder is retained only
    so callers can continue to encode and score the ignored child.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the ignored distribution.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Name for param dists (place holder only).

    """

    def __init__(
        self,
        encoder: Optional[DataSequenceEncoder] = NullDataEncoder(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a no-op accumulator with the child's encoder.

        Args:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for the ignored
                distribution.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Name for param dists (place holder only).

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name
        self.keys = keys

    def update(
        self, x: T, weight: float, estimate: Optional[IgnoredDistribution]
    ) -> None:
        """Ignore one observation and its weight."""

    def seq_update(
        self,
        x: "IgnoredEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[IgnoredDistribution],
    ) -> None:
        """Ignore an encoded sequence and its weights."""

    def initialize(self, x: T, weight: float, rng: Optional[RandomState]) -> None:
        """Ignore one initialization observation."""

    def seq_initialize(
        self,
        x: "IgnoredEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Ignore an encoded initialization sequence."""

    def combine(self, suff_stat: Any) -> "IgnoredAccumulator":
        """Ignore another statistic and return this accumulator."""
        return self

    def value(self) -> None:
        """Return the invariant null sufficient statistic."""
        return None

    def from_value(self, x: Any) -> "IgnoredAccumulator":
        """Ignore a supplied statistic and return this accumulator."""
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Leave the shared-statistics dictionary unchanged."""

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Leave this no-op accumulator unchanged."""

    def acc_to_encoder(self) -> "IgnoredDataEncoder":
        """Return an ignored wrapper around the retained child encoder."""
        return IgnoredDataEncoder(encoder=self.encoder)


class IgnoredAccumulatorFactory(StatisticAccumulatorFactory):
    """IgnoredAccumulatorFactory for creating IgnoredAccumulator objects.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for base distribution.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Keys for distribution (just a place holder).

    """

    def __init__(
        self,
        encoder: Optional[DataSequenceEncoder] = NullDataEncoder(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a factory for no-op accumulators.

        Args:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for base
                distribution.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Keys for distribution (just a place holder).

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name
        self.keys = keys

    def make(self) -> "IgnoredAccumulator":
        """Create a no-op accumulator retaining the configured encoder."""
        return IgnoredAccumulator(encoder=self.encoder, name=self.name, keys=self.keys)


class IgnoredEstimator(ParameterEstimator):
    """Return the original child regardless of accumulated data.

    ``pseudo_count`` and ``suff_stat`` are placeholders. The produced accumulator
    is a no-op and ``estimate`` never invokes an estimator on ``dist``.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution to be ignored.
        pseudo_count (Optional[float]): Place holder for consistency.
        suff_stat (Optional[Any]): Place holder for consistency.
        keys (Optional[str]): Place holder for consistency.
        name (Optional[str]): Set name for object instance.

    """

    def __init__(
        self,
        dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Any] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an estimator that freezes ``dist``.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to
                be ignored.
            pseudo_count (Optional[float]): Place holder for consistency.
            suff_stat (Optional[Any]): Place holder for consistency.
            keys (Optional[str]): Place holder for consistency.
            name (Optional[str]): Set name for object instance.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("IgnoredEstimator requires keys to be of type 'str'.")

        self.dist = dist if dist is not None else NullDistribution()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IgnoredAccumulatorFactory":
        """Create a no-op accumulator factory using the child's encoder."""
        return IgnoredAccumulatorFactory(
            self.dist.dist_to_encoder(), name=self.name, keys=self.keys
        )

    def estimate(self, nobs: Optional[float], suff_stat: Any) -> IgnoredDistribution:
        """Wrap the unchanged child, ignoring counts and sufficient statistics."""
        return IgnoredDistribution(self.dist, name=self.name)


class IgnoredDataEncoder(DataSequenceEncoder):
    """IgnoredDataEncoder object for encoding sequences of data of ignored distribution.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for ignored distribution.
        null (bool): True if the DataSequenceEncoder is NullDataEncoder.

    """

    def __init__(
        self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder()
    ) -> None:
        """Initialize a transparent wrapper around a child encoder.

        Attributes:
            encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for ignored
                distribution.

        """
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.null = isinstance(self.encoder, NullDataEncoder)

    def __str__(self) -> str:
        """Return a representation containing the child encoder."""
        return "IgnoredDataEncoder(dist=" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Compare wrappers by their child encoders."""
        if isinstance(other, IgnoredDataEncoder):
            return other.encoder == self.encoder
        return False

    def seq_encode(self, x: Sequence[T]) -> "IgnoredEncodedDataSequence":
        """Encode all observations with the child encoder and wrap the result."""
        return IgnoredEncodedDataSequence(data=self.encoder.seq_encode(x))


class IgnoredEncodedDataSequence(EncodedDataSequence):
    """IgnoredEncodedDataSequence object for vectorized calls.

    Attributes:
        data (EncodedDataSequence): EncodedDataSequence object for ignored distribution.

    """

    def __init__(self, data: EncodedDataSequence):
        """Store a child encoding without changing it.

        Args:
            data (EncodedDataSequence): EncodedDataSequence object for ignored
                distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the child encoding."""
        return f"IgnoredEncodedDataSequence(data={self.data})"
