"""Provide a wrapper for observations carrying multiplicative weights.

This Distribution simply allows from weights on observations. I.e. Data type D is
observed and an associated
score/weight is assigned to the data. This simply passes the weights and data downstream
in aggregation.

Likelihood evals are equivalent to normal likelihood calls to the base distribution.

"""

from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np

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
SS = TypeVar("SS")


class WeightedDistribution(SequenceEncodableProbabilityDistribution):
    """Scale a child's log likelihood by an observation's embedded weight.

    Observations are ``(value, weight)`` pairs and scoring returns
    ``weight * child.log_density(value)``. Thus the density is the child density
    raised to ``weight``; it is an objective contribution and is not generally a
    normalized joint distribution over value-weight pairs. Child support is
    preserved for the value field, subject to the numeric embedded weight.

    Encoding separates child values from a float weight vector. Accumulation
    multiplies the caller's weight by each embedded weight before delegating to the
    child. Sampling delegates directly to the child and therefore returns unweighted
    child values, not value-weight pairs.

    Notes:
        Distribution acts only on the value for likelihood calls and treats weight as
        number of replicates.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution for values.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Keys for parameters of dist.

    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a weighted-objective wrapper around ``dist``.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution for values.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Keys for parameters of dist.

        """
        super().__init__()
        self.dist = dist
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable representation of the wrapper."""
        return (
            f"WeightedDistribution(dist={repr(self.dist)}, name={repr(self.name)}, "
            f"keys={repr(self.keys)})"
        )

    def density(self, x: Tuple[T, float]) -> float:
        """Return the exponentiated weighted child log density."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Tuple[T, float]) -> float:
        """Multiply the child's log density by the embedded weight."""
        return self.dist.log_density(x[0]) * x[1]

    def seq_log_density(self, x: "WeightedEncodedDataSequence") -> np.ndarray:
        """Multiply vectorized child log densities by encoded weights.

        Raises:
            TypeError: If ``x`` is not a weighted encoded sequence.
        """
        if not isinstance(x, WeightedEncodedDataSequence):
            raise TypeError(
                "WeightedEncodedDataSequence required for seq_log_density()."
            )

        return np.asarray(self.dist.seq_log_density(x.data[0]) * x.data[1])

    def dist_to_encoder(self) -> "WeightedDataEncoder":
        """Create an encoder for child values and embedded weights."""
        return WeightedDataEncoder(encoder=self.dist.dist_to_encoder())

    def estimator(self, pseudo_count: Optional[float] = None) -> "WeightedEstimator":
        """Wrap the child's estimator, forwarding any pseudo-count."""
        if pseudo_count is not None:
            return WeightedEstimator(
                estimator=self.dist.estimator(pseudo_count=pseudo_count),
                name=self.name,
                keys=self.keys,
            )
        return WeightedEstimator(
            estimator=self.dist.estimator(), name=self.name, keys=self.keys
        )

    def sampler(self, seed: Optional[int] = None) -> "DistributionSampler":
        """Return the child sampler without adding embedded weights."""
        return self.dist.sampler(seed)


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate statistics using external weight times embedded weight.

    The wrapper adds no statistic of its own: ``value``, ``combine``, and
    ``from_value`` use the child's statistic directly. A wrapper key shares that
    child statistic but does not recurse through the child's own key contract.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for base
            distribution.
        keys (Optional[str]): Key for sufficient statistics of base distribution.
        name (Optional[str]): Optional name for distribution.

    """

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize a weighted wrapper around a child accumulator.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for base
                distribution.
            keys (Optional[str]): Key for sufficient statistics of base distribution.
            name (Optional[str]): Optional name for distribution.

        """
        self.accumulator = accumulator
        self.keys = keys
        self.name = name

    def initialize(
        self, x: Tuple[T, float], weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize the child using the product of both weights."""
        self.accumulator.initialize(x[0], weight * x[1], rng)

    def update(
        self, x: Tuple[T, float], weight: float, estimate: WeightedDistribution
    ) -> None:
        """Update the child using the product of both weights."""
        self.accumulator.update(x[0], weight * x[1], estimate.dist)

    def seq_update(
        self,
        x: "WeightedEncodedDataSequence",
        weights: np.ndarray,
        estimate: WeightedDistribution,
    ) -> None:
        """Update encoded child values using elementwise weight products."""
        self.accumulator.seq_update(x.data[0], weights * x.data[1], estimate.dist)

    def seq_initialize(
        self,
        x: "WeightedEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize encoded child values using elementwise weight products."""
        self.accumulator.seq_initialize(x.data[0], weights * x.data[1], rng)

    def combine(self, suff_stat: SS) -> "WeightedAccumulator":
        """Combine a child sufficient statistic and return this wrapper."""
        self.accumulator.combine(suff_stat)

        return self

    def from_value(self, x: SS) -> "WeightedAccumulator":
        """Restore the child sufficient statistic and return this wrapper."""
        self.accumulator.from_value(x)

        return self

    def value(self) -> Any:
        """Return the child sufficient statistic unchanged."""
        return self.accumulator.value()

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the child statistic under the wrapper key when configured."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.accumulator.combine(stats_dict[self.keys].value())
            else:
                stats_dict[self.keys] = self.accumulator

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the child statistic from the wrapper key when available."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.accumulator.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "WeightedDataEncoder":
        """Wrap the child accumulator's encoder."""
        return WeightedDataEncoder(encoder=self.accumulator.acc_to_encoder())


class WeightedAccumulatorFactory(StatisticAccumulatorFactory):
    """WeightedAccumulatorFactory object for creating WeightedAccumulator objects.

    Attributes:
        factory (StatisticAccumulatorFactory): Accumulator for base distribution.
        keys (Optional[str]): Optional keys for base distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(
        self,
        factory: StatisticAccumulatorFactory,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize a factory around a child accumulator factory.

        Args:
            factory (StatisticAccumulatorFactory): Accumulator for base distribution.
            keys (Optional[str]): Optional keys for base distribution.
            name (Optional[str]): Name for object.

        """
        self.factory = factory
        self.keys = keys
        self.name = name

    def make(self) -> "WeightedAccumulator":
        """Create a weighted wrapper around a new child accumulator."""
        return WeightedAccumulator(
            accumulator=self.factory.make(), name=self.name, keys=self.keys
        )


class WeightedEstimator(ParameterEstimator):
    """Fit a child from already weighted sufficient statistics.

    The wrapper passes ``nobs`` through unchanged to the child estimator; embedded
    weights affect the accumulated child statistic but do not independently replace
    that argument. The fitted child is returned inside a new weighted wrapper.

    Attributes:
        estimator (ParameterEstimator): Estimator for the base distribution.
        keys (Optional[str]): Keys for the base distribution.
        name (Optional[str]): Optional name for object.

    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize a wrapper around a child estimator.

        Args:
            estimator (ParameterEstimator): Estimator for the base distribution.
            keys (Optional[str]): Keys for the base distribution.
            name (Optional[str]): Optional name for object.

        """
        self.estimator = estimator
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "WeightedAccumulatorFactory":
        """Create a weighted factory around the child's factory."""
        return WeightedAccumulatorFactory(
            factory=self.estimator.accumulator_factory(), keys=self.keys, name=self.name
        )

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> "WeightedDistribution":
        """Fit the child with the supplied count and child statistic."""
        return WeightedDistribution(
            dist=self.estimator.estimate(nobs, suff_stat), name=self.name
        )


class WeightedDataEncoder(DataSequenceEncoder):
    """WeightedDataEncoder object for encoding iid sequences of WeightedDistribution.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the base distribution.

    """

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """Initialize an encoder around the child encoder.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder for the base
                distribution.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a representation containing the child encoder."""
        return f"WeightedDataEncoder(encoder={repr(self.encoder)})"

    def __eq__(self, other: object) -> bool:
        """Compare weighted encoders by their child encoders."""
        if isinstance(other, WeightedDataEncoder):
            return other.encoder == self.encoder
        return False

    def seq_encode(self, x: Sequence[Tuple[T, float]]) -> "WeightedEncodedDataSequence":
        """Encode child values and store embedded weights as a float array."""
        rv_enc = self.encoder.seq_encode([xx[0] for xx in x]), np.asarray(
            [xx[1] for xx in x], dtype=float
        )

        return WeightedEncodedDataSequence(data=rv_enc)


class WeightedEncodedDataSequence(EncodedDataSequence):
    """WeightedEncodedDataSequence object for vectorized calls.

    Attributes:
        data (Tuple[EncodedDataSequence, np.ndarray]): EncodedDataSequence for base
            distribution and array of counts.

    """

    def __init__(self, data: Tuple[EncodedDataSequence, np.ndarray]) -> None:
        """Store the child encoding alongside its embedded weight vector.

        Args:
            data (Tuple[EncodedDataSequence, np.ndarray]): EncodedDataSequence for base
                distribution and array of counts.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing child data and weights."""
        return f"WeightedEncodedDataSequence(data={self.data})"
