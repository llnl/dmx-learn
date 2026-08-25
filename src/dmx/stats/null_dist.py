"""Create, estimate, and sample from a null distribution.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory, NullAccumulator,
NullEstimator, and the NullDataEncoder classes for use with dmx-learn.

The NullDistribution object and its related classes are space filling objects meant for
consistency in type hints.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    The sampler generates None for any size input.
    Sequence encodings return None for any input.

"""

from typing import Any, Dict, Optional

import numpy as np

import dmx.utils.vector as vec
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class NullDistribution(SequenceEncodableProbabilityDistribution):
    """Provide a neutral placeholder for an absent distribution component.

    The scalar density is the constant one on every supplied value and the log
    density is zero. This is a neutral likelihood factor, not a normalized
    distribution on an arbitrary non-singleton support. Sampling always returns
    ``None``. Encoding discards all observations, so vectorized scoring returns the
    implementation's fixed one-element zero array rather than preserving batch size.

    Attributes:
        name (Optional[str]): Name for object.

    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize a named null placeholder.

        Args:
            name (Optional[str]): Name for object.

        """
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        """Return an evaluable representation of the placeholder."""
        return f"NullDistribution(name={repr(self.name)})"

    def density(self, x: Optional[Any]) -> float:
        """Density for NullDistribution.

        Args:
            x (Optional[Any]): Can pass any value.

        Returns:
            float: Always evaluates to 1.0.

        """
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        """Log-density for NullDistribution.

        Args:
            x (Optional[Any]): Can pass any value.

        Returns:
            float: Always evaluates to 0.0.

        """
        return 0.0

    def seq_log_density(self, x: "NullEncodedDataSequence") -> np.ndarray:
        """Return a one-element zero vector regardless of encoded input."""
        return vec.zeros(1)

    def sampler(self, seed: Optional[int] = None) -> "NullSampler":
        """Create a sampler that always returns ``None``."""
        return NullSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "NullEstimator":
        """Create an estimator whose result is another null distribution."""
        if pseudo_count is None:
            return NullEstimator(name=self.name)

        return NullEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> "NullDataEncoder":
        """Return an encoder that discards its input."""
        return NullDataEncoder()


class NullSampler(DistributionSampler):
    """NullSampler object, always generates None as sample type.

    Note:
        This generally serves as a place-holder for consistency with other classes. Try
        to remove it before sampling.

    Attributes:
        rng (RandomState): For consistency with other samplers.
        dist (NullDistribution): For consistency with other samplers.

    """

    def __init__(self, dist: "NullDistribution", seed: Optional[int] = None) -> None:
        """Initialize a sampler for the null placeholder.

        Args:
            seed (Optional[int]): For consistency with other samplers.
            dist (NullDistribution): For consistency with other samplers.

        """
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> None:
        """Generate samples from NullDistribution.

        Notes:
            Always returns None regardless of size.

        Args:
            size (Optional[int]): For consistency, does not control number of samples.

        Returns:
            None
        """
        return None


class NullAccumulator(SequenceEncodableStatisticAccumulator):
    """Implement the accumulator protocol with an invariant ``None`` statistic.

    Notes:
        All functions do nothing. They are kept for consistency with other classes to
        ensure type checks.

    Attributes:
        keys (Optional[str]): Set key for distribution.


    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """Initialize a null accumulator.

        Args:
            keys (Optional[str]): Set key for distribution.


        """
        self.key = keys

    def update(
        self, x: Optional[Any], weight: float, estimate: Optional["NullDistribution"]
    ) -> None:
        """Ignore one observation, its weight, and its estimate."""
        pass

    def seq_update(
        self,
        x: "NullEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["NullDistribution"],
    ) -> None:
        """Ignore an encoded sequence, weights, and estimate."""
        pass

    def initialize(
        self, x: Optional[Any], weight: float, rng: Optional["np.random.RandomState"]
    ) -> None:
        """Ignore one initialization observation."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: "NullEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Ignore an encoded initialization sequence."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Optional[Any]) -> "NullAccumulator":
        """Ignore another statistic and return this accumulator."""
        return self

    def value(self) -> None:
        """Return the invariant ``None`` sufficient statistic."""
        return None

    def from_value(self, x: Optional[Any]) -> "NullAccumulator":
        """Ignore a supplied value and return this accumulator."""
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Register ``None`` under the configured key if it is absent."""
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Leave this stateless accumulator unchanged."""
        pass

    def acc_to_encoder(self) -> "NullDataEncoder":
        """Return an encoder that discards every observation."""
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory):
    """NullAccumulatorFactory object for creating NullAccumulator objects.

    Notes:
        All functions do nothing. They are kept for consistency with other classes to
        ensure type checks.

    Attributes:
        keys (Optional[str]): Set key for distribution.


    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """Initialize a factory for null accumulators.

        Args:
            keys (Optional[str]): Set key for distribution.

        """
        self.keys = keys

    def make(self) -> "NullAccumulator":
        """Create a null accumulator with the configured key."""
        return NullAccumulator(keys=self.keys)


class NullEstimator(ParameterEstimator):
    """Produce a null distribution without using any observations.

    Notes:
        Always estimates to same NullDistribution object. This is simply a placeholder.

    Attributes:
        pseudo_count (Optional[float]): Regularize sufficient statistics (ignored).
        suff_stat (Optional[Any]): Can pass anything, is simply ignored.
        keys (Optional[str]): Key for distribution (not meaningful as all estimates are
            NullDistribution())
        name (Optional[str]): Name for estimator.


    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Any] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a stateless null estimator.

        Args:
            pseudo_count (Optional[float]): Regularize sufficient statistics (ignored).
            suff_stat (Optional[Any]): Can pass anything, is simply ignored.
            keys (Optional[str]): Key for distribution (not meaningful as all estimates
                are NullDistribution())
            name (Optional[str]): Name for estimator.


        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("NullEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "NullAccumulatorFactory":
        """Create a factory for invariant null sufficient statistics."""
        return NullAccumulatorFactory(self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Optional[Any] = None
    ) -> "NullDistribution":
        """Return a named null distribution, ignoring all supplied statistics."""
        return NullDistribution(name=self.name)


class NullDataEncoder(DataSequenceEncoder):
    """NullDataEncoder object for consistency with DataSequenceEncoders.

    Notes:
        This enables consistency in type-hints and type-checks for other encodings.


    """

    def __str__(self) -> str:
        """Return the encoder name."""
        return "NullDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether the other object is also a null encoder."""
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Any) -> "NullEncodedDataSequence":
        """Discard the input sequence and return a null encoding."""
        return NullEncodedDataSequence(data=None)


class NullEncodedDataSequence(EncodedDataSequence):
    """NullEncodedDataSequence object for vectorized calls.

    Notes:
        This enables consistency in type-hints and type-checks for other encodings.

    Attributes:
        data (None): None is passed as placeholder.

    """

    def __init__(self, data: None) -> None:
        """Initialize an encoded placeholder containing ``None``.

        Args:
            data (None): None is passed as placeholder.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return the historical representation of the null encoding."""
        return "NullEncodedDataSequence(data=None}"
