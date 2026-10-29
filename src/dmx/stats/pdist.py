"""Abstract classes for probability distributions and statistical accumulators.

This module provides the foundational abstract base classes for the
`dmx.stats` package, including probability distributions, statistic
accumulators, samplers, and encoders.

Classes:
    ProbabilityDistribution: Abstract base class for probability distributions.
    SequenceEncodableProbabilityDistribution: Probability distribution with
        sequence encoding.
    DistributionSampler: Abstract sampler for probability distributions.
    ConditionalSampler: Abstract sampler for conditional distributions.
    StatisticAccumulator: Abstract accumulator for sufficient statistics.
    SequenceEncodableStatisticAccumulator: Statistic accumulator with sequence
        encoding.
    StatisticAccumulatorFactory: Factory for creating statistic accumulators.
    ParameterEstimator: Abstract estimator for distribution parameters.
    DataSequenceEncoder: Abstract encoder for data sequences.
    EncodedDataSequence: Container for encoded sequence data.
"""

# pylint: disable=unnecessary-ellipsis
# Rationale: Ellipsis (...) is the standard Python idiom for abstract method stubs
# and is preferred over 'pass' as it more clearly indicates that the method must
# be implemented by subclasses.

import math
from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Set, TypeVar

import numpy as np

from dmx.arithmetic import maxrandint
from dmx.utils.fisher import (
    FisherScoreArray,
    FisherScoreOptions,
    fisher_score_unsupported,
)
from dmx.utils.seeding import resolve_sampler_seed

SS = TypeVar("SS")


def equal_object(x: Any, other: Any) -> bool:
    """Lazy object comparison."""
    if not isinstance(other, type(x)):
        return False

    other_vars = vars(other)
    self_vars = vars(x)

    for k, v in self_vars.items():
        if isinstance(other_vars[k], float) and np.isnan(other_vars[k]):
            if isinstance(v, float) and np.isnan(v):
                continue
            return False
        if not np.all(other_vars[k] == v):
            return False

    return True


class ProbabilityDistribution:
    """Defines ProbabilityDistribution Abstract Class.

    Note:
        This is generally used as an inherited class for
        SequenceEncodableProbabilityDistribution.

    """

    def __init__(self) -> None:
        """Initialize the probability distribution."""
        ...

    def __repr__(self) -> str:
        """Return string representation of the distribution."""
        return self.__str__()

    @abstractmethod
    def density(self, x: Any) -> float:
        """Compute the probability density at x.

        Args:
            x: Input value to evaluate density.

        Returns:
            Probability density value.

        """
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Evaluate the log-density of distribution.

        Returns:
            float

        """
        ...

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> "DistributionSampler":
        """Create a sampler for a probability distribution.

        Args:
            seed (Optional[int]): Set seed for drawing samples from distribution.

        """
        ...

    @abstractmethod
    def estimator(self, pseudo_count: Optional[float] = None) -> "ParameterEstimator":
        """Create a parameter estimator for this distribution.

        Args:
            pseudo_count (Optional[float]): Regularize sufficient statistics in
                the estimation step.

        Returns:
            ParameterEstimator

        """
        ...

    def num_params(self) -> int:
        """Return the number of free parameters in the distribution.

        The count is the dimension of the parameter space searched by the
        estimator, as used by BIC, AIC, and ICL. A probability vector over ``K``
        values contributes ``K - 1``. Fixed structure such as support bounds,
        dimensions, and sequence lengths contributes nothing. Combinators sum the
        counts of their children recursively.

        Parameters tied through estimator ``keys`` are counted once. A key that
        appears more than once in the tree contributes only at its first
        occurrence, and the later occurrences add nothing for that parameter
        block.

        Returns:
            int: Number of free parameters.

        """
        return self._num_params(set())

    def _num_params(self, seen_keys: Set[str]) -> int:
        """Count free parameters, skipping blocks whose key is in ``seen_keys``.

        Subclasses implement this hook. Implementations add every key they
        count to ``seen_keys`` (see ``_key_is_new``) and count child
        distributions with ``_child_num_params`` using the same set.

        Args:
            seen_keys (Set[str]): Keys whose parameters were already counted.

        Returns:
            int: Number of free parameters not already counted.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement num_params()."
        )

    @staticmethod
    def _child_num_params(child: "ProbabilityDistribution", seen_keys: Set[str]) -> int:
        """Count the free parameters of a child distribution.

        Args:
            child (ProbabilityDistribution): Child distribution to count.
            seen_keys (Set[str]): Keys whose parameters were already counted.

        Returns:
            int: Number of free parameters in ``child`` not already counted.

        """
        # pylint: disable-next=protected-access
        return child._num_params(seen_keys)

    @staticmethod
    def _key_is_new(key: Optional[str], seen_keys: Set[str]) -> bool:
        """Return True if parameters under ``key`` still need to be counted.

        Unkeyed parameters (``key is None``) are always counted. A new key is
        recorded in ``seen_keys`` so later occurrences are skipped.

        Args:
            key (Optional[str]): Key tying a parameter block, or None.
            seen_keys (Set[str]): Keys whose parameters were already counted.

        Returns:
            bool: True if the parameter block should be counted.

        """
        if key is None:
            return True
        if key in seen_keys:
            return False
        seen_keys.add(key)
        return True

    def seq_fisher_score(
        self, x: Any, options: Optional[FisherScoreOptions] = None
    ) -> FisherScoreArray:
        """Return the whitened Fisher score of each encoded observation.

        The score is ``I^{-1/2} (T(x) - E[T])`` for sufficient statistics
        ``T`` and Fisher information ``I``, so it has mean zero and identity
        covariance under this distribution. Combinators concatenate or
        reweight their children's scores using the same ``options``. See
        :mod:`dmx.utils.fisher`.

        Args:
            x: Encoded sequence of ``N`` observations from
                ``dist_to_encoder().seq_encode``.
            options: Score options; None uses the defaults.

        Returns:
            FisherScoreArray: Float64 array, or CSR matrix for high-cardinality
            encodings, of shape ``(N, fisher_dim(options))``.

        Raises:
            NotImplementedError: If this distribution has no Fisher score.

        """
        raise fisher_score_unsupported(self, "seq_fisher_score")

    def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int:
        """Return the number of columns returned by ``seq_fisher_score``.

        Combinators sum their children's dimensions, so a successful call
        means the whole distribution tree supports Fisher scores.

        Args:
            options: Score options; some change the dimension.

        Returns:
            int: Score dimension.

        Raises:
            NotImplementedError: If this distribution has no Fisher score.

        """
        raise fisher_score_unsupported(self, "fisher_dim")

    def fisher_field_slices(
        self, options: Optional[FisherScoreOptions] = None
    ) -> Dict[str, slice]:
        """Map field names to their columns in ``seq_fisher_score``.

        The default covers the whole score with one entry named after the
        distribution's ``name``, or its class name when unnamed. Combinators
        return one entry per field.

        Args:
            options: Score options passed to ``fisher_dim``.

        Returns:
            Dict[str, slice]: Field name to column slice.

        Raises:
            NotImplementedError: If this distribution has no Fisher score.

        """
        name = getattr(self, "name", None) or type(self).__name__
        return {name: slice(0, self.fisher_dim(options))}

    def __eq__(self, other: Any) -> bool:
        """Tests if a ProbabilityDistribution is equivilent to another.

        Args:
            other (Any): Object to test against.

        Returns:
            True if the objects match.

        """
        return equal_object(self, other)


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):
    """Extends the ProbabilityDistribution to handle vectorized calls."""

    @abstractmethod
    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized evaluation of the log density.

        Args:
            x: Encoded sequence for the corresponding probability distribution.

        Returns:
            np.ndarray

        """
        ...

    @abstractmethod
    def dist_to_encoder(self) -> "DataSequenceEncoder":
        """Create a data sequence encoder for this distribution.

        Returns:
            DataSequenceEncoder

        """
        ...

    def seq_log_density_lambda(self) -> list[Any]:
        """Return a list containing the sequence log density method.

        Returns:
            List with single element: the seq_log_density method.

        """
        return [self.seq_log_density]

    def seq_ld_lambda(self) -> list[Any]:
        """Legacy method stub for compatibility.

        Returns the vectorized log-density callable list used by older helpers.

        """
        return [self.seq_log_density]


class DistributionSampler:
    """DistributionSampler is an Abstract class for distribution samplers.

    Attributes:
        dist: Distribution to sample from.
        seed (Any): Seed ``rng`` was built from, an int unless an array-like
            seed was passed. When the sampler is created with ``seed=None`` a
            fresh seed is drawn from OS entropy, so
            ``dist.sampler(seed=sampler.seed)`` reproduces the samples.
        rng (RandomState): Random number generator.

    """

    def __init__(self, dist: Any, seed: Optional[int] = None) -> None:
        """Initialize DistributionSampler.

        Args:
            dist: Distribution to sample from.
            seed (Optional[int]): Seed for ``rng``. ``None`` draws a fresh
                seed, stored in :attr:`seed`.

        """
        self.dist: Any = dist
        self.seed: Any = resolve_sampler_seed(seed)
        self.rng = np.random.RandomState(self.seed)

    def new_seed(self) -> int:
        """Generate a new random seed from the random number generator.

        Returns:
            A new random seed integer.

        """
        return self.rng.randint(0, maxrandint)

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> Any:
        """Generate samples from distribution.

        Args:
            size (Optional[int]): Number of samples to generate.

        Returns:
            Samples from distribution.

        """
        ...


class ConditionalSampler:
    """AbstractClass for ConditionalSampler.

    Note:
        This is only implemented for samples of conditional distributions.

    """

    @abstractmethod
    def sample_given(self, x: Any) -> Any:
        """Sample at conditional value.

        Args:
            x (Any): Conditioned on x, sample from dist.

        Returns:
            Sample from conditional distribution.

        """


class StatisticAccumulator(Generic[SS]):
    """Abstract base class for sufficient statistic accumulators.

    Accumulators maintain and update sufficient statistics for parameter estimation.

    Type Parameters:
        SS: Type of the sufficient statistics.

    """

    def __eq__(self, other: Any) -> bool:
        """Tests if a ProbabilityDistribution is equivilent to another.

        Args:
            other (Any): Object to test against.

        Returns:
            True if the objects match.

        """
        return equal_object(self, other)

    @abstractmethod
    def update(
        self,
        x: Any,
        weight: float,
        estimate: Any,
    ) -> None:
        """Accumulate sufficient statistics for a single data observation.

        Note:
            Used for debugging only.

        Args:
            x (Any): Data type corresponding to StatisticAccumulator object.
            weight (float): Weight associated with single observation.
            estimate: Previous estimate of distribution.

        """
        ...

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize sufficient statistics for a single data observation.

        Note:
            Used for debugging only.

        Args:
            x (Any): Data type corresponding to StatisticAccumulator object.
            weight (float): Weight associated with single observation.
            rng: Seed for initialization. Unused in the base implementation.

        """
        del rng
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> "StatisticAccumulator":
        """Method for combining aggregated sufficient statistics.

        Args:
            suff_stat (SS): Sufficient statistics.

        Returns:
            None


        """
        ...

    @abstractmethod
    def value(self) -> SS:
        """Return sufficient statistics of StatisticAccumulator."""
        ...

    @abstractmethod
    def from_value(self, x: SS) -> "SequenceEncodableStatisticAccumulator":
        """Set sufficient statistics equal to passed value.

        Args:
            x (SS): Generic sufficient statistic for instance of StatisticAccumulator.

        """
        ...

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient
                statistic values or accumulators.

        """
        ...

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of accumulator instance to key'd values.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient
                statistic values or accumulators.

        """
        ...


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):
    """Statistic accumulator with support for sequence-based updates.

    Extends StatisticAccumulator to handle vectorized updates over sequences
    of encoded data.

    Type Parameters:
        SS: Type of the sufficient statistics.

    """

    def get_seq_lambda(self) -> list[Any]:
        """Legacy method stub for compatibility.

        Returns the vectorized update callable list used by older helpers.

        """
        return [self.seq_update]

    @abstractmethod
    def seq_update(
        self,
        x: Any,
        weights: np.ndarray,
        estimate: Any,
    ) -> None:
        """Vectorized accumulation of sufficient statistics for EM updates.

        Args:
            x: Encoded sequence for this accumulator type.
            weights (np.ndarray): weights for observations.
            estimate: Optional previous estimate of distribution.

        """
        ...

    @abstractmethod
    def seq_initialize(self, x: Any, weights: np.ndarray, rng: Any) -> None:
        """Vectorized initialization of sufficient statistics.

        Args:
            x: Encoded sequence for this accumulator type.
            weights (np.ndarray): weights for observations.
            rng: RandomState used to set the initialization seed.

        """
        ...

    @abstractmethod
    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Create a data sequence encoder for this accumulator."""
        ...


class StatisticAccumulatorFactory:
    """Factory for creating SequenceEncodableStatsiticAccumulator objects."""

    def __eq__(self, other: Any) -> bool:
        """Tests if a ProbabilityDistribution is equivilent to another.

        Args:
            other (Any): Object to test against.

        Returns:
            True if the objects match.

        """
        return equal_object(self, other)

    @abstractmethod
    def make(self) -> "SequenceEncodableStatisticAccumulator":
        """Create SequenceEncodableStatisticAccumulator object."""
        ...


class ParameterEstimator(Generic[SS]):
    """Abstract class for ParameterEstimator object."""

    @abstractmethod
    def __init__(self, *args: Any) -> None:
        """Initialize the ParameterEstimator.

        Args:
            *args: Variable length argument list for initialization.

        """
        ...

    @abstractmethod
    def estimate(
        self, nobs: Optional[float], suff_stat: SS
    ) -> "SequenceEncodableProbabilityDistribution":
        """Estimate a probability distribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
                Sufficient statistics for a Dirichlet distribution.

        Returns:
            SequenceEncodableProbabilityDistribution

        """
        ...

    @abstractmethod
    def accumulator_factory(self) -> "StatisticAccumulatorFactory":
        """Create SequenceEncodableStatisticAccumulator object."""
        ...

    def __eq__(self, other: Any) -> bool:
        """Tests if a ParameterEstimator is equivilent to another.

        Args:
            other (Any): Object to test against.

        Returns:
            True if the objects match.

        """
        return equal_object(self, other)


class DataSequenceEncoder:
    """Abstract base class for encoding data sequences.

    Encoders transform raw data sequences into encoded representations
    suitable for probability distribution operations.

    """

    def __str__(self) -> str:
        """Return string representation of the encoder."""
        return self.__str__()

    @abstractmethod
    def seq_encode(self, x: Any) -> "EncodedDataSequence":
        """Create an encoded sequence from IID observations.

        Args:
            x (Any): Sequence of observations from corresponding distribution.

        Returns:
            EncodedDataSequence

        """
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if object is an instance of DataSequenceEncoder.

        Used to avoid repeated sequence encodings when appropriate.

        Args:
            other (object): Object to compare.

        Returns:
            True if object is an instance of ExponentialDataEncoder, else False.

        """
        ...


class EncodedDataSequence:
    """Container for encoded data sequences.

    EncodedDataSequence is the output data structure from DataSequenceEncoder.
    This object is used for vectorized functions and type checks.

    Attributes:
        data: The encoded data for vectorized calls.

    """

    def __init__(self, data: Any) -> None:
        """Create instance of EncodedDataSequence.

        Args:
            data: Store the data encoded for vectorized calls.

        """
        self.data = data

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the encoded sequence."""
        ...
