"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the dmx.stats.

"""
import math
import numpy as np
from abc import abstractmethod
from dmx.arithmetic import *
from typing import TypeVar, Optional, Any, Generic, Dict

SS = TypeVar('SS')

def equal_object(x: Any, other: Any) -> bool:
    """Lazy object comparison."""
    if isinstance(other, type(x)):
        other_vars = vars(other)
        self_vars = vars(x)

        for k, v in self_vars.items():
            if isinstance(other_vars[k], float) and np.isnan(other_vars[k]):
                if isinstance(v, float) and np.isnan(v):
                    continue
                else:
                    return False
            if not np.all(other_vars[k] == v):
                return False
            
        return True
    
    else:
        return False


class ProbabilityDistribution:
    """Defines ProbabilityDistribution Abstract Class. 

    Note:
        This is generally used as an inherited class for
        SequenceEncodableProbabilityDistribution.

    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def density(self, x: Any) -> float:
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Evaluate the log-density of distribution.

        Returns:
            float

        """
        ...

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler':
        """Create a DistributionSampler object for a given ProbabilityDistribution.

        Args:
            seed (Optional[int]): Set seed for drawing samples from distribution.

        """
        ...

    @abstractmethod
    def estimator(self, pseudo_count: Optional[float] = None) -> 'ParameterEstimator':
        """Create a ParameterEstimator for corresponding SequenceEncodableProbabilityDistribution.

        Args:
            pseudo_count (Optional[float]): Regularize sufficient statistics in estimation step.

        Returns:
            ParameterEstimator

        """
        ...
    
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
    def seq_log_density(self, x: 'EncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of the log density.

        Args:
            x (EncodedDataSequence): EncodedDataSequence for corresponding SequenceEncodedProbabilityDistribution.

        Returns:
            np.ndarray

        """
        ...

    @abstractmethod
    def dist_to_encoder(self) -> 'DataSequenceEncoder':
        """Create DataSequenceEncoder object for SequenceEncodableProbabilityDistribution instance.

        Returns:
            DataSequenceEncoder

        """
        ...

    def seq_log_density_lambda(self):
        return [self.seq_log_density]

    def seq_ld_lambda(self):
        pass


class DistributionSampler(object):
    """DistributionSampler is an Abstract class for distribution samplers.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution to sample from.
        rng (RandomState): Random number generator.

    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, seed: Optional[int] = None) -> None:
        """Initialize DistributionSampler.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution to sample from.
            seed (Optional[int]): Used to set seed on rng.

        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def new_seed(self) -> int:
        """Generates a new seed from rng"""
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


class ConditionalSampler(object):
    """AbstractClass for ConditionalSampler.

    Note:
        This is only implemented for samples of conditional distributions.

    """

    @abstractmethod
    def sample_given(self, x: Any):
        """Sample at conditional value.
        
        Args:
            x (Any): Conditioned on x, sample from dist.

        Returns:
            Sample from conditional distribution.

        """

class StatisticAccumulator(Generic[SS]):

    def __eq__(self, other: Any) -> bool:
        """Tests if a ProbabilityDistribution is equivilent to another.
        
        Args:
            other (Any): Object to test against.

        Returns:
            True if the objects match. 

        """
        return equal_object(self, other)

    def update(self, x: Any, weight: float, estimate: Optional[SequenceEncodableProbabilityDistribution]) -> None:
        """Accumulate sufficient statistics for a single data observation.

        Note:
            Used for debugging only.

        Args:
            x (Any): Data type corresponding to StatisticAccumulator object.
            weight (float): Weight associated with single observation.
            estimate (SequenceEncodableProbabilityDistribution): Previous estimate of distribution.

        """
        ...

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics for a single data observation.

        Note:
            Used for debugging only.

        Args:
            x (Any): Data type corresponding to StatisticAccumulator object.
            weight (float): Weight associated with single observation.
            rng (np.random.RandomState): Set seed for initialization.

        """
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> 'StatisticAccumulator':
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
    def from_value(self, x: SS) -> 'SequenceEncodableStatisticAccumulator':
        """Set sufficient statistics equal to passed value.

        Args:
            x (SS): Generic sufficient statistic for instance of StatisticAccumulator.

        """
        ...

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistic value or accumulator.

        """
        ...

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of accumulator instance to key'd values.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistic value or accumulator.

        """
        ...


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):

    def get_seq_lambda(self):
        pass

    @abstractmethod
    def seq_update(self, x: 'EncodedDataSequence', weights: np.ndarray, estimate: Optional[SequenceEncodableProbabilityDistribution]) -> None:
        """Vectorized accumulation of sufficient statistics for EM updates.

        Args:
            x (EncodedDataSequence): EncodedDataSequence for given SequenceEncodableStatisticAccumulator type.
            weights (np.ndarray): weights for observations.
            estimate (Optional[SequenceEncodableProbabilityDistribution]): Optional previous estimate of distribution.

        """
        ...

    @abstractmethod
    def seq_initialize(self, x: 'EncodedDataSequence', weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics.

        Args:
            x (EncodedDataSequence): EncodedDataSequence for given SequenceEncodableStatisticAccumulator type.
            weights (np.ndarray): weights for observations.
            rng (np.random.RandomState): RandomState object for setting seed on initialization.

        """
        ...

    @abstractmethod
    def acc_to_encoder(self) -> 'DataSequenceEncoder':
        """Create DataSequenceEncoder object for SequenceEncodableStatisticAccumulator instance."""
        ...


class StatisticAccumulatorFactory(object):
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
    def make(self) -> 'SequenceEncodableStatisticAccumulator':
        """Create SequenceEncodableStatisticAccumulator object. """
        ...


class ParameterEstimator(Generic[SS]):
    """Abstract class for ParameterEstimator object. """

    @abstractmethod
    def __init__(self, *args):
        """Must implement constructor for ParameterEstimator"""
        ...

    @abstractmethod
    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'SequenceEncodableProbabilityDistribution':
        """Estimate SequenceEncodableProbabilityDistribution for sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sufficient statistics for dirichlet distribution.

        Returns:
            SequenceEncodableProbabilityDistribution

        """
        ...

    @abstractmethod
    def accumulator_factory(self) -> 'StatisticAccumulatorFactory':
        """Create SequenceEncodableStatisticAccumulator object. """
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

    def __str__(self) -> str:
        return self.__str__()

    @abstractmethod
    def seq_encode(self, x: Any) -> 'EncodedDataSequence':
        """Create EncodedDataSequence from iid observations from SequenceEncodedProbabilityDistribution.

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
class EncodedDataSequence(object):
    """EncodedDatSequence is the outputed data structure from
    DataSeqeunceEncoder. Object is used for vectorized functions and type
    checks.
    """

    def __init__(self, data: Any) -> None:
        """Create instance of EncodedDataSequence.

        Args:
            data (Any): Store the data encocded for vectorized calls.

        """
        self.data = data

    @abstractmethod
    def __repr__(self) -> str: ...









