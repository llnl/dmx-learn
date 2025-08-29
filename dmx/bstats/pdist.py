from typing import Generic, TypeVar, Optional, Any
from collections.abc import Iterable
from abc import abstractmethod
from dmx.arithmetic import exp, one
import numpy as np
import pandas as pd

X = TypeVar('X') # Observation type
E = TypeVar('E')
P = TypeVar('P') # Parameter type
V = TypeVar('V') # Encoding type

noname_instance_count = 0

class ProbabilityDistribution(Generic[X,P,V]):

	def __init__(self):
		return

	def get_parameters(self) -> P:
		return self.params

	def set_parameters(self, value: P) -> None:
		self.params = value

	def get_name(self) -> str:
		return self.name

	def set_name(self, name: Optional[str]) -> None:
		self.name = name

	def add_parent(self, dist) -> None:
		#self.parents.append(dist)
		pass

	def density(self, x: X) -> float:
		return exp(self.log_density(x))

	def log_density(self, x: X) -> float:
		return None

	def expected_log_density(self, x: X) -> float:
		return None

	def seq_log_density(self, x: V) -> np.ndarray:
		return np.asarray([self.log_density(u) for u in x])

	def seq_encode(self, x: Iterable[X]) -> V:
		return x

	def df_log_density(self, df) -> pd.DataFrame:
		return df[self.name].map(self.log_density)

	def sampler(self, seed: Optional[int] = None):
		return None

	def estimator(self) -> Any:
		return None


class ProbabilityDistributionFactory(object):

	def make(self, params) -> ProbabilityDistribution:
		pass




class StatisticAccumulator(object):

	def update(self, x, weight, estimate):
		pass

	def initialize(self, x, weight, rng):
		self.update(x, weight, estimate=None)

	def combine(self, suff_stat):
		pass

	def value(self):
		pass

	def from_value(self, x):
		pass

	def key_merge(self, stats_dict):
		pass

	def key_replace(self, stats_dict):
		pass

class ParameterEstimator(object):

	def estimate(self, suff_stat):
		pass

	def accumulator_factory(self):
		pass


class SequenceEncodableDistribution(ProbabilityDistribution):

	def seq_log_density(self, x):
		return np.asarray([self.log_density(u) for u in x])

	def seq_log_density_lambda(self):
		return [self.seq_log_density]

	def seq_encode(self, x):
		return x

class DataFrameEncodableDistribution(ProbabilityDistribution):

	def get_name(self):
		return self.name

	def set_name(self, name):
		self.name = name

	def df_log_density(self, df):
		return df[self.name].map(self.log_density)

class SequenceEncodableAccumulator(StatisticAccumulator):

	def get_seq_lambda(self):
		pass

	def seq_initialize(self, x, weights, rng):
		pass

	def seq_update(self, x, weights, estimate):
		pass


class DataFrameEncodableAccumulator(StatisticAccumulator):

	def df_initialize(self, df, weights, rng):
		for v,w in zip(df[self.name], weights):
			self.initialize(v,w,rng)

	def df_update(self, df, weights, estimate):
		for v,w in zip(df[self.name], weights):
			self.update(v,w,estimate)


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
        self.data = data

    @abstractmethod
    def __repr__(self) -> str: ...






