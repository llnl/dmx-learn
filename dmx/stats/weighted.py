""""WeightedDistribution class.

This Distribution simply allows from weights on observations. I.e. Data type D is observed and an associated
score/weight is assigned to the data. This simply passes the weights and data downstream in aggregation.

Likelihood evals are equivalent to normal likelihood calls to the base distribution.

"""
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from numpy.random import RandomState
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence, TypeVar

T = TypeVar('T')
SS = TypeVar('SS')


class WeightedDistribution(SequenceEncodableProbabilityDistribution):
    """WeightedDistribution object for creating a distribution that acts on tuples of (value, counts).

    Notes:
        Distribution acts only on the value for likelihood calls and treats weight as number of replicates.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution for values.
        name (Optional[str]): Name for distribution.
        keys (Optional[str]): Keys for parameters of dist. 

    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, name: Optional[str] = None, keys: Optional[str] = None):
        """WeightedDistribution object.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution for values.
            name (Optional[str]): Name for distribution.
            keys (Optional[str]): Keys for parameters of dist. 

        """
        self.dist = dist
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'WeightedDistribution(dist=%s, name=%s, keys=%s)' % (repr(self.dist), repr(self.name), repr(self.keys))

    def density(self, x: Tuple[T, float]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Tuple[T, float]) -> float:
        return self.dist.log_density(x[0])*x[1]

    def seq_log_density(self, x: 'WeightedEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, WeightedEncodedDataSequence):
            raise Exception('WeightedEncodedDataSequence required for seq_log_density().')

        return self.dist.seq_log_density(x.data[0])*x.data[1]

    def dist_to_encoder(self) -> 'WeightedDataEncoder':
        return WeightedDataEncoder(encoder=self.dist.dist_to_encoder())

    def estimator(self, pseudo_count: Optional[float] = None) -> 'WeightedEstimator':
        if pseudo_count is not None:
            return WeightedEstimator(estimator=self.dist.estimator(pseudo_count=pseudo_count), name=self.name, keys=self.keys)
        else:
            return WeightedEstimator(estimator=self.dist.estimator(), name=self.name, keys=self.keys)

    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler':
        return self.dist.sampler(seed)


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):
    """WeightedAccumulator object for accumulating sufficient statistics.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for base distribution.
        keys (Optional[str]): Key for sufficient statistics of base distribution.
        name (Optional[str]): Optional name for distribution.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, keys: Optional[str] = None,
                 name: Optional[str] = None):
        """WeightedAccumulator object.

         Args:
             accumulator (SequenceEncodableStatisticAccumulator): Accumulator for base distribution.
             keys (Optional[str]): Key for sufficient statistics of base distribution.
             name (Optional[str]): Optional name for distribution.

         """
        self.accumulator = accumulator
        self.keys = keys
        self.name = name

    def initialize(self, x: Tuple[T, float], weight: float, rng: np.random.RandomState) -> None:
        self.accumulator.initialize(x[0], weight*x[1], rng)

    def update(self, x: Tuple[T, float], weight: float, estimate: WeightedDistribution) -> None:
        self.accumulator.update(x[0], weight*x[1], estimate.dist)

    def seq_update(self, x: 'WeightedEncodedDataSequence', weights: np.ndarray, estimate: WeightedDistribution) -> None:
        self.accumulator.seq_update(x.data[0], weights*x.data[1], estimate.dist)

    def seq_initialize(self, x: 'WeightedEncodedDataSequence', weights: np.ndarray, rng: np.random.RandomState) -> None:
        self.accumulator.seq_initialize(x.data[0], weights*x.data[1], rng)

    def combine(self, suff_stat: SS) -> 'WeightedAccumulator':
        self.accumulator.combine(SS)

        return self

    def from_value(self, x: SS) -> 'WeightedAccumulator':
        self.accumulator.from_value(x)

        return self

    def value(self) -> Any:
        return self.accumulator.value()

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.accumulator.combine(stats_dict[self.keys].value())
            else:
                stats_dict[self.keys] = self.accumulator

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.accumulator.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> 'WeightedDataEncoder':
        return WeightedDataEncoder(encoder=self.accumulator.acc_to_encoder())

class WeightedAccumulatorFactory(StatisticAccumulatorFactory):
    """WeightedAccumulatorFactory object for creating WeightedAccumulator objects.

    Attributes:
        factory (StatisticAccumulatorFactory): Accumulator for base distribution.
        keys (Optional[str]): Optional keys for base distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(self, factory: StatisticAccumulatorFactory, keys: Optional[str] = None, name: Optional[str] = None):
        """WeightedAccumulatorFactory object for creating WeightedAccumulator objects.

        Args:
            factory (StatisticAccumulatorFactory): Accumulator for base distribution.
            keys (Optional[str]): Optional keys for base distribution.
            name (Optional[str]): Name for object.

        """
        self.factory = factory
        self.keys = keys
        self.name = name

    def make(self) -> 'WeightedAccumulator':
        return WeightedAccumulator(accumulator=self.factory.make(), name=self.name, keys=self.keys)


class WeightedEstimator(ParameterEstimator):
    """WeightedEstimator object for estimating WeightedDistribution.

    Attributes:
        estimator (ParameterEstimator): Estimator for the base distribution.
        keys (Optional[str]): Keys for the base distribution.
        name (Optional[str]): Optional name for object.

    """

    def __init__(self, estimator: ParameterEstimator, keys: Optional[str] = None, name: Optional[str] = None):
        """WeightedEstimator object.

        Args:
            estimator (ParameterEstimator): Estimator for the base distribution.
            keys (Optional[str]): Keys for the base distribution.
            name (Optional[str]): Optional name for object.

        """
        self.estimator = estimator
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'WeightedAccumulatorFactory':
        return WeightedAccumulatorFactory(factory=self.estimator.accumulator_factory(), keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'WeightedDistribution':
        return WeightedDistribution(dist=self.estimator.estimate(nobs, suff_stat), name=self.name)


class WeightedDataEncoder(DataSequenceEncoder):
    """WeightedDataEncoder object for encoding iid sequences of WeightedDistribution.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the base distribution.

    """

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """WeightedDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder for the base distribution.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        return 'WeightedDataEncoder(encoder=%s)' % (repr(self.encoder))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WeightedDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[T, float]]) -> 'WeightedEncodedDataSequence':
        rv_enc = self.encoder.seq_encode([xx[0] for xx in x]), np.asarray([xx[1] for xx in x], dtype=float)

        return WeightedEncodedDataSequence(data=rv_enc)

class WeightedEncodedDataSequence(EncodedDataSequence):
    """WeightedEncodedDataSequence object for vectorized calls.

    Attributes:
        data (Tuple[EncodedDataSequence, np.ndarray]): EncodedDataSequence for base distribution and array of counts.

    """

    def __init__(self, data: Tuple[EncodedDataSequence, np.ndarray]):
        """WeightedEncodedDataSequence object.

        Args:
            data (Tuple[EncodedDataSequence, np.ndarray]): EncodedDataSequence for base distribution and array of counts.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'WeightedEncodedDataSequence(data={self.data})'



