from typing import Optional, Any, TypeVar, Sequence
from dmx.arithmetic import *
from dmx.stats.pdist import (
        SequenceEncodableProbabilityDistribution,
        SequenceEncodableStatisticAccumulator,
        ParameterEstimator,
        ProbabilityDistribution,
        EncodedDataSequence,
        DataSequenceEncoder)
from numpy.random import RandomState
from dmx.bstats.nulldist import NullDistribution
import numpy as np

T = TypeVar('T')
null_dist = NullDistribution()

class DiracDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return 'DiracDistribution(%s)'%(str(self.value))

    def get_prior(self):
        return self.dist.get_prior()

    def get_parameters(self):
        return self.value

    def density(self, x):
        return np.exp(self.log_density(x))

    def log_density(self, x):
        return self.dist.log_density(x)

    def seq_log_density(self, x):
        return self.dist.seq_log_density(x)

    def seq_encode(self, x):
        return np.asarray(x, dtype=object)

    def sampler(self, seed=None):
        return DiracSampler(self, seed)

    def estimator(self):
        return DiracEstimator()

    def dist_to_encoder(self) -> 'DiracDataEncoder':
        return DiracDataEncoder()


class DiracSampler(object):

    def __init__(self, dist, seed=None):
        self.dist_sampler = dist.dist.sampler(seed)

    def sample(self, size=None):
        return self.dist_sampler.sample(size=size)


class DiracAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self):
        pass

    def update(self, x, weight, estimate):
        pass

    def seq_update(self, x, weights, estimate):
        pass

    def initialize(self, x, weight, rng):
        pass

    def combine(self, suff_stat):
        return self

    def value(self):
        return None

    def from_value(self, x):
        return self

    def key_merge(self, stats_dict):
        pass

    def key_replace(self, stats_dict):
        pass

    def acc_to_encoder(self) -> 'DiracDataEncoder':
        return DiracDataEncoder()

class DiracEstimator(ParameterEstimator):

    def __init__(self, value, prior: ProbabilityDistribution = null_dist, keys=None):

        self.value  = value
        self.prior  = prior
        self.keys   = keys

    def accumulator_factory(self):
        obj = type('', (object,), {'make': lambda o: DiracAccumulator()})()
        return(obj)

    def get_prior(self):
        return self.dist.get_prior()

    def set_prior(self):
        self.dist.set_prior()

    def estimate(self, suff_stat):
        return DiracDistribution(self.value)

class DiracDataEncoder(DataSequenceEncoder):
    
    def __str__(self) -> str:
        return 'DiracDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiracDataEncoder)

    def seq_encode(self, x: Sequence[T]) -> 'DiracEncodedData':
        return DiracEncodedData(data=np.asarray(x, dtype=object))


class DiracEncodedData(EncodedDataSequence):
    def __init__(self, data: np.ndarray):
        super().__init__(data)

    def __repr__(self) -> str:
        return f'DiracEncodedData(data={self.data})'


