from typing import Optional, Any
from dmx.bstats.pdist import StatisticAccumulator, ParameterEstimator, ProbabilityDistribution
from dmx.bstats.pdist import EncodedDataSequence, DataSequenceEncoder
import numpy as np


class NullDistribution(ProbabilityDistribution[Any, None, None]):

    def __init__(self):
        self.parents = []
        pass

    def __str__(self):
        return 'NullDistribution()'

    def get_prior(self):
        return self

    def set_prior(self, prior):
        pass

    def get_parameters(self):
        return None

    def set_parameters(self, params):
        pass

    def moments(self, p, o):
        return 1.0

    def cross_entropy(self, dist):
        return 0.0

    def entropy(self):
        return 0.0

    def density(self, x):
        return 1.0

    def log_density(self, x):
        return 0.0

    def seq_log_density(self, x):
        return np.zeros(len(x))

    def seq_encode(self, x):
        return x

    def expected_log_density(self, x):
        return 0.0

    def seq_expected_log_density(self, x):
        return np.zeros(len(x))

    def sampler(self, seed=None):
        return NullSampler(self, seed)

    def estimator(self):
        return NullEstimator()

    def dist_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


null_dist = NullDistribution()


class NullSampler(object):

    def __init__(self, seed=None):
        pass

    def sample(self, size=None):
        if size is None:
            return None
        else:
            return [None] * size


class NullAccumulator(StatisticAccumulator):

    def __init__(self):
        pass

    def update(self, x, weight, estimate):
        pass

    def seq_update(self, x, weights, estimate):
        pass

    def initialize(self, x, weight, rng):
        pass

    def seq_initialize(self, x, weights, rng):
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

    def acc_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


class NullEstimator(ParameterEstimator):

    def __init__(self, prior=None, keys=None):
        pass

    def accumulator_factory(self):
        obj = type('', (object,), {'make': lambda o: NullAccumulator()})()
        return obj

    def get_prior(self):
        return null_dist

    def set_prior(self, prior):
        pass

    def estimate(self, suff_stat):
        return null_dist


class NullDataEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return 'NullDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Any) -> 'NullEncodedData':
        return NullEncodedData(data=None)


class NullEncodedData(EncodedDataSequence):
    def __init__(self, data: None):
        self.data = None

    def __repr__(self) -> str:
        return 'NullEncodedData'


null_estimator = NullEstimator()
