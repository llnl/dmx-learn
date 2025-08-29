from typing import Optional
from dmx.arithmetic import *
from dmx.bstats.pdist import StatisticAccumulator, ParameterEstimator, ProbabilityDistribution
from numpy.random import RandomState
from dmx.bstats.nulldist import NullDistribution
import numpy as np

null_dist = NullDistribution()

class IgnoredDistribution(ProbabilityDistribution):

	def __init__(self, dist : ProbabilityDistribution = null_dist):
		self.dist = dist

	def __str__(self):
		return 'IgnoredDistribution(%s)'%(str(self.dist))

	def get_prior(self):
		return self.dist.get_prior()

	def set_prior(self, dist):
		self.dist.set_prior(dist)

	def set_parameters(self, params):
		self.dist.set_parameters(params)

	def get_parameters(self):
		return self.dist.get_parameters()

	def cross_entropy(self, dist):
		return self.dist.cross_entropy(dist)

	def entropy(self):
		return self.dist.entropy()

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):
		return self.dist.log_density(x)

	def expected_log_density(self, x):
		return self.dist.expected_log_density(x)

	def seq_log_density(self, x):
		return self.dist.seq_log_density(x)

	def seq_expected_log_density(self, x):
		return self.dist.seq_expected_log_density(x)

	def seq_encode(self, x):
		return self.dist.seq_encode(x)

	def sampler(self, seed=None):
		return IgnoredSampler(self, seed)

	def estimator(self):
		return IgnoredEstimator(dist=self.dist)


class IgnoredSampler(object):

	def __init__(self, dist, seed=None):
		self.dist_sampler = dist.dist.sampler(seed)

	def sample(self, size=None):
		return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(StatisticAccumulator):

	def __init__(self):
		pass

	def update(self, x, weight, estimate):
		pass

	def seq_update(self, x, weights, estimate):
		pass

	def seq_initialize(self, x, weights, rng):
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


class IgnoredEstimator(ParameterEstimator):

	def __init__(self, dist: ProbabilityDistribution = null_dist, prior: ProbabilityDistribution = null_dist, keys=None):

		self.dist   = dist
		self.prior  = prior
		self.keys   = keys

	def accumulator_factory(self):
		obj = type('', (object,), {'make': lambda o: IgnoredAccumulator()})()
		return(obj)

	def get_prior(self):
		return self.dist.get_prior()

	def set_prior(self):
		self.dist.set_prior()

	def estimate(self, suff_stat):
		return IgnoredDistribution(self.dist)
