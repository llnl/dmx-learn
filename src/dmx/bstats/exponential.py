from typing import Dict, Tuple, Sequence, Optional

from dmx.arithmetic import *
from dmx.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
import numpy as np
from scipy.special import gammaln, digamma

from dmx.bstats.gamma import GammaDistribution

default_prior = GammaDistribution(1.0001, 1.0e6)


class ExponentialDistribution(ProbabilityDistribution):

	def __init__(self, lam: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

		self.name  = name
		self.set_parameters(lam)
		self.set_prior(prior)

	def __str__(self):
		return 'ExponentialDistribution(%f, name=%s, prior=%s)'%(self.lam, self.name, str(self.prior))

	def get_parameters(self) -> float:
		return self.lam

	def set_parameters(self, params: float) -> None:
		self.lam = params
		self.log_lam = np.log(params)
		self.params = params

	def get_prior(self) -> ProbabilityDistribution:
		return self.prior

	def set_prior(self, prior: ProbabilityDistribution):

		self.prior = prior

		if isinstance(prior, GammaDistribution):
			self.conj_prior_params = [prior.k, 1/prior.theta]

			a, b = self.conj_prior_params
			# eta = -lambda
			# E[ eta ] = -k * theta
			# E[ a( eta ) ] = E[ -ln( -eta ) ] = digamma(k) + ln(theta)
			# E[ ln h(x) ] = 0

			e1 = -a / b
			ea = -(digamma(a) - log(b))

			self.expected_nparams = [ea, 0, e1]

		else:
			self.conj_prior_params = None
			self.expected_nparams = None

	def log_density(self, x: float) -> float:
		if x < 0:
			return -inf
		else:
			return -x*self.lam + self.log_lam

	def expected_log_density(self, x):

		ea, eb, e1 = self.expected_nparams

		return e1*x + (eb - ea)

	def seq_log_density(self, x):
		rv = x*(-self.lam)
		rv += self.log_lam
		return rv

	def seq_expected_log_density(self, x):

		ea, eb, e1 = self.expected_nparams

		return e1*x + (eb - ea)

	def seq_encode(self, x):
		rv = np.asarray(x)
		return rv

	def value(self):
		return [self.lam]

	def sampler(self, seed=None):
		return ExponentialSampler(self, seed)

	def estimator(self):
		return ExponentialEstimator(prior=self.prior)


class ExponentialSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = np.random.RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		return self.rng.exponential(scale=1/self.dist.lam, size=size)


class ExponentialAccumulator(StatisticAccumulator):

	def __init__(self, keys):
		self.sum  = 0.0
		self.count = 0.0
		self.key   = keys[0]

	def update(self, x, weight, estimate):
		if x >= 0:
			self.sum  += x*weight
			self.count += weight

	def seq_update(self, x, weights, estimate):
		self.sum += np.dot(x, weights)
		self.count += np.sum(weights)

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_initialize(self, x, weights, rng):
		self.seq_update(x, weights, None)

	def combine(self, suff_stat):
		self.sum  += suff_stat[1]
		self.count += suff_stat[0]
		return self

	def value(self):
		return self.count, self.sum

	def from_value(self, x):
		self.count = x[0]
		self.sum = x[1]
		return self

	def key_merge(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
			else:
				stats_dict[self.key] = (self.count, self.sum)

	def key_replace(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				self.count = vals[0]
				self.sum   = vals[1]


class ExponentialAccumulatorFactory(object):

	def __init__(self, keys):
		self.keys = keys

	def make(self):
		return ExponentialAccumulator(self.keys)


class ExponentialEstimator(ParameterEstimator):

	def __init__(self, prior=default_prior, name=None,  keys=(None,)):

		self.keys   = keys
		self.name   = name
		self.set_prior(prior)

	def accumulator_factory(self):
		return ExponentialAccumulatorFactory(self.keys)

	def get_prior(self):
		return self.prior

	def set_prior(self, prior):
		self.prior  = prior

		if isinstance(prior, GammaDistribution):
			self.conj_prior_params = [prior.k, 1/prior.theta]
			pass
		elif isinstance(prior, list):
			self.conj_prior_params = prior
		else:
			self.conj_prior_params = None


	def estimate(self, suff_stat):

		if self.conj_prior_params is not None:

			a, b = self.conj_prior_params

			n = suff_stat[0] + a
			s = suff_stat[1] + b

			#conj_prior_params = (n + 1, s)

			return ExponentialDistribution((n - 1) / s, name=self.name, prior=GammaDistribution(n,1/s))


		else:

			n = suff_stat[0]
			s = suff_stat[1]

			return ExponentialDistribution(n / s, name=self.name)
