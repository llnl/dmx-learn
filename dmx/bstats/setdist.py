from dmx.arithmetic import *
from numpy.random import RandomState
from collections import defaultdict
from dmx.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, ParameterEstimator
import numpy as np
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.mixture import MixtureDistribution
from dmx.utils.special import gammaln


default_prior = BetaDistribution(1,1)

class BernoulliSetDistribution(ProbabilityDistribution):

	def __init__(self, pmap, name=None, prior=None):

		self.name = name
		self.prior = prior
		self.pmap = pmap
		self.log_pmap = {k: np.log1p(v) if v < 0 else np.log(v) for k, v in pmap.items()}
		self.log_nmap = {k: np.log(-v) if v < 0 else np.log1p(-v) for k, v in pmap.items()}
		self.nmap_sum = sum([u for u in self.log_nmap.values() if u != -np.inf])

	def __str__(self):
		return 'BernoulliSetDistribution(%s, name=%s, prior=%s)'%(str(self.pmap), str(self.name), str(self.prior))

	def get_parameters(self):
		return self.pmap

	def set_parameters(self, params):
		self.pmap = params

	def get_prior(self):
		return self.prior

	def set_prior(self, prior):
		self.prior = prior

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):
		rv = self.nmap_sum
		for u in x:
			rv += self.log_pmap[u]-self.log_nmap[u]
		return rv

	def seq_log_density(self, x):

		sz, idx, val_map_inv, xs = x

		log_prob_loc = np.asarray([self.log_pmap[u] - self.log_nmap[u] for u in val_map_inv])

		rv  = np.bincount(idx, weights=log_prob_loc[xs], minlength=sz)
		rv += self.nmap_sum

		return rv


	def seq_encode(self, x):

		idx = []
		xflat = []

		for i in range(len(x)):
			m = len(x[i])
			idx.extend([i]*m)
			xflat.extend(x[i])

		val_map_inv, xs = np.unique(xflat, return_inverse=True)
		idx = np.asarray(idx, dtype=int)

		return len(x), idx, val_map_inv, xs

	def sampler(self, seed=None):
		return BernoulliSetSampler(self, seed)

	def estimator(self):
		return BernoulliSetEstimator()


class BernoulliSetSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):

		if size is not None:
			retval = [[] for i in range(size)]
			for k,v in self.dist.pmap.items():
				for i in np.flatnonzero(self.rng.rand(size) <= (v%1)):
					retval[i].append(k)
			return retval

		else:
			retval = []
			for k,v in self.dist.pmap.items():
				if self.rng.rand() <= (v%1):
					retval.append(k)
			return retval

class BernoulliSetAccumulator(SequenceEncodableAccumulator):

	def __init__(self):
		self.pmap = defaultdict(float)
		self.tot_sum = 0.0

	def update(self, x, weight, estimate):
		for u in x:
			self.pmap[u] += weight
		self.tot_sum += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_update(self, x, weights, estimate):

		sz, idx, val_map_inv, xs = x
		agg_cnt = np.bincount(xs, weights[idx])

		for i,v in enumerate(agg_cnt):
			self.pmap[val_map_inv[i]] += v

		self.tot_sum += weights.sum()

	def combine(self, suff_stat):
		for k,v in suff_stat[0].items():
			self.pmap[k] += v
		self.tot_sum += suff_stat[1]
		return self

	def value(self):
		return self.pmap, self.tot_sum

	def from_value(self, x):
		self.pmap = x[0]
		self.tot_sum = x[1]
		return self

class BernoulliSetEstimator(ParameterEstimator):

	def __init__(self, name=None, prior=default_prior, keys=(None,)):
		self.name = name
		self.prior = prior
		self.keys = keys

	def accumulator_factory(self):
		obj = type('', (object,), {'make': lambda self: BernoulliSetAccumulator()})()
		return obj

	def get_prior(self):
		return self.prior

	def set_prior(self, prior):
		self.prior = prior

	def estimate(self, suff_stat):

		if isinstance(self.prior, BetaDistribution):
			pmap = bernoulli_beta_posterior_mode(suff_stat[0], suff_stat[1], self.prior.get_parameters())

		elif isinstance(self.prior, MixtureDistribution) and all([isinstance(u,BetaDistribution) for u in self.prior.components]):

			beta_params = np.asarray([[u.a, u.b] for u in self.prior.components])
			pmap = bernoulli_betamix_posterior_mode(suff_stat[0], suff_stat[1], self.prior.w, beta_params)

		else:
			pmap = dict()
			for k,v in suff_stat[0].items():
				if v*2 > suff_stat[1]:
					pmap[k] = -(suff_stat[1] - v)/suff_stat[1]
				else:
					pmap[k] = v / suff_stat[1]

		return BernoulliSetDistribution(pmap)


def bernoulli_beta_posterior_mode(obs_cnt, tot_cnt, beta_params):

	pmap = dict()
	for k, v in obs_cnt.items():
		a = (beta_params[0] - 1) + v
		b = (beta_params[1] - 1) - v + tot_cnt

		if a > 0 and b > 0 and a > b:
			p = -b / (a + b)
		elif a > 0 and b > 0 and b > a:
			p = (a - 1) / (a + b - 2)
		elif a == 0 and b == 0:
			p = 0.5
		elif b > a:
			p = 0.0
		else:
			p = 1.0

		pmap[k] = p

	return pmap

def bernoulli_betamix_posterior_mode(obs_cnt, tot_cnt, w, beta_params):

	dc = -gammaln(beta_params.sum(axis=1) + tot_cnt)
	lc = -gammaln(beta_params).sum(axis=1) + gammaln(beta_params.sum(axis=1)) + dc
	log_w = np.log(w)

	pmap = dict()
	for k, v in obs_cnt.items():

		ll = log_w + gammaln(beta_params[:,0]+v) + gammaln(beta_params[:,1]+(tot_cnt-v)) + lc
		bidx = ll.argmax()

		a = (beta_params[bidx,0] - 1) + v
		b = (beta_params[bidx,1] - 1) - v + tot_cnt

		if a > 0 and b > 0 and a > b:
			p = -b / (a + b)
		elif a > 0 and b > 0 and b > a:
			p = (a - 1) / (a + b - 2)
		elif a == 0 and b == 0:
			p = 0.5
		elif b > a:
			p = 0.0
		else:
			p = 1.0

		pmap[k] = p

	return pmap