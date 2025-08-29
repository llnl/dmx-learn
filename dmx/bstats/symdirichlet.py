from typing import Optional, Union, List, Dict
from collections import defaultdict
from dmx.bstats.pdist import SequenceEncodableDistribution, ProbabilityDistribution
from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.nulldist import null_dist
from scipy.special import gammaln
import numpy as np


class SymmetricDirichletDistribution(SequenceEncodableDistribution):

	def __init__(self, alpha: float):
		self.set_parameters(alpha)

	def __str__(self):
		return 'SymmetricDirichletDistribution(%s)'%(str(self.alpha))

	def get_parameters(self) -> float:
		return self.alpha

	def set_parameters(self, params: float) -> None:
		self.alpha = params

	def density(self, x: Union[float, np.ndarray, List[float]]) -> float:
		return np.exp(self.log_density(x))

	def log_density(self, x: Union[float, np.ndarray, List[float]]) -> float:
		nc = len(x) * gammaln(self.alpha) - gammaln(len(x) * self.alpha)

		if self.alpha == 1:
			return nc
		else:
			return np.sum(np.log(x) * (self.alpha-1)) - nc

	def sampler(self, seed: Optional[int] = None):
		return SymmetricDirichletSampler(self, seed)


class SymmetricDirichletSampler(object):

	def __init__(self, dist: SymmetricDirichletDistribution, seed: Optional[int] = None):
		self.dist = dist
		self.dir  = np.random.RandomState(seed)

	def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
		a = self.dist.alpha
		n = self.dist.ndim
		return self.dir.dirichlet(np.ones(n)*a, size=size)

