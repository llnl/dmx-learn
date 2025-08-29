"""Create, estimate, and sample from a Spearman ranking distribution.

Defines the SpearmanRankingDistribution, SpearmanRankingSampler, SpearmanRankingAccumulatorFactory,
SpearmanRankingAccumulator, SpearmanRankingEstimator, and the SpearmanRankingDataEncoder
classes for use with dmx-learn.

Data type: List[int] (Component-wise rank of K dimensional observation vector)

The Spearman ranking distribution with dimension K, has probability function

    p_mat(x_k;rho, sigma) = exp(-rho * ||x_k-sigma||^2 ) / sum_{k=0}^{K-1} exp(-rho * ||x_k-sigma||^2 ), for k = 0,1,..,K-1

where x_k list of integers containing a permutation of the integers 0,1,2,...K-1. Note sigma is a list of floats with
dimension equal to K representing the mean of the rank variables, and rho is a correlation coefficient.

"""

import numpy as np
from numpy.random import RandomState
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence
import itertools

from typing import Optional, Sequence, Union, Any, Dict, List, Tuple


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):
    """SpearmanRankingDistribution object for defining a Spearman ranking distribution.

    Attributes:
        sigma (np.ndarray]): Numpy array of means for the rank variables.
        rho (float): Decay rate on variance of ranks.
        name (Optional[str]): Name for object instance.
        dim (int): Dimension of the rank variable.
        keys (Optional[str]): Set keys for object instance.

    """

    def __init__(self, sigma: Union[Sequence[float], np.ndarray], rho: float = 1.0, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SpearmanRankingDistribution object.

        Args:
            sigma (np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Name for object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        self.sigma = np.asarray(sigma)
        self.rho = rho
        self.name = name
        self.dim = len(sigma)
        self.keys = keys

        perms = map(np.asarray, map(list, itertools.permutations(range(self.dim))))
        self.log_const = np.log(sum(map(lambda u: np.exp(-rho * np.dot(self.sigma - u, self.sigma - u)), perms)))

    def __str__(self) -> str:
        return 'SpearmanRankingDistribution(sigma=%s, rho=%s, name=%s, keys=%s)' % (
            repr(self.sigma.tolist()), repr(self.rho), repr(self.name), repr(self.keys))

    def density(self, x: List[int]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: List[int]) -> float:
        temp = np.subtract(x, self.sigma)
        return -self.rho * np.dot(temp, temp) - self.log_const

    def seq_log_density(self, x: 'SpearmanRankingEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, SpearmanRankingEncodedDataSequence):
            raise Exception('SpearmanRankingEncodedDataSequence required for seq_log_density().')

        temp = x.data - self.sigma
        temp *= temp
        rv = np.sum(temp, axis=1) * -self.rho
        rv -= self.log_const
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SpearmanRankingSampler':
        return SpearmanRankingSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SpearmanRankingEstimator':
        return SpearmanRankingEstimator(self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'SpearmanRankingDataEncoder':
        return SpearmanRankingDataEncoder()


class SpearmanRankingSampler(DistributionSampler):
    """SpearmanRankingSampler object for sampling from SpearmanRankingDistribution.

    Attributes:
        rng (RandomState): Seed samples.
        dist (SpearmanRankingDistribution): Distribution to draw samples from.
        perms (List[List[int]]): List of all possible rankings.
        probs (np.ndarray): Probability of each permutation.

    """

    def __init__(self, dist: SpearmanRankingDistribution, seed: Optional[int] = None) -> None:
        """SpearmanRankingSampler object..

        Args:
            dist (SpearmanRankingDistribution): Distribution to draw samples from.
            seed (Optional[int]): Set seed for generating samples.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.perms = list(map(list, itertools.permutations(range(dist.dim))))
        encoder = self.dist.dist_to_encoder()
        self.probs = np.exp(dist.seq_log_density(encoder.seq_encode(self.perms)))

    def sample(self, size: Optional[int] = None) -> Union[List[int], Sequence[List[int]]]:
        idx = self.rng.choice(len(self.perms), p=self.probs, replace=True, size=size)

        if size is None:
            return self.perms[idx]
        else:
            return [self.perms[u] for u in idx]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):
    """SpearmanRankingAccumulatorFactory object for creating SpearmanRankingAccumulator objects.

    Attributes:
        sum (np.ndarray): Suff stat counts
        count (float): Suff stat total weight count.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.
    """

    def __init__(self, dim: int, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """SpearmanRankingAccumulator object.

        Args:
            dim (int): Dimension of rankings.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        self.sum = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: Union[List[int], np.ndarray], weight: float, estimate: Optional[SpearmanRankingDistribution])\
            -> None:
        self.sum += np.multiply(x, weight)
        self.count += weight

    def initialize(self, x: Union[List[int], np.ndarray], weight: float, rng: RandomState) -> None:
        if weight != 0:
            self.sum += np.multiply(x, weight)
            self.count += 0

    def seq_update(self, x: 'SpearmanRankingEncodedDataSequence', weights: np.ndarray, estimate: Optional[SpearmanRankingDistribution]) -> None:
        self.sum += np.dot(x.data.T, weights)
        self.count += weights.sum()

    def seq_initialize(self, x: 'SpearmanRankingEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, np.ndarray]) -> 'SpearmanRankingAccumulator':
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self) -> Tuple[float, np.ndarray]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, np.ndarray]) -> 'SpearmanRankingAccumulator':
        self.sum = x[1]
        self.count = x[0]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                stats_dict[self.keys] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                self.count = vals[0]
                self.sum = vals[1]

    def acc_to_encoder(self) -> 'SpearmanRankingDataEncoder':
        return SpearmanRankingDataEncoder()


class SpearmanRankingAccumulatorFactory(StatisticAccumulatorFactory):
    """SpearmanRankingAccumulatorFactory object for creating SpearmanRankingAccumulator objects.

    Attributes:
        dim (int): Dimension of rankings.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(self, dim: int, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """SpearmanRankingAccumulatorFactory object.

        Args:
            dim (int): Dimension of rankings.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> 'SpearmanRankingAccumulator':
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):
    """SpearmanRankingEstimator object for estimating Spearman ranking distribution.

    Attributes:
        dim (int): Dimension of rankings.
        psuedo_count (Optional[float]): Regularize suff stat for estimates.
        suff_stat (Optional[Tuple[float, np.ndarray]]): Suff stat for regularization.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(self, dim: int, pseudo_count: Optional[float] = None, suff_stat: Optional[Tuple[float, np.ndarray]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SpearmanRankingEstimator object.

        Args:
            dim (int): Dimension of rankings.
            pseudo_count (Optional[float]): Regularize suff stat for estimates.
            suff_stat (Optional[Tuple[float, np.ndarray]]): Suff stat for regularization.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("SpearmanRankingEstimator requires keys to be of type 'str'.")

        self.dim = dim
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'SpearmanRankingAccumulatorFactory':
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, np.ndarray]) -> 'SpearmanRankingDistribution':

        count, vsum = suff_stat

        if count > 0:
            sigma = np.argsort(vsum)
            rho = 1.0
        else:
            sigma = vsum
            rho = 0.0

        return SpearmanRankingDistribution(sigma, rho, name=self.name, keys=self.keys)


class SpearmanRankingDataEncoder(DataSequenceEncoder):
    """SpearmanRankingDataEncoder for encoding sequences of spearman rho observations. """

    def __str__(self) -> str:
        return 'SpearmanRankingDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpearmanRankingDataEncoder)

    def seq_encode(self, x: Sequence[List[int]]) -> 'SpearmanRankingEncodedDataSequence':
        rv = np.asarray(x)

        return SpearmanRankingEncodedDataSequence(data=rv)

class SpearmanRankingEncodedDataSequence(EncodedDataSequence):
    """SpearmanRankingEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (np.ndarray): Iid observations from spearman rho ranking distribution.

    """

    def __init__(self, data: np.ndarray):
        """SpearmanRankingEncodedDataSequence object.

        Args:
            data (np.ndarray): Iid observations from spearman rho ranking distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'SpearmanRankingEncodedDataSequence(data={self.data})'

