"""Create, estimate, and sample from a Conditional distribution.

Defines the BernoulliSetDistribution, BernoulliSetDistributionSampler, BernoulliSetDistributionAccumulatorFactory,
BernoulliSetDistributionAccumulator, BernoulliSetDistributionEstimator, and the BernoulliSetDistributionDataEncoder
classes for use with dmx-learn.


Let S = {s_1,s_2,....,s_N} be the state space of elements of any type. Let x be a random set of variable length,
with domain on the subsets of S. The Bernoulli set distribution models x with of element s_k in the

    p_k = P(S_k is in x) , k = 1,2,...,N.

A comment on estimation: Note that probability of an element s_k belonging to the set is 0 if we do not encounter any
elements an observation sequence. For this reason, we need not state the support of the state-space in estimation.

"""
import numpy as np
from numpy.random import RandomState
from collections import defaultdict, OrderedDict
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, StatisticAccumulatorFactory, DistributionSampler, EncodedDataSequence

from typing import Optional, Dict, Tuple, Any, Dict, List, Sequence, TypeVar, Union


class BernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """BernoulliSetDistribution object for creating a Bernoulli set distribution.

    Attributes:
        keys (Optional[str]): Keys for object instance.
        name (Optional[str]): Name to object instance.
        pmap (Dict[Any, float]): Maps elements in support to probabilities.
        required (Set): An observation must contain this subset of elements. Else, return probability 0.0.
        nlog_sum (float): Normalizing term for computing numerically stable likelihood.
        log_dmap (Dict[Any, float]):Map from elements to their corrected log probability of inclusion in the set.
        min_prob (float): Minimum probability for elements. Corrects for prob = 0.
        num_required (int): Number of required elements in a subset. Corrected if min_prob was non-zero.

    """

    def __init__(self, pmap: Dict[Any, float], min_prob: float = 1.0e-128, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """BernoulliSetDistribution object.

        Args:
            pmap (Dict[Any, float]): Maps values to probabilities.
            min_prob (float): Minimum probability for numerical stability in log prob calculations.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        self.keys = keys
        self.name = name
        self.pmap = pmap
        self.required = set()
        self.nlog_sum = 0.0
        self.log_dmap = dict()

        if min_prob == 0:
            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = 0.0
                    self.required.add(k)
                elif v == 0.0:
                    self.log_dmap[k] = -np.inf
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv
            self.min_prob = 0.0
            self.num_required = len(self.required)

        else:
            min_pv = np.log(min_prob)
            min_nv = np.log1p(-min_prob)

            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = min_nv - min_pv
                    self.nlog_sum += min_pv
                elif v == 0.0:
                    self.log_dmap[k] = min_pv - min_nv
                    self.nlog_sum += min_nv
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv

            self.min_prob = min_prob
            self.num_required = 0

    def __str__(self) -> str:
        s1 = repr(sorted(self.pmap.items(), key=lambda t: t[0]))
        s2 = repr(self.min_prob)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return 'BernoulliSetDistribution(dict(%s), min_prob=%s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: Sequence[Any]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Any]) -> float:

        if not self.required.issubset(x):
            return -np.inf
        rv = 0.0
        for v in x:
            rv += self.log_dmap[v]

        return self.nlog_sum + rv

    def seq_log_density(self, x: 'BernoulliSetEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, BernoulliSetEncodedDataSequence):
            raise Exception('BernoulliSetEncodedDataSequence required for seq_log_density().')

        sz, idx, val_map_inv, xs = x.data

        dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)

        rv = np.bincount(idx, weights=dlog_loc[xs], minlength=sz)
        rv += self.nlog_sum

        if self.num_required != 0:
            required_loc = np.isin(val_map_inv, self.required)
            req_cnt = np.bincount(idx, weights=required_loc[xs], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'BernoulliSetSampler':
        return BernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'BernoulliSetEstimator':

        if pseudo_count is None:
            return BernoulliSetEstimator(min_prob=self.min_prob, name=self.name, keys=self.keys)
        else:
            return BernoulliSetEstimator(min_prob=self.min_prob, pseudo_count=pseudo_count, suff_stat=self.pmap,
                                         name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'BernoulliSetDataEncoder':
        return BernoulliSetDataEncoder()


class BernoulliSetSampler(DistributionSampler):
    """BernoulliSetSampler object for generating samples from BernoulliSetDistribution object instance.

    Attributes:
        dist (BernoulliSetDistribution): Object instance to sample from.
        seed (Optional[int]): Set seed for random number generator.

    """

    def __init__(self, dist: BernoulliSetDistribution, seed: Optional[int] = None) -> None:
        """BernoulliSetSampler object.

        Args:
            dist (BernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Set seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], List[Sequence[Any]]]:

        if size is not None:
            retval = [[] for i in range(size)]
            for k, v in self.dist.pmap.items():
                for i in np.flatnonzero(self.rng.rand(size) <= v):
                    retval[i].append(k)
            return retval

        else:
            retval = []
            for k, v in self.dist.pmap.items():
                if self.rng.rand() <= v:
                    retval.append(k)
            return retval


class BernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """BernoulliSetAccumulator object for accumulating sufficient statistics from observed data.

    Attributes:
        pmap (Dict[Any, float]): Dictionary mapping values to set-inclusion probabilities.
        tot_sum (float): Weighted observation count.
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """BernoulliSetAccumulator object.

        Args:
            keys (Optional[str]): Set keys for merging sufficient statistics.
            name (Optional[str]): Name for object. 

        """
        self.pmap = defaultdict(float)
        self.tot_sum = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: Sequence[Any], weight: float, estimate: Optional[BernoulliSetDistribution]) -> None:
        for u in x:
            self.pmap[u] += weight
        self.tot_sum += weight

    def initialize(self, x: Sequence[Any], weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: 'BernoulliSetEncodedDataSequence', weights: np.ndarray,
                   estimate: Optional[BernoulliSetDistribution]) -> None:

        sz, idx, val_map_inv, xs = x.data
        agg_cnt = np.bincount(xs, weights[idx])

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += weights.sum()

    def seq_initialize(self, x: 'BernoulliSetEncodedDataSequence', weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[Dict[Any, float], float]) -> 'BernoulliSetAccumulator':
        for k, v in suff_stat[0].items():
            self.pmap[k] += v
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[Dict[Any, float], float]:
        return dict(self.pmap), self.tot_sum

    def from_value(self, x: Tuple[Dict[Any, float], float]) -> 'BernoulliSetAccumulator':
        self.pmap = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> 'BernoulliSetDataEncoder':
        return BernoulliSetDataEncoder()


class BernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """BernoulliSetAccumulatorFactory object for creating instances of BernoulliSetAccumulator objects.

    Attributes:
        keys (Optional[str]): Key for suff stats.
        name (Optional[str]): Name for object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """BernoulliSetAccumulatorFactory object.

        Attributes:
            keys (Optional[str]): Key for suff stats.
            name (Optional[str]): Name for object.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'BernoulliSetAccumulator':
        return BernoulliSetAccumulator(keys=self.keys, name=self.name)


class BernoulliSetEstimator(ParameterEstimator):
    """BernoulliSetEstimator object for estimating Bernoulli set distribution from aggregated sufficient statistics.

    Attributes:
        min_prob (float): Minimum probability for elements estimated with prob = 0.
        pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
        suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Set key for merging sufficient statistics.

    """

    def __init__(self, min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Dict[Any, float]] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """BernoulliSetEstimator object.

        Args:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for merging sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("BernoulliSetEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> 'BernoulliSetAccumulatorFactory':
        return BernoulliSetAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Dict[Any, float], float]) -> 'BernoulliSetDistribution':

        if self.pseudo_count is not None and self.suff_stat is not None:
            keys = set(suff_stat[0].keys())
            keys.update(self.suff_stat.keys())

            pmap = {k: (self.suff_stat.get(k, 0.0) * self.pseudo_count + suff_stat[0].get(k, 0.0)) / (
                        self.pseudo_count + suff_stat[1]) for k in keys}

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            cnt = float(p + suff_stat[1])
            pmap = {k: (v + (p / 2.0)) / cnt for k, v in suff_stat[0].items()}

        else:

            if suff_stat[1] != 0:
                pmap = {k: v / suff_stat[1] for k, v in suff_stat[0].items()}
            else:
                pmap = {k: 0.5 for k in suff_stat[0].keys()}

        return BernoulliSetDistribution(pmap, min_prob=self.min_prob, name=self.name)

class BernoulliSetDataEncoder(DataSequenceEncoder):
    """BernoulliSetDataEncoder for encoding sequences of iid observations."""

    def __str__(self) -> str:
        return 'BernoulliSetDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BernoulliSetDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> 'BernoulliSetEncodedDataSequence':
        idx = []
        xs = []

        for i in range(len(x)):
            idx.extend([i] * len(x[i]))
            xs.extend(x[i])

        val_map, xs = np.unique(xs, return_inverse=True)

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        return BernoulliSetEncodedDataSequence(data=(len(x), idx, val_map, xs))

class BernoulliSetEncodedDataSequence(EncodedDataSequence):
    """BernoulliSetEncodedDataSequence object for vectorized function calls.

    Notes:
        Encoded value 'E' is a Tuple of length 4 containing:
        E[0] (int): Number of observed sets.
        E[1] (np.ndarray): Numpy array of integer indices for flattened array of values.
        E[2] (np.ndarray): Numpy array of unique values. (dtype is object).
        E[3] (np.ndarray): Numpy array of val_map (rv[2]) integer indices for flattened array of values.

    Attributes:
        data (E): See above.

    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
        """BernoulliSetEncodedDataSequence object.

        Args:
            data (E): See above.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'BernoulliSetEncodedDataSequence(data={self.data})'

