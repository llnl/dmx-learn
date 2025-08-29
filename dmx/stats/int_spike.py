"""Evaluate, estimate, and sample from a uniform distribution over integers in range [min_val, max_val] with a spike
  placed on the integer value k.

Defines the SpikeAndSlabDistribution, SpikeAndSlabSampler, SpikeAndSlabAccumulatorFactory,
SpikeAndSlabAccumulator, SpikeAndSlabEstimator, and the SpikeAndSlabDataEncoder classes for use
with dmx-learn.

"""

from typing import List, Union, Tuple, Optional, Dict, Any

import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence


class SpikeAndSlabDistribution(SequenceEncodableProbabilityDistribution):
    """SpikeAndSlabDistribution object for creating a uniform integer distribution with a spike on k.

    Attributes:
        p (float): Probability of drawing from k.
        min_val (int): Lower bound for the range.
        max_val (int): Max value for the range.
        k (int): Integer to place the spike on.
        log_p (float): Log of p.
        log_1p (float): Log of 1-p
        num_vals (int): Total number of integers in range.
        name (Optional[str]): Name for object instance.
        keys (Optional[str]): Key for parameters. 

    """

    def __init__(self, k: int, num_vals: int,  p: float, min_val: Optional[int] = 0, name: Optional[str] = None, keys: Optional[str] = None) \
            -> None:
        """SpikeAndSlabDistribution object.

        Args:
            k (int): Integer value to place spike on. Must be within [min_val,min_val+num_vals)
            num_vals (int): Number of integers in the range.
            p (float): Probability of drawing k. (1-p)/(num_vals-1) to draw any other integer in range.
            min_val (Optional[int]): Defaults to 0. Set bottom of integer range.
            name (Optional[str]): Set name for object.
            keys (Optional[str]): Key for parameters. 

        """
        self.p = p
        self.min_val = min_val
        self.max_val = min_val + num_vals

        if not self.min_val <= k <= self.max_val:
            raise Exception('Spike value k must be between [%s, %s].' % (repr(self.min_val), repr(self.max_val)))
        else:
            self.k = k

        self.log_p = np.log(p)
        self.num_vals = num_vals
        self.log_1p = np.log1p(-self.p) - np.log(self.num_vals-1)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s1 = str(self.min_val)
        s2 = str(self.num_vals)
        s3 = repr(self.p)
        s4 = repr(self.k)
        s5 = repr(self.name)
        s6 = repr(self.keys)

        return 'SpikeAndSlabDistribution(p=%s, min_val=%s, num_vals=%s,k=%s, name=%s, keys=%s)' % (s3, s1, s2, s4, s5, s6)

    def density(self, x: int) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        if self.max_val >= x >= self.min_val:
            return self.log_p if x == self.k else self.log_1p
        else:
            return -np.inf

    def seq_log_density(self, x: 'SpikeAndSlabEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, SpikeAndSlabEncodedDataSequence):
            raise Exception("SpikeAndSlabEncodedDataSequence required for seq_log_density().")

        rv = np.zeros(len(x.data), dtype=float)
        rv.fill(-np.inf)

        in_range = np.bitwise_and(x.data >= self.min_val, x.data <= self.max_val)
        in_range_k = x.data[in_range] == self.k

        rv1 = rv[in_range]
        rv1[in_range_k] = self.log_p
        rv1[~in_range_k] = self.log_1p
        rv[in_range] = rv1

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SpikeAndSlabSampler':
        return SpikeAndSlabSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SpikeAndSlabEstimator':
        if pseudo_count is None:
            return SpikeAndSlabEstimator(min_val=self.min_val, max_val=self.max_val, name=self.name, keys=self.keys)

        else:
            return SpikeAndSlabEstimator(min_val=self.min_val, max_val=self.max_val,
                                                pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'SpikeAndSlabDataEncoder':
        return SpikeAndSlabDataEncoder()


class SpikeAndSlabSampler(DistributionSampler):
    """SpikeAndSlabSampler object for sampling from spike and slab distribution on integers.

    Attributes:
        rng (RandomState): RandomState for seeding samples.
        dist (SpikeAndSlabDistribution): SpikeAndSlabDistribution to sample from.
        non_k (np.ndarray): All integers outside of the spiked value 'k'.
    """

    def __init__(self, dist: 'SpikeAndSlabDistribution', seed: Optional[int] = None) -> None:
        """SpikeAndSlabSampler object.

        Args:
            dist (SpikeAndSlabDistribution): SpikeAndSlabDistribution to sample from.
            seed (Optional[int]): Seed for generating samples.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.non_k = np.delete(np.arange(self.dist.min_val, self.dist.max_val), self.dist.k)

    def sample(self, size: Optional[int] = None) -> Union[int, np.array]:

        if size is None:
            z = self.rng.binomial(n=1, p=self.dist.p)
            if z == 1:
                return self.dist.k
            else:
                return self.rng.choice(self.non_k)
        else:

            rv = np.zeros(size, dtype=int)
            rv.fill(self.dist.k)
            z = self.rng.binomial(n=1, p=self.dist.p, size=size)
            idx = np.flatnonzero(z == 0)

            if len(idx) > 0:
                rv[idx] = self.rng.choice(self.non_k, replace=True, size=len(idx))

            return rv

class SpikeAndSlabAccumulator(SequenceEncodableStatisticAccumulator):
    """SpikeAndSlabAccumulator object for accumulating sufficient statistics.

    Attributes:
        min_val (Optional[int]): Smallest integer value in the range. Defaults to 0.
        max_val (Optional[int]): Set to the min val plus number of values - 1.
        count_vec (Optional[np.ndarray]): suff stat, counts for each numeric value.
        count (float): Weighted obs count.
        keys (Optional[str]): Set keys for object instance.
        name (Optional[str]): Set name for object instance.

    """

    def __init__(self, min_val: Optional[int], max_val: Optional[int], keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """SpikeAndSlabAccumulator object.

        Args:
            min_val (Optional[int]): Smallest integer value in the range. Defaults to 0.
            max_val (Optional[int]): Set to the min val plus number of values - 1.
            num_vals (Optional[
            keys (Optional[str]): Set keys for object instance.
            name (Optional[str]): Set name for object instance.

        """
        self.min_val = min_val
        self.max_val = max_val

        if self.min_val is not None and self.max_val is not None:
            self.count_vec = np.zeros(self.max_val-self.min_val + 1, dtype=float)
        else:
            self.count_vec = None

        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional['SpikeAndSlabDistribution']) -> None:

        if self.count_vec is None:
            self.min_val = x
            self.max_val = x
            self.count_vec = np.asarray([weight])

        elif self.max_val < x:
            temp_vec = self.count_vec
            self.max_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[:len(temp_vec)] = temp_vec
            self.count_vec[x - self.min_val] += weight

        elif self.min_val > x:
            temp_vec = self.count_vec
            temp_diff = self.min_val - x
            self.min_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[temp_diff:] = temp_vec
            self.count_vec[x - self.min_val] += weight

        else:
            self.count_vec[x - self.min_val] += weight

    def initialize(self, x: int, weight: float, rng: RandomState) -> None:
        return self.update(x, weight, None)

    def seq_initialize(self, x: 'SpikeAndSlabEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        return self.seq_update(x, weights, None)

    def seq_update(self, x: 'SpikeAndSlabEncodedDataSequence', weights: np.ndarray,
                   estimate: Optional['SpikeAndSlabDistribution']) -> None:

        min_x = np.min(x.data)
        max_x = np.max(x.data)

        loc_cnt = np.bincount(x.data - min_x, weights=weights)

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff:(min_diff + len(loc_cnt))] += loc_cnt

    def combine(self, suff_stat: Tuple[int, np.ndarray]) -> 'SpikeAndSlabAccumulator':
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> Tuple[int, np.ndarray]:
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> 'SpikeAndSlabAccumulator':
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'SpikeAndSlabDataEncoder':
        return SpikeAndSlabDataEncoder()


class SpikeAndSlabAccumulatorFactory(StatisticAccumulatorFactory):
    """SpikeAndSlabAccumulatorFactory object for creating accumulators.

    Attributes:
            min_val (int]): Smallest integer value in the range. Defaults to 0.
            max_val (int): Set to the min val plus number of values - 1.
            keys (Optional[str]): Set keys for object instance.
            name (Optional[str]): Set name for object instance.

    """

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """SpikeAndSlabAccumulatorFactory object.

        Args:
            min_val (Optional[int]): Smallest integer value in the range. Defaults to 0.
            max_val (Optional[int]): Set to the min val plus number of values - 1.
            keys (Optional[str]): Set keys for object instance.
            name (Optional[str]): Set name for object instance.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys
        self.name = name

    def make(self) -> 'SpikeAndSlabAccumulator':
        return SpikeAndSlabAccumulator(min_val=self.min_val, max_val=self.max_val, keys=self.keys,
                                              name=self.name)


class SpikeAndSlabEstimator(ParameterEstimator):
    """SpikeAndSlabEstimator object instance for estimating SpikeAndSlabDistribution objects.

    Attributes:
        pseudo_count (Optional[float]): Regularize value k.
        min_val (int): Smallest integer value in the range. Defaults to 0.
        max_val (int): Set to the min val plus number of values - 1.
        suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Set keys for object instance.

    """

    def __init__(self, min_val: Optional[int] = None,
                 max_val: Optional[int] = None,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[int, Optional[float]]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SpikeAndSlabEstimator object.

        Args:
            min_val (Optional[int]): Smallest integer value in the range.
            pseudo_count (Optional[float]): Regularize value k.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("SpikeAndSlabEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat if suff_stat is not None else (None, None)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'SpikeAndSlabAccumulatorFactory':
        return SpikeAndSlabAccumulatorFactory(min_val=self.min_val, max_val=self.max_val,
                                                     keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray]) -> 'SpikeAndSlabDistribution':
        min_val, count_vec = suff_stat

        with np.errstate(divide='ignore'):
            if self.pseudo_count is None:
                count = np.sum(count_vec)
                p_vec = count_vec / count
                ll = np.log1p(-p_vec)
                ll -= np.log(len(count_vec)-1)
                ll *= (count-count_vec)
                ll += count_vec*np.log(p_vec)
                k = np.argmax(ll)
                p = p_vec[k]

                return SpikeAndSlabDistribution(k=k if min_val is None else k+min_val,
                                                       min_val=min_val, num_vals=len(count_vec),
                                                       p=p, name=self.name)
            if self.pseudo_count is not None:
                if self.suff_stat[0] is not None and self.suff_stat[1] is None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return SpikeAndSlabDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)

                elif self.suff_stat[0] is not None and self.suff_stat[1] is not None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count*self.suff_stat[1]
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return SpikeAndSlabDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)
                else:
                    count_vec += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return SpikeAndSlabDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)


class SpikeAndSlabDataEncoder(DataSequenceEncoder):
    """IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical observations. """

    def __str__(self) -> str:
        return 'IntegerCategoricalDataEncoder'

    def __eq__(self, other: object) -> bool:
        return True if isinstance(other, SpikeAndSlabDataEncoder) else False

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> 'SpikeAndSlabEncodedDataSequence':
        return SpikeAndSlabEncodedDataSequence(data=np.asarray(x, dtype=int))

class SpikeAndSlabEncodedDataSequence(EncodedDataSequence):
    """SpikeAndSlabEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (np.ndarray): Encoded sequence of integer values.

    """

    def __init__(self, data: np.ndarray):
        """SpikeAndSlabEncodedDataSequence object for vectorized function calls.

        Args:
            data (np.ndarray): Encoded sequence of integer values.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'SpikeAndSlabEncodedDataSequence(data={self.data})'




