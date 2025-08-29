"""Create, estimate, and sample from a Categorical distribution defined on a range of integers starting a user defined minimum value.

Defines the IntegerCategoricalDistribution, IntegerCategoricalSampler, IntegerCategoricalAccumulatorFactory,
IntegerCategoricalAccumulator, IntegerCategoricalEstimator, and the IntegerCategoricalDataEncoder classes for use
with dmx-learn.

Data type (int): The integer categorical distribution is defined through summary statistics min_val (int)
and vector of probabilities p_vec (np.ndarray[float]) that sum to 1.0. The range of values is given by
[min_val, min_val + len(p_vec) - ). The density is then,

"""
import numpy as np
from numpy.random import RandomState
import dmx.utils.vector as vec
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence
from typing import List, Union, Tuple, Optional, Dict, Any


class IntegerCategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """IntegerCategoricalDistribution object defining an integer categorical distribution.

    Attributes:
        p_vec (np.ndarray[float]): Must sum to 1.0. First probability is probability for p_mat(x_mat=min_val).
        min_val (int): Minimum value in support of integer categorical
        max_val (int): Maximum value in support of integer categorical set to min_val + length(p_vec) - 1.
        log_p_vec (np.ndarray[float]): Log of p_vec.
        num_vals (int): Total number of values in support of IntegerCategoricalDistribution instance.
        name (Optional[str]): Name for object. 
        keys (Optional[str]): Key for parameter. 

    """

    def __init__(self, min_val: int, p_vec: Union[List[float], np.ndarray], name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """IntegerCategoricalDistribution object.

        Args:
            min_val (int): Minimum value of the integer categorical support.
            p_vec (Union[List[float], np.ndarray]): Probability vector containing probability of each integer in the
                support range.
            name (Optional[str]): Assign name to IntegerCategoricalDistribution object.
            keys (Optional[str]): Key for parameter. 

        """
        with np.errstate(divide='ignore'):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.name = name
            self.keys = keys

    def __str__(self) -> str:
        s1 = str(self.min_val)
        s2 = repr(self.p_vec.tolist())
        s3 = repr(self.name)
        s4 = repr(self.keys)

        return 'IntegerCategoricalDistribution(%s, %s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: int) -> float:
        """Evaluate the density of the integer categorical at observation x.

        Args:
            x (int): Integer value.

        Returns:
            float: Density at x.

        """
        return zero if x < self.min_val or x > self.max_val else self.p_vec[x - self.min_val]

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the integer categorical at observation x.

        Args:
            x (int): Integer value.

        Returns:
            float: Log-density at x.

        """
        return -inf if (x < self.min_val or x > self.max_val) else self.log_p_vec[x - self.min_val]

    def seq_log_density(self, x: 'IntegerCategoricalEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, IntegerCategoricalEncodedDataSequence):
            raise Exception('IntegerCategoricalEncodedDataSequence required for seq_log_density().')

        v = x.data - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x.data))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerCategoricalSampler':

        return IntegerCategoricalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerCategoricalEstimator':

        if pseudo_count is None:
            return IntegerCategoricalEstimator(name=self.name, keys=self.keys)

        else:
            return IntegerCategoricalEstimator(
                pseudo_count=pseudo_count, 
                suff_stat=(self.min_val, self.p_vec),
                name=self.name,
                keys=self.keys)

    def dist_to_encoder(self) -> 'IntegerCategoricalDataEncoder':
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalSampler(DistributionSampler):
    """IntegerCategoricalSampler object for sampling from IntegerCategoricalDistribution.

    Attributes:
        dist (IntegerCategoricalDistribution): IntegerCategoricalDistribution instance to sample from.
        rng (RandomState): RandomState object with seed set if passed.

    """

    def __init__(self, dist: 'IntegerCategoricalDistribution', seed: Optional[int] = None) -> None:
        """IntegerCategoricalSampler object.

        Args:
            dist (IntegerCategoricalDistribution): Set IntegerCategoricalDistribution instance to sample from.
            seed (Optional[int]): Set the seed for random number generator used to sample.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw iid samples from IntegerCategoricalSampler object.

        Note: If size is None, a single sample is returned as an integer. If size > 0, a List of integers with
        length equal to size is returned.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Integer or List[int] of iid samples from IntegerCategoricalSampler instance.

        """
        if size is None:
            return int(self.rng.choice(range(self.dist.min_val, self.dist.max_val +
                                         1), p=self.dist.p_vec))

        else:
            return self.rng.choice(range(self.dist.min_val, self.dist.max_val +
                                         1), p=self.dist.p_vec,
                                   size=size).tolist()


class IntegerCategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """IntegerCategoricalAccumulator object for accumulating sufficient statistics from observed data.

    Notes:
        If min_val and max_val are not provided, they are obtained from the data in accumulation step.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical range.
        max_val (Optional[int]): Maximum value of integer categorical range.
        count_vec (Optional[np.ndarray]): Numpy array of floats for tracking probability weights for each integer
            value in support. Set to None if min_val and max_val are both not None.
        name (Optional[str]): Name for object. 
        keys (Optional[str]): Key for merging sufficient statistics of integer IntegerCategoricalAccumulator
            objects.

    """

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, keys: Optional[str] = None, name: Optional[str] = None) \
            -> None:
        """IntegerCategoricalAccumulator object.

        Args:
            min_val (Optional[int]): Sets the minimum value of integer categorical range.
            max_val (Optional[int]): Sets the maximum value of integer categorical range.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set key for merging sufficient statistics of integer IntegerCategoricalAccumulator
                objects.

        """
        self.min_val = min_val
        self.max_val = max_val

        if min_val is not None and max_val is not None:
            self.count_vec = vec.zeros(max_val - min_val + 1)

        else:
            self.count_vec = None

        self.name = name
        self.keys = keys

    def update(self, x: int, weight: float, estimate: Optional['IntegerCategoricalDistribution']) -> None:

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

    def seq_initialize(self, x: 'IntegerCategoricalEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        return self.seq_update(x, weights, None)

    def seq_update(self, x: 'IntegerCategoricalEncodedDataSequence', weights: np.ndarray, estimate: Optional['IntegerCategoricalDistribution'])\
            -> None:
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

    def combine(self, suff_stat: Tuple[Optional[int], Optional[np.ndarray]]) -> 'IntegerCategoricalAccumulator':

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

    def from_value(self, x: Tuple[int, np.ndarray]) -> 'IntegerCategoricalAccumulator':
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

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

    def acc_to_encoder(self) -> 'IntegerCategoricalDataEncoder':
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """IntegerCategoricalAccumulatorFactory object for creating IntegerCategoricalAccumulator object.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical, if None estimated from data.
        max_val (Optional[int]): Maximum value of integer categorical, if None estimated from data.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Key used for accumulating merging statistics of IntegerCategoricalAccumulator objects.

    """

    def __init__(self, min_val: Optional[int] = None, 
                 max_val: Optional[int] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerCategoricalAccumulatorFactory object.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.keys = keys

    def make(self) -> 'IntegerCategoricalAccumulator':
        return IntegerCategoricalAccumulator(min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name)


class IntegerCategoricalEstimator(ParameterEstimator):
    """IntegerCategoricalEstimator object for estimating IntegerCategoricalDistribution

    Notes:
        Must set either min_val and max_val, or suff_stat must be passed as arg.
    
    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical.
        max_val (Optional[int]): Maximum value of integer categorical.
        pseudo_count (Optional[float]): Used to re-weight suff_stat when merged with new aggregated data.
        suff_stat (Tuple[int, np.ndarray]): min value and prob vec
        name (Optional[str]): Name to IntegerCategoricalEstimator object.
        keys (Optional[str]): Keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

    """

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[int, np.ndarray]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerCategoricalEstimator object.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            pseudo_count (Optional[float]): Used to re-weight suff_stat member variables in merging of sufficient
                statistics
            suff_stat: Set sufficient statistics. See above for details.
            name (Optional[str]): Assign a name to IntegerCategoricalEstimator object.
            keys (Optional[str]): Set keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("IntegerCategoricalEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'IntegerCategoricalAccumulatorFactory':
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        return IntegerCategoricalAccumulatorFactory(min_val=min_val, max_val=max_val, name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Tuple[int, np.ndarray]])\
            -> 'IntegerCategoricalDistribution':

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(suff_stat[0], (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs,
                                                  name=self.name)

        elif self.pseudo_count is not None and self.min_val is not None and self.max_val is not None:

            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(min_val, (count_vec + pseudo_count_per_level) / adjusted_nobs,
                                                  name=self.name)

        elif self.pseudo_count is not None and self.suff_stat is not None:

            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            return IntegerCategoricalDistribution(min_val, count_vec / (count_vec.sum()), name=self.name)

        else:
            return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name)


class IntegerCategoricalDataEncoder(DataSequenceEncoder):
    """IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical observations."""

    def __str__(self) -> str:
        """Returns IntegerCategoricalDataEncoder object for encoding data sequences."""
        return 'IntegerCategoricalDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerCategoricalDataEncoder, False is else."""
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> 'IntegerCategoricalEncodedDataSequence':
        return IntegerCategoricalEncodedDataSequence(data=np.asarray(x, dtype=int))


class IntegerCategoricalEncodedDataSequence(EncodedDataSequence):
    """IntegerCategoricalEncodedDataSequence for vectorized function calls.

    Attributes:
        data (np.ndarray): IID observations from integer categorical distribution.

    """

    def __init__(self, data: np.ndarray):
        """IntegerCategoricalEncodedDataSequence object.

        Args:
            data (np.ndarray): IID observations from integer categorical distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'IntegerCategoricalEncodedDataSequence(data={self.data})'


