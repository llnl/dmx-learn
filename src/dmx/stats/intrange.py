"""Categorical distributions over contiguous finite integer ranges.

``IntegerCategoricalDistribution`` assigns a probability vector of shape
``(K,)`` to consecutive integers beginning at ``min_val``. Scalar observations
are integers, while encoded sequences are one-dimensional integer arrays.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import inf, zero
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class IntegerCategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a categorical distribution on a contiguous integer range.

    Entry ``i`` of ``p_vec`` is the mass for category ``min_val + i``; the
    inclusive support ends at ``min_val + len(p_vec) - 1``.

    Attributes:
        p_vec (np.ndarray[float]): Must sum to 1.0. First probability is probability for
            p_mat(x_mat=min_val).
        min_val (int): Minimum value in support of integer categorical
        max_val (int): Maximum value in support of integer categorical set to min_val +
            length(p_vec) - 1.
        log_p_vec (np.ndarray[float]): Log of p_vec.
        num_vals (int): Total number of values in support of
            IntegerCategoricalDistribution instance.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Key for parameter.

    """

    def __init__(
        self,
        min_val: int,
        p_vec: Union[List[float], np.ndarray],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize probabilities and the support origin.

        Args:
            min_val (int): Minimum value of the integer categorical support.
            p_vec (Union[List[float], np.ndarray]): Probability vector containing
                probability of each integer in the
                support range.
            name (Optional[str]): Assign name to IntegerCategoricalDistribution object.
            keys (Optional[str]): Key for parameter.

        """
        super().__init__()
        with np.errstate(divide="ignore"):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.name = name
            self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        s1 = str(self.min_val)
        s2 = repr(self.p_vec.tolist())
        s3 = repr(self.name)
        s4 = repr(self.keys)

        return f"IntegerCategoricalDistribution({s1}, {s2}, name={s3}, keys={s4})"

    def density(self, x: int) -> float:
        """Evaluate the density of the integer categorical at observation x.

        Args:
            x (int): Integer value.

        Returns:
            float: Density at x.

        """
        return (
            zero
            if x < self.min_val or x > self.max_val
            else float(self.p_vec[x - self.min_val])
        )

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the integer categorical at observation x.

        Args:
            x (int): Integer value.

        Returns:
            float: Log-density at x.

        """
        return (
            -inf
            if (x < self.min_val or x > self.max_val)
            else float(self.log_p_vec[x - self.min_val])
        )

    def seq_log_density(self, x: "IntegerCategoricalEncodedDataSequence") -> np.ndarray:
        """Evaluate log masses for an encoded array of ``N`` integers.

        Returns:
            Array of shape ``(N,)`` with one log mass per observation.
        """
        if not isinstance(x, IntegerCategoricalEncodedDataSequence):
            raise TypeError(
                "IntegerCategoricalEncodedDataSequence required for seq_log_density()."
            )

        v = x.data - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x.data))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerCategoricalSampler":
        """Create a sampler for this distribution."""
        return IntegerCategoricalSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerCategoricalEstimator":
        """Create an estimator, optionally using current probabilities as a prior."""
        if pseudo_count is None:
            return IntegerCategoricalEstimator(name=self.name, keys=self.keys)

        return IntegerCategoricalEstimator(
            pseudo_count=pseudo_count,
            suff_stat=(self.min_val, self.p_vec),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Create the compatible integer encoder."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalSampler(DistributionSampler):
    """Draw integers from an integer categorical distribution.

    Attributes:
        dist (IntegerCategoricalDistribution): IntegerCategoricalDistribution instance
            to sample from.
        rng (RandomState): RandomState object with seed set if passed.

    """

    def __init__(
        self, dist: "IntegerCategoricalDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler for ``dist``.

        Args:
            dist (IntegerCategoricalDistribution): Set IntegerCategoricalDistribution
                instance to sample from.
            seed (Optional[int]): Set the seed for random number generator used to
                sample.

        """
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw iid samples from IntegerCategoricalSampler object.

        Notes:
            If size is None, a single sample is returned as an integer. If size > 0, a
            List of integers with length equal to size is returned.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Integer or List[int] of iid samples from IntegerCategoricalSampler instance.

        """
        if size is None:
            return int(
                self.rng.choice(
                    range(self.dist.min_val, self.dist.max_val + 1), p=self.dist.p_vec
                )
            )

        sample = self.rng.choice(
            range(self.dist.min_val, self.dist.max_val + 1),
            p=self.dist.p_vec,
            size=size,
        )
        return [int(v) for v in sample]


class IntegerCategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted counts over a contiguous integer range.

    The sufficient statistic is ``(minimum, count_vector)``. Entry ``i`` of the
    one-dimensional count vector is the weight for integer ``minimum + i``.

    Notes:
        If min_val and max_val are not provided, they are obtained from the data in
        accumulation step.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical range.
        max_val (Optional[int]): Maximum value of integer categorical range.
        count_vec (Optional[np.ndarray]): Numpy array of floats for tracking probability
            weights for each integer
            value in support. Set to None if min_val and max_val are both not None.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Key for merging sufficient statistics of integer
            IntegerCategoricalAccumulator
            objects.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an optional fixed range and its zero counts.

        Args:
            min_val (Optional[int]): Sets the minimum value of integer categorical
                range.
            max_val (Optional[int]): Sets the maximum value of integer categorical
                range.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set key for merging sufficient statistics of integer
                IntegerCategoricalAccumulator
                objects.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.count_vec: Optional[np.ndarray]

        if min_val is not None and max_val is not None:
            self.count_vec = vec.zeros(max_val - min_val + 1)

        else:
            self.count_vec = None

        self.name = name
        self.keys = keys

    def update(
        self,
        x: int,
        weight: float,
        estimate: Optional["IntegerCategoricalDistribution"],
    ) -> None:
        """Add one weighted integer, extending the represented range."""
        if self.count_vec is None:
            self.min_val = x
            self.max_val = x
            self.count_vec = np.asarray([weight])
            return

        assert self.min_val is not None
        assert self.max_val is not None

        if self.max_val < x:
            temp_vec = self.count_vec
            self.max_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[: len(temp_vec)] = temp_vec
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
        """Add one weighted integer during randomized initialization."""
        del rng
        return self.update(x, weight, None)

    def seq_initialize(
        self,
        x: "IntegerCategoricalEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Add encoded observations during randomized initialization."""
        return self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "IntegerCategoricalEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["IntegerCategoricalDistribution"],
    ) -> None:
        """Add ``N`` encoded integers with weights of shape ``(N,)``."""
        min_x = int(np.min(x.data))
        max_x = int(np.max(x.data))

        loc_cnt = np.bincount(x.data - min_x, weights=weights)

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        assert self.min_val is not None
        assert self.max_val is not None
        assert self.count_vec is not None

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def combine(
        self, suff_stat: Tuple[Optional[int], Optional[np.ndarray]]
    ) -> "IntegerCategoricalAccumulator":
        """Merge a ``(minimum, count_vector)`` statistic."""
        suff_min, suff_count = suff_stat
        if suff_min is None or suff_count is None:
            return self

        if self.count_vec is None:
            self.min_val = suff_min
            self.max_val = suff_min + len(suff_count) - 1
            self.count_vec = suff_count

        elif self.count_vec is not None:
            assert self.min_val is not None
            assert self.max_val is not None
            if self.min_val == suff_min and len(self.count_vec) == len(suff_count):
                self.count_vec += suff_count

            else:
                min_val = min(self.min_val, suff_min)
                max_val = max(self.max_val, suff_min + len(suff_count) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_min - min_val
                i1 = (suff_min + len(suff_count) - 1) - min_val + 1
                count_vec[i0:i1] += suff_count

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> Tuple[int, np.ndarray]:
        """Return ``(minimum, count_vector)`` sufficient statistics."""
        assert self.min_val is not None
        assert self.count_vec is not None
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> "IntegerCategoricalAccumulator":
        """Restore ``(minimum, count_vector)`` sufficient statistics."""
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics under the configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())

            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from the configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Create the compatible integer encoder."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer categorical accumulators.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical, if None estimated
            from data.
        max_val (Optional[int]): Maximum value of integer categorical, if None estimated
            from data.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Key used for accumulating merging statistics of
            IntegerCategoricalAccumulator objects.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Store support and metadata copied to each accumulator.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for accumulating merging statistics of
                IntegerCategoricalAccumulator objects.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.keys = keys

    def make(self) -> "IntegerCategoricalAccumulator":
        """Create an empty integer categorical accumulator."""
        return IntegerCategoricalAccumulator(
            min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name
        )


class IntegerCategoricalEstimator(ParameterEstimator):
    """Estimate an integer categorical distribution from weighted counts.

    Notes:
        Must set either min_val and max_val, or suff_stat must be passed as arg.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical.
        max_val (Optional[int]): Maximum value of integer categorical.
        pseudo_count (Optional[float]): Used to re-weight suff_stat when merged with new
            aggregated data.
        suff_stat (Tuple[int, np.ndarray]): min value and prob vec
        name (Optional[str]): Name to IntegerCategoricalEstimator object.
        keys (Optional[str]): Keys for accumulating merging statistics of
            IntegerCategoricalAccumulator objects.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[int, np.ndarray]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize support bounds, smoothing, and metadata.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            pseudo_count (Optional[float]): Used to re-weight suff_stat member variables
                in merging of sufficient
                statistics
            suff_stat: Set sufficient statistics. See above for details.
            name (Optional[str]): Assign a name to IntegerCategoricalEstimator object.
            keys (Optional[str]): Set keys for accumulating merging statistics of
                IntegerCategoricalAccumulator objects.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError(
                "IntegerCategoricalEstimator requires keys to be of type 'str'."
            )

        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IntegerCategoricalAccumulatorFactory":
        """Create a compatible accumulator factory."""
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        return IntegerCategoricalAccumulatorFactory(
            min_val=min_val, max_val=max_val, name=self.name, keys=self.keys
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Optional[Tuple[int, np.ndarray]]
    ) -> "IntegerCategoricalDistribution":
        """Estimate probabilities from ``(minimum, count_vector)`` statistics."""
        assert suff_stat is not None

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(
                suff_stat[0],
                (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs,
                name=self.name,
            )

        if (
            self.pseudo_count is not None
            and self.min_val is not None
            and self.max_val is not None
        ):

            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(
                min_val,
                (count_vec + pseudo_count_per_level) / adjusted_nobs,
                name=self.name,
            )

        if self.pseudo_count is not None and self.suff_stat is not None:

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

            return IntegerCategoricalDistribution(
                min_val, count_vec / (count_vec.sum()), name=self.name
            )

        return IntegerCategoricalDistribution(
            suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name
        )


class IntegerCategoricalDataEncoder(DataSequenceEncoder):
    """Encode integer observations as a one-dimensional NumPy array."""

    def __str__(self) -> str:
        """Returns IntegerCategoricalDataEncoder object for encoding data sequences."""
        return "IntegerCategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerCategoricalDataEncoder, False is else."""
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(
        self, x: Union[List[int], np.ndarray]
    ) -> "IntegerCategoricalEncodedDataSequence":
        """Encode ``N`` integers as an array of shape ``(N,)``."""
        return IntegerCategoricalEncodedDataSequence(data=np.asarray(x, dtype=int))


class IntegerCategoricalEncodedDataSequence(EncodedDataSequence):
    """Contain an encoded one-dimensional sequence of integers.

    Attributes:
        data (np.ndarray): IID observations from integer categorical distribution.

    """

    def __init__(self, data: np.ndarray):
        """Store an integer array of shape ``(N,)``.

        Args:
            data (np.ndarray): IID observations from integer categorical distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded array."""
        return f"IntegerCategoricalEncodedDataSequence(data={self.data})"
