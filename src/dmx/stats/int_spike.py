"""Spike-and-uniform-slab distributions on a finite integer range.

``SpikeAndSlabDistribution`` gives one integer ``k`` mass ``p`` and divides the
remaining mass uniformly among the other configured integers. Encoded sequences
are one-dimensional integer arrays; accumulators use contiguous count vectors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class SpikeAndSlabDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a point spike and uniform slab over integers.

    Sampling uses integers in ``[min_val, min_val + num_vals)``. For legacy
    compatibility, scalar and sequence scoring also treat the cached endpoint
    ``min_val + num_vals`` as an in-range slab value.

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

    def __init__(
        self,
        k: int,
        num_vals: int,
        p: float,
        min_val: Optional[int] = 0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize the spike, support, and component probability.

        Args:
            k (int): Integer value to place spike on. Must be within
                [min_val,min_val+num_vals)
            num_vals (int): Number of integers in the range.
            p (float): Probability of drawing k. (1-p)/(num_vals-1) to draw any other
                integer in range.
            min_val (Optional[int]): Defaults to 0. Set bottom of integer range.
            name (Optional[str]): Set name for object.
            keys (Optional[str]): Key for parameters.

        """
        super().__init__()
        self.p = p
        self.min_val = 0 if min_val is None else min_val
        self.max_val = self.min_val + num_vals

        if not self.min_val <= k <= self.max_val:
            raise ValueError(
                f"Spike value k must be between [{repr(self.min_val)}, "
                f"{repr(self.max_val)}]."
            )
        self.k = k

        self.log_p = np.log(p)
        self.num_vals = num_vals
        self.log_1p = np.log1p(-self.p) - np.log(self.num_vals - 1)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        s1 = str(self.min_val)
        s2 = str(self.num_vals)
        s3 = repr(self.p)
        s4 = repr(self.k)
        s5 = repr(self.name)
        s6 = repr(self.keys)

        return (
            f"SpikeAndSlabDistribution(p={s3}, min_val={s1}, num_vals={s2},k={s4}, "
            f"name={s5}, keys={s6})"
        )

    def density(self, x: int) -> float:
        """Evaluate the probability mass at integer ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        """Evaluate the log-probability mass at integer ``x``."""
        if self.max_val >= x >= self.min_val:
            return float(self.log_p if x == self.k else self.log_1p)
        return -np.inf

    def seq_log_density(self, x: "SpikeAndSlabEncodedDataSequence") -> np.ndarray:
        """Evaluate log masses for an encoded array of ``N`` integers."""
        if not isinstance(x, SpikeAndSlabEncodedDataSequence):
            raise TypeError(
                "SpikeAndSlabEncodedDataSequence required for seq_log_density()."
            )

        rv = np.zeros(len(x.data), dtype=float)
        rv.fill(-np.inf)

        in_range = np.bitwise_and(x.data >= self.min_val, x.data <= self.max_val)
        in_range_k = x.data[in_range] == self.k

        rv1 = rv[in_range]
        rv1[in_range_k] = self.log_p
        rv1[~in_range_k] = self.log_1p
        rv[in_range] = rv1

        return rv

    def sampler(self, seed: Optional[int] = None) -> "SpikeAndSlabSampler":
        """Create a sampler for this distribution."""
        return SpikeAndSlabSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "SpikeAndSlabEstimator":
        """Create an estimator retaining support and optional smoothing."""
        if pseudo_count is None:
            return SpikeAndSlabEstimator(
                min_val=self.min_val,
                max_val=self.max_val,
                name=self.name,
                keys=self.keys,
            )

        return SpikeAndSlabEstimator(
            min_val=self.min_val,
            max_val=self.max_val,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "SpikeAndSlabDataEncoder":
        """Create the compatible integer encoder."""
        return SpikeAndSlabDataEncoder()


class SpikeAndSlabSampler(DistributionSampler):
    """Draw integers from a spike-and-uniform-slab distribution.

    Attributes:
        rng (RandomState): RandomState for seeding samples.
        dist (SpikeAndSlabDistribution): SpikeAndSlabDistribution to sample from.
        non_k (np.ndarray): All integers outside of the spiked value 'k'.
    """

    def __init__(
        self, dist: "SpikeAndSlabDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler for ``dist``.

        Args:
            dist (SpikeAndSlabDistribution): SpikeAndSlabDistribution to sample from.
            seed (Optional[int]): Seed for generating samples.

        """
        super().__init__(dist, seed)
        self.non_k = np.delete(
            np.arange(self.dist.min_val, self.dist.max_val), self.dist.k
        )

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Draw one integer or an integer array of shape ``(size,)``."""
        if size is None:
            z = self.rng.binomial(n=1, p=self.dist.p)
            if z == 1:
                return int(self.dist.k)
            return int(self.rng.choice(self.non_k))

        rv = np.zeros(size, dtype=int)
        rv.fill(self.dist.k)
        z = self.rng.binomial(n=1, p=self.dist.p, size=size)
        idx = np.flatnonzero(z == 0)

        if len(idx) > 0:
            rv[idx] = self.rng.choice(self.non_k, replace=True, size=len(idx))

        return rv


class SpikeAndSlabAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted counts over a contiguous integer range.

    The sufficient statistic is ``(minimum, count_vector)``. Entry ``i`` of the
    one-dimensional vector counts integer ``minimum + i``.

    Attributes:
        min_val (Optional[int]): Smallest integer value in the range. Defaults to 0.
        max_val (Optional[int]): Set to the min val plus number of values - 1.
        count_vec (Optional[np.ndarray]): suff stat, counts for each numeric value.
        count (float): Weighted obs count.
        keys (Optional[str]): Set keys for object instance.
        name (Optional[str]): Set name for object instance.

    """

    def __init__(
        self,
        min_val: Optional[int],
        max_val: Optional[int],
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an optional fixed range and zero counts.

        Args:
            min_val (Optional[int]): Smallest integer value in the range. Defaults to 0.
            max_val (Optional[int]): Set to the min val plus number of values - 1.
            num_vals (Optional[
            keys (Optional[str]): Set keys for object instance.
            name (Optional[str]): Set name for object instance.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.count_vec: Optional[np.ndarray]

        if self.min_val is not None and self.max_val is not None:
            self.count_vec = np.zeros(self.max_val - self.min_val + 1, dtype=float)
        else:
            self.count_vec = None

        self.count = 0.0
        self.key = keys
        self.name = name

    def update(
        self, x: int, weight: float, estimate: Optional["SpikeAndSlabDistribution"]
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
        x: "SpikeAndSlabEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Add encoded observations during randomized initialization."""
        return self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "SpikeAndSlabEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["SpikeAndSlabDistribution"],
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

    def combine(self, suff_stat: Tuple[int, np.ndarray]) -> "SpikeAndSlabAccumulator":
        """Merge a ``(minimum, count_vector)`` statistic."""
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            assert self.min_val is not None
            assert self.max_val is not None
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(
                suff_stat[1]
            ):
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
        """Return ``(minimum, count_vector)`` sufficient statistics."""
        assert self.min_val is not None
        assert self.count_vec is not None
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> "SpikeAndSlabAccumulator":
        """Restore ``(minimum, count_vector)`` sufficient statistics."""
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics under the configured key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from the configured key."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "SpikeAndSlabDataEncoder":
        """Create the compatible integer encoder."""
        return SpikeAndSlabDataEncoder()


class SpikeAndSlabAccumulatorFactory(StatisticAccumulatorFactory):
    """Create spike-and-slab accumulators.

    Attributes:
            min_val (int]): Smallest integer value in the range. Defaults to 0.
            max_val (int): Set to the min val plus number of values - 1.
            keys (Optional[str]): Set keys for object instance.
            name (Optional[str]): Set name for object instance.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Store support and metadata copied to each accumulator.

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

    def make(self) -> "SpikeAndSlabAccumulator":
        """Create an empty spike-and-slab accumulator."""
        return SpikeAndSlabAccumulator(
            min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name
        )


class SpikeAndSlabEstimator(ParameterEstimator):
    """Estimate a spike location and mass from contiguous integer counts.

    Attributes:
        pseudo_count (Optional[float]): Regularize value k.
        min_val (int): Smallest integer value in the range. Defaults to 0.
        max_val (int): Set to the min val plus number of values - 1.
        suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and
            optional value of p for k.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Set keys for object instance.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[int, Optional[float]]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize support, pseudo-count settings, and metadata.

        Args:
            min_val (Optional[int]): Smallest integer value in the range.
            max_val (Optional[int]): Largest represented integer value.
            pseudo_count (Optional[float]): Regularize value k.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize
                and optional value of p for k.
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

    def accumulator_factory(self) -> "SpikeAndSlabAccumulatorFactory":
        """Create a compatible accumulator factory."""
        return SpikeAndSlabAccumulatorFactory(
            min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray]
    ) -> "SpikeAndSlabDistribution":
        """Estimate a distribution from ``(minimum, count_vector)``."""
        min_val, count_vec = suff_stat

        with np.errstate(divide="ignore"):
            if self.pseudo_count is None:
                count = np.sum(count_vec)
                p_vec = count_vec / count
                ll = np.log1p(-p_vec)
                ll -= np.log(len(count_vec) - 1)
                ll *= count - count_vec
                ll += count_vec * np.log(p_vec)
                k = np.argmax(ll)
                p = p_vec[k]

                return SpikeAndSlabDistribution(
                    k=k if min_val is None else k + min_val,
                    min_val=min_val,
                    num_vals=len(count_vec),
                    p=p,
                    name=self.name,
                )

            if self.pseudo_count is not None:
                if self.suff_stat[0] is not None and self.suff_stat[1] is None:
                    k_pseudo = (
                        self.suff_stat[0]
                        if min_val is None
                        else self.suff_stat[0] - min_val
                    )
                    count_vec[k_pseudo] += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= count - count_vec
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return SpikeAndSlabDistribution(
                        k=k if min_val is None else k + min_val,
                        min_val=min_val,
                        num_vals=len(count_vec),
                        p=p,
                        name=self.name,
                    )

                if self.suff_stat[0] is not None and self.suff_stat[1] is not None:
                    k_pseudo = (
                        self.suff_stat[0]
                        if min_val is None
                        else self.suff_stat[0] - min_val
                    )
                    count_vec[k_pseudo] += self.pseudo_count * self.suff_stat[1]
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= count - count_vec
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return SpikeAndSlabDistribution(
                        k=k if min_val is None else k + min_val,
                        min_val=min_val,
                        num_vals=len(count_vec),
                        p=p,
                        name=self.name,
                    )
                count_vec += self.pseudo_count
                count = np.sum(count_vec)
                p_vec = count_vec / count
                ll = np.log1p(-p_vec)
                ll -= np.log(len(count_vec) - 1)
                ll *= count - count_vec
                ll += count_vec * np.log(p_vec)
                k = np.argmax(ll)
                p = p_vec[k]

                return SpikeAndSlabDistribution(
                    k=k if min_val is None else k + min_val,
                    min_val=min_val,
                    num_vals=len(count_vec),
                    p=p,
                    name=self.name,
                )

        return None


class SpikeAndSlabDataEncoder(DataSequenceEncoder):
    """Encode integer observations as a one-dimensional NumPy array."""

    def __str__(self) -> str:
        """Return the legacy stable encoder name."""
        return "IntegerCategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has spike-and-slab semantics."""
        return isinstance(other, SpikeAndSlabDataEncoder)

    def seq_encode(
        self, x: Union[List[int], np.ndarray]
    ) -> "SpikeAndSlabEncodedDataSequence":
        """Encode ``N`` integers as an array of shape ``(N,)``."""
        return SpikeAndSlabEncodedDataSequence(data=np.asarray(x, dtype=int))


class SpikeAndSlabEncodedDataSequence(EncodedDataSequence):
    """Contain an encoded one-dimensional sequence of integers.

    Attributes:
        data (np.ndarray): Encoded sequence of integer values.

    """

    def __init__(self, data: np.ndarray):
        """Store an integer array of shape ``(N,)``.

        Args:
            data (np.ndarray): Encoded sequence of integer values.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded array."""
        return f"SpikeAndSlabEncodedDataSequence(data={self.data})"
