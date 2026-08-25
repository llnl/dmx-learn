"""Bernoulli distributions over subsets of a finite integer support.

``IntegerBernoulliSetDistribution`` models independent inclusion indicators for
integers ``0`` through ``K - 1``. An observation is a sequence or one-dimensional
array containing the included integers. The encoder flattens ``N`` such sets into
observation indices and integer values for vectorized evaluation and accumulation.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import exp
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class IntegerBernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Represent independent inclusions on integers ``[0, K)``.

    ``log_pvec[k]`` is the log probability that integer ``k`` is present.
    ``log_nvec[k]``, when supplied, is the log probability that it is absent;
    otherwise absence probabilities are derived as ``log(1 - exp(log_pvec))``.

    Attributes:
        name (Optional[str]): Name for object instance.
        log_pvec (np.ndarray): Shape ``(K,)`` log inclusion probabilities.
        log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional log
            absence probabilities of shape ``(K,)``.
        log_dvec (np.ndarray): Shape ``(K,)`` inclusion log-odds.
        log_nsum (float): Sum of finite log absence probabilities.
        keys (Optional[str]): Key for tying sufficient statistics.

    """

    def __init__(
        self,
        log_pvec: Union[Sequence[float], np.ndarray],
        log_nvec: Optional[Union[Sequence[float], np.ndarray]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize an integer Bernoulli set distribution.

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Shape ``(K,)`` log
                inclusion probabilities, indexed directly by integers ``0`` to
                ``K - 1``.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional
                Shape ``(K,)`` log absence probabilities.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        super().__init__()
        num_vals = len(log_pvec)
        self.name = name
        self.num_vals = num_vals
        self.log_pvec = np.asarray(log_pvec, dtype=np.float64).copy()
        self.keys = keys

        if log_nvec is None:
            log_nvec_arr = np.log1p(-np.exp(self.log_pvec))
            self.log_nvec = None
            self.log_dvec = self.log_pvec - log_nvec_arr
            self.log_nsum = np.sum(log_nvec_arr[np.isfinite(log_nvec_arr)])
        else:
            self.log_nvec = np.asarray(log_nvec, dtype=np.float64)
            self.log_dvec = self.log_pvec - self.log_nvec
            self.log_nsum = np.sum(self.log_nvec[np.isfinite(self.log_nvec)])

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        s1 = repr(self.log_pvec.tolist())
        s2 = repr(None if self.log_nvec is None else self.log_nvec.tolist())
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return (
            f"IntegerBernoulliSetDistribution({s1}, log_nvec={s2}, "
            f"name={s3}, keys={s4})"
        )

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Evaluate the probability mass of one integer set."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Evaluate the log-probability mass of one integer set."""
        xx = np.asarray(x, dtype=int)
        return float(np.sum(self.log_dvec[xx]) + self.log_nsum)

    def seq_log_density(
        self, x: "IntegerBernoulliSetEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate log masses for ``N`` encoded set observations.

        Returns:
            Array of shape ``(N,)`` with one log mass per set.
        """
        if not isinstance(x, IntegerBernoulliSetEncodedDataSequence):
            raise TypeError(
                "IntegerBernoulliSetEncodedDataSequence required for seq_log_density()."
            )

        sz, idx, xs = x.data
        rv = np.zeros(sz, dtype=np.float64)
        rv += np.bincount(idx, weights=self.log_dvec[xs], minlength=sz)
        rv += self.log_nsum
        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerBernoulliSetSampler":
        """Create a sampler for this distribution."""
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerBernoulliSetEstimator":
        """Create an estimator initialized with this support size."""
        return IntegerBernoulliSetEstimator(
            self.num_vals, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Create the compatible set-data encoder."""
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetSampler(DistributionSampler):
    """Draw independent integer sets from a Bernoulli set distribution.

    Attributes:
        rng (RandomState): RandomState object with seed set if passed in args.
        dist (IntegerBernoulliSetDistribution): Object instance to sample from.

    """

    def __init__(
        self, dist: IntegerBernoulliSetDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler for ``dist``.

        Args:
            dist (IntegerBernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        super().__init__(dist, seed)

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Sequence[int]], Sequence[int]]:
        """Draw one integer set or a list of ``size`` sets."""
        if size is None:
            log_u = np.log(self.rng.rand(self.dist.num_vals))
            return [int(v) for v in np.flatnonzero(log_u <= self.dist.log_pvec)]
        rv: List[Sequence[int]] = []
        for _ in range(size):
            log_u = np.log(self.rng.rand(self.dist.num_vals))
            rv.append([int(v) for v in np.flatnonzero(log_u <= self.dist.log_pvec)])
        return rv


class IntegerBernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted inclusion counts over integers ``[0, K)``.

    The sufficient statistic is ``(positive_counts, total_weight)``.
    ``positive_counts`` has shape ``(K,)`` and entry ``k`` is the weighted
    number of sets containing integer ``k``; ``total_weight`` is the weighted
    number of observed sets.

    Attributes:
       pcnt (np.ndarray): Used for aggregating weighted counts of integers.
       keys (Optional[str]): Keys for merging sufficient statistics with matching key'd
           objects.
       num_vals (int): Number of values in integer range for the set.
       tot_sum (float): Sum of weights for observations.
       name (Optional[str]): Name for object.


    """

    def __init__(
        self, num_vals: int, keys: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """Initialize zero inclusion counts for ``num_vals`` integers.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching
                key'd objects.
            name (Optional[str]): Name for object.

        """
        self.pcnt = np.zeros(num_vals, dtype=np.float64)
        self.keys = keys
        self.name = name
        self.num_vals = num_vals
        self.tot_sum = 0.0

    def update(
        self,
        x: Union[Sequence[int], np.ndarray],
        weight: float,
        estimate: Optional[IntegerBernoulliSetDistribution],
    ) -> None:
        """Add one weighted integer-set observation."""
        xx = np.asarray(x, dtype=int)
        self.pcnt[xx] += weight
        self.tot_sum += weight

    def initialize(
        self,
        x: Union[Sequence[int], np.ndarray],
        weight: float,
        rng: Optional[RandomState],
    ) -> None:
        """Add one observation during randomized initialization."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: "IntegerBernoulliSetEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[IntegerBernoulliSetDistribution],
    ) -> None:
        """Add ``N`` encoded sets with a weight array of shape ``(N,)``."""
        _sz, idx, xs = x.data
        agg_cnt = np.bincount(xs, weights=weights[idx])
        n = len(agg_cnt)
        self.pcnt[:n] += agg_cnt
        self.tot_sum += weights.sum()

    def seq_initialize(
        self,
        x: "IntegerBernoulliSetEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Add encoded sets during randomized initialization."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        """Merge ``(positive_counts, total_weight)`` statistics."""
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[np.ndarray, float]:
        """Return ``(positive_counts, total_weight)``."""
        return self.pcnt, self.tot_sum

    def from_value(
        self, x: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        """Restore ``(positive_counts, total_weight)`` statistics."""
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics into ``stats_dict`` under the configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                temp = stats_dict[self.keys]
                stats_dict[self.keys] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.keys] = (self.pcnt, self.tot_sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from ``stats_dict`` when the key is present."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.keys]

    def acc_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Create the compatible set-data encoder."""
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer Bernoulli set accumulators.

    Attributes:
        keys (Optional[str]): Keys for merging sufficient statistics with matching key'd
            objects.
        num_vals (int): Number of values in integer range for the set.
        name (Optional[str]): Name for object.

    """

    def __init__(
        self, num_vals: int, keys: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """Store settings copied to each accumulator.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with matching
                key'd objects.
            num_vals (int): Number of values in integer range for the set.
            name (Optional[str]): Name for object.

        """
        self.keys = keys
        self.num_vals = num_vals
        self.name = name

    def make(self) -> "IntegerBernoulliSetAccumulator":
        """Create an empty integer Bernoulli set accumulator."""
        return IntegerBernoulliSetAccumulator(
            self.num_vals, keys=self.keys, name=self.name
        )


class IntegerBernoulliSetEstimator(ParameterEstimator):
    """Estimate Bernoulli inclusion probabilities from weighted counts.

    Attributes:
        num_vals (int): Number of values in integer range for the set.
        keys (Optional[str]): Keys for merging sufficient statistics with matching key'd
            objects.
        pseudo_count (Optional[float]): Re-weight suff stats in estimation.
        suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
        name (Optional[str]): Set name for object instance.
        min_prob (float): Minimum probability for an integer in range of set dist.

    """

    def __init__(
        self,
        num_vals: int,
        min_prob: float = 1.0e-128,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize estimator support, smoothing, and metadata.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in range of set dist.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Keys for merging sufficient statistics with matching
                key'd objects.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError(
                "IntegerBernoulliSetEstimator requires keys to be of type 'str'."
            )

        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> "IntegerBernoulliSetAccumulatorFactory":
        """Create a compatible accumulator factory."""
        return IntegerBernoulliSetAccumulatorFactory(
            self.num_vals, keys=self.keys, name=self.name
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Optional[np.ndarray] = None
    ) -> "IntegerBernoulliSetDistribution":
        """Estimate log inclusion and absence probabilities.

        Args:
            nobs: Ignored legacy observation count.
            suff_stat: ``(positive_counts, total_weight)`` statistic, where
                ``positive_counts`` has shape ``(K,)``.

        Returns:
            Fitted distribution on integers ``[0, K)``.
        """
        assert suff_stat is not None
        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = self.suff_stat * self.pseudo_count
            p1 = np.subtract(1.0, self.suff_stat) * self.pseudo_count
            pvec = np.log(suff_stat[0] + p0)
            nvec = np.log((suff_stat[1] - suff_stat[0]) + p1)
            tsum = np.log(suff_stat[1] + self.pseudo_count)
            log_pvec = np.log(pvec) - tsum
            log_nvec = np.log(nvec) - tsum

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            log_c = np.log(suff_stat[1] + p)
            log_pvec = np.log(suff_stat[0] + (p / 2.0)) - log_c
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + (p / 2.0)) - log_c

        else:

            if suff_stat[1] == 0:
                log_pvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5
                log_nvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5

            elif self.min_prob > 0:
                log_pvec = np.log(
                    np.maximum(suff_stat[0] / suff_stat[1], self.min_prob)
                )
                log_nvec = np.log(
                    np.maximum(
                        (suff_stat[1] - suff_stat[0]) / suff_stat[1], self.min_prob
                    )
                )

            else:
                pvec = suff_stat[0] / suff_stat[1]
                nvec = (suff_stat[1] - suff_stat[0]) / suff_stat[1]

                is_zero = pvec == 0
                is_one = nvec == 0

                log_pvec = np.zeros(self.num_vals, dtype=np.float64)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64)

                log_pvec[~is_zero] = np.log(pvec[~is_zero])
                log_pvec[is_zero] = -np.inf
                log_nvec[~is_one] = np.log(nvec[~is_one])
                log_nvec[is_one] = -np.inf

        return IntegerBernoulliSetDistribution(log_pvec, log_nvec, name=self.name)


class IntegerBernoulliSetDataEncoder(DataSequenceEncoder):
    """Encode sequences of integer-set observations into flattened arrays."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "IntegerBernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same semantics."""
        return isinstance(other, IntegerBernoulliSetDataEncoder)

    def seq_encode(
        self, x: Sequence[Sequence[int]]
    ) -> "IntegerBernoulliSetEncodedDataSequence":
        """Encode ``N`` integer sets for vectorized calculations.

        Args:
            x (Sequence[Sequence[int]]): ``N`` set observations, each represented
                by its included integer values.

        Returns:
            Encoded tuple ``(N, indices, values)``. The two arrays have shape
            ``(M,)``, where ``M`` is the total number of included values;
            ``indices[j]`` identifies the source set for ``values[j]``.

        """
        idx: List[int] = []
        xs: List[int] = []
        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx_arr = np.asarray(idx, dtype=np.int32)
        xs_arr = np.asarray(xs, dtype=np.int32)

        return IntegerBernoulliSetEncodedDataSequence(data=(len(x), idx_arr, xs_arr))


class IntegerBernoulliSetEncodedDataSequence(EncodedDataSequence):
    """Contain flattened integer Bernoulli set observations.

    Attributes:
        data (Tuple[int, np.ndarray, np.ndarray]): Encoded Bernoulli Set observations.

    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray]):
        """Store an ``(N, indices, values)`` encoded tuple.

        Args:
            data (Tuple[int, np.ndarray, np.ndarray]): Encoded Bernoulli Set
                observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded tuple."""
        return f"IntegerBernoulliSetEncodedDataSequence(data={self.data})"
