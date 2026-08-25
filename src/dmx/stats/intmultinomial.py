"""Multinomial count distributions over contiguous integer categories.

An observation is a sparse sequence of ``(integer, count)`` pairs. Entry ``i``
of a probability vector of shape ``(K,)`` corresponds to category
``min_val + i``. An optional length distribution models the sum of the counts.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import inf, maxrandint
from dmx.stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

SS0 = TypeVar("SS0")
D = Sequence[Tuple[int, float]]
E0 = TypeVar("E0")
E = Tuple[int, np.ndarray, np.ndarray, np.ndarray, Optional[E0]]


class IntegerMultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Represent multinomial counts on consecutive integer categories.

    This implementation scores the product of category probabilities raised to
    their counts, together with the total-count distribution. It intentionally
    omits the multinomial combinatorial coefficient.

    Attributes:
        p_vec (ndarray): Probability of each integer category for a trial.
        min_val (int): Smallest integer value for category range. Defaults to 0.
        max_val (int): Largest value of category range. Set by min_val + len(p_vec) - 1.
        log_p_vec (ndarray): Log of p_vec member instance.
        num_vals (int): Total number of integer valued categories.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution for number of
            trials. Set to
            NullDistribution if None.
        keys (Optional[str]): Keys for distribution passed when ParameterEstimator is
            created.
        name (Optional[str]): Name for object instance.

    """

    def __init__(
        self,
        min_val: int = 0,
        p_vec: Optional[Union[List[float], np.ndarray]] = None,
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize category probabilities and the total-count model.

        Args:
            min_val (int): Set the minimum value on range of values.
            p_vec (Union[List[float],np.ndarray): Probabilities for values. Length
                determines number of categories.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                length distributions serving as
                for the number of trials.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for distribution.

        """
        super().__init__()
        p_vec = np.empty(0, dtype=np.float64) if p_vec is None else p_vec

        with np.errstate(divide="ignore"):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.keys = keys
            self.name = name

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        s1 = repr(self.min_val)
        s2 = repr(self.p_vec.tolist())
        s3 = str(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return (
            f"IntegerMultinomialDistribution({s1}, {s2}, len_dist={s3}, name={s4}, "
            f"keys={s5})"
        )

    def density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the density of IntegerMultinomialDistribution at observed value x.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer
                category value and number of
                successes.

        Returns:
            float: Density at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the log mass of one sparse count observation.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer
                category value and number of
                successes.

        Returns:
            float: Log-density at x.

        """
        rv = 0.0
        sz = 0.0
        for xx, cnt in x:
            rv += (
                -inf
                if (xx < self.min_val or xx > self.max_val)
                else self.log_p_vec[xx - self.min_val]
            ) * cnt
            sz += cnt
        return rv + self.len_dist.log_density(sz)

    def seq_log_density(self, x: "IntegerMultinomialEncodedDataSequence") -> np.ndarray:
        """Evaluate log masses for ``N`` encoded count observations."""
        if not isinstance(x, IntegerMultinomialEncodedDataSequence):
            raise TypeError(
                "IntegerMultinomialEncodedDataSequence required for seq_log_density()."
            )
        sz, idx, cnt, val, tcnt = x.data

        v = val - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(v))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]
        rv[u] *= cnt[u]
        ll = np.bincount(idx, weights=rv, minlength=sz)

        if tcnt is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll

    def sampler(self, seed: Optional[int] = None) -> "IntegerMultinomialSampler":
        """Create a sampler when a total-count distribution is configured."""
        if isinstance(self.len_dist, NullDistribution):
            raise RuntimeError(
                "IntegerMultinomialDistribution must have len_dist set to distribution "
                "with support on "
                "non-negative integers."
            )
        return IntegerMultinomialSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerMultinomialEstimator":
        """Create an estimator retaining support and optional smoothing."""
        len_est = (
            NullEstimator()
            if self.len_dist is None
            else self.len_dist.estimator(pseudo_count=pseudo_count)
        )

        if pseudo_count is None:
            return IntegerMultinomialEstimator(
                len_estimator=len_est, name=self.name, keys=self.keys
            )
        return IntegerMultinomialEstimator(
            min_val=self.min_val,
            max_val=self.max_val,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            suff_stat=(self.min_val, self.p_vec),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        """Create an encoder compatible with the total-count distribution."""
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialSampler(DistributionSampler):
    """Draw sparse integer-category count observations.

    Attributes:
        dist (IntegerMultinomialDistribution): IntegerMultinomialDistribution object
            instance to sample from.
        rng (RandomState): RandomState set with seed if passed.
        len_sampler (DistributionSampler): DistributionSampler object for number of
            trials.

    """

    def __init__(
        self, dist: IntegerMultinomialDistribution, seed: Optional[int] = None
    ) -> None:
        """Create IntegerMultinomialSampler object.

        Args:
            dist (IntegerMultinomialDistribution): IntegerMultinomialDistribution object
                instance to sample from.
            seed (Optional[int]): Optional seed for random number generator.

        """
        super().__init__(dist, seed)
        self.len_sampler = self.dist.len_dist.sampler(
            seed=self.rng.randint(0, maxrandint)
        )

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """Draw independent samples from an integer multinomial distribution.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            List of samples. If size is None, returns one sample as a
            List[Tuple[int, float]].

        """
        if size is None:
            cnt = int(self.len_sampler.sample())
            entry = self.rng.multinomial(cnt, self.dist.p_vec)
            rrv: List[Tuple[int, float]] = []
            for raw_j in np.flatnonzero(entry):
                j = int(raw_j)
                rrv.append((j + self.dist.min_val, int(entry[j])))
            return rrv

        cnt = self.len_sampler.sample(size=size)
        rv: List[List[Tuple[int, float]]] = []

        for i in range(size):
            rrv = []
            entry = self.rng.multinomial(int(cnt[i]), self.dist.p_vec)
            for raw_j in np.flatnonzero(entry):
                j = int(raw_j)
                rrv.append((j + self.dist.min_val, int(entry[j])))
            rv.append(rrv)
        return rv


class IntegerMultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate category counts and total-count statistics.

    The statistic is ``(minimum, count_vector, length_statistic)``. Entry ``i``
    of the one-dimensional count vector stores the weighted count for category
    ``minimum + i``; the final element belongs to the length accumulator.

    Attributes:
        min_val (Optional[int]): Minimum value for integer multinomial.
        max_val (Optional[int]): Maximum value for integer multinomial.
        name (Optional[str]): Name for object instance.
        len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional
            accumulator for number of
            integer multinomial trial counts. Set to NullAccumulator() if None.
        count_vec (Optional[ndarray]): Set counter for the number of values in integer
            multinomial range to zero
            ndarray if min_val and max_val are passed. Else, set to none.
        key (Optional[str]): Keys for merging sufficient stats with other objects
            containing matching key.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        len_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
    ) -> None:
        """Initialize category and length accumulators.

        Args:
            min_val (Optional[int]): Set minimum value for integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient stats with other
                objects containing matching keys.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional
                accumulator for number of
                integer multinomial trial counts.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator()
        )
        self.count_vec: Optional[np.ndarray] = (
            vec.zeros(max_val - min_val + 1)
            if min_val is not None and max_val is not None
            else None
        )
        self.keys = keys

    def update(
        self,
        x: Sequence[Tuple[int, float]],
        weight: float,
        estimate: Optional[IntegerMultinomialDistribution],
    ) -> None:
        """Add one weighted sparse count observation."""
        cc = 0.0
        for xx, cnt in x:
            cc += cnt
            if self.count_vec is None:
                self.min_val = xx
                self.max_val = xx
                self.count_vec = vec.make([weight * cnt])
                continue

            assert self.min_val is not None
            assert self.max_val is not None

            if self.max_val < xx:
                temp_vec = self.count_vec
                self.max_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[: len(temp_vec)] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            elif self.min_val > xx:
                temp_vec = self.count_vec
                temp_diff = self.min_val - xx
                self.min_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[temp_diff:] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            else:
                self.count_vec[xx - self.min_val] += weight * cnt

        if estimate is None:
            self.len_accumulator.update(cc, weight, None)
        else:
            self.len_accumulator.update(cc, weight, estimate.len_dist)

    def initialize(
        self, x: Sequence[Tuple[int, float]], weight: float, rng: Optional[RandomState]
    ) -> None:
        """Add one observation during randomized initialization."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: "IntegerMultinomialEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[IntegerMultinomialDistribution],
    ) -> None:
        """Add ``N`` encoded observations with weights of shape ``(N,)``."""
        _sz, idx, cnt, val, tenc = x.data

        min_x = int(val.min())
        max_x = int(val.max())

        loc_cnt = np.bincount(val - min_x, weights=cnt * weights[idx])

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

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.seq_update(tenc, weights, None)
            else:
                self.len_accumulator.seq_update(tenc, weights, estimate.len_dist)

    def seq_initialize(
        self,
        x: "IntegerMultinomialEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Add encoded observations during randomized initialization."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[int, np.ndarray, Optional[SS0]]
    ) -> "IntegerMultinomialAccumulator":
        """Merge category and length sufficient statistics."""
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

        self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[int, np.ndarray, Optional[Any]]:
        """Return ``(minimum, count_vector, length_statistic)``."""
        assert self.min_val is not None
        assert self.count_vec is not None
        return self.min_val, self.count_vec, self.len_accumulator.value()

    def from_value(
        self, x: Tuple[int, np.ndarray, Optional[SS0]]
    ) -> "IntegerMultinomialAccumulator":
        """Restore category and length sufficient statistics."""
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge category and length statistics through configured keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace category and length statistics through configured keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        """Create an encoder compatible with both accumulators."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer multinomial accumulators.

    Attributes:
        min_val (Optional[int]): Optional minimum value for
            IntegerMultinomialAccumulator.
        max_val (Optional[int]): Optional maximum value for
            IntegerMultinomialAccumulator.
        name (Optional[str]): Optional name for object instance.
        keys (Optional[str]): Optional keys for merging sufficient statistics of object
            instance.
        len_factory (Optional[StatisticAccumulatorFactory]): Optional
            StatisticAccumulatorFactory object for
            creating StatisticAccumulator object for number of trials. Default to
            NullAccumulatorFactory()

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
    ) -> None:
        """Store category and length settings copied to each accumulator.

        Args:
            min_val (Optional[int]): Optional minimum value for
                IntegerMultinomialAccumulator.
            max_val (Optional[int]): Optional maximum value for
                IntegerMultinomialAccumulator.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional keys for merging sufficient statistics of
                object instance.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional
                StatisticAccumulatorFactory object for
                creating StatisticAccumulator object for number of trials.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys

    def make(self) -> "IntegerMultinomialAccumulator":
        """Create an empty integer multinomial accumulator."""
        len_acc = self.len_factory.make()
        return IntegerMultinomialAccumulator(
            min_val=self.min_val,
            max_val=self.max_val,
            name=self.name,
            keys=self.keys,
            len_accumulator=len_acc,
        )


class IntegerMultinomialEstimator(ParameterEstimator):
    """Estimate category and total-count distributions from statistics.

    Attributes:
        min_val (Optional[int]): Set minimum value integer multinomial.
        max_val (Optional[int]): Set maximum value for integer multinomial.
        len_estimator (ParameterEstimator): ParameterEstimator for number of trials, set
            to NullEstimator() if None
            is passed as arg.
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
            SequenceEncodableProbabilityDistribution for fixing distribution on number
            of trials.
        name (Optional[str]): Set name for object instance.
        pseudo_count (Optional[float]): Used to re-weight sufficient statistics if
            suff_stat is passed.
        suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and counts for
            categories. If 'min_val'
            and 'max_val' are both not None, this is ignored in estimation.
        keys (Optional[str]): Set key for merging sufficient statistics of objects with
            matching keys.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
        name: Optional[str] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[int, np.ndarray]] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize support, smoothing, length estimation, and metadata.

        Args:
            min_val (Optional[int]): Set minimum value integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator
                for number of trials.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution for fixing distribution on
                number of trials.
            name (Optional[str]): Set name for object instance.
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics if
                suff_stat is passed.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and counts
                for categories.
            keys (Optional[str]): Set key for merging sufficient statistics of objects
                with matching keys.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError(
                "IntegerMultinomialEstimator requires keys to be of type 'str'."
            )

        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IntegerMultinomialAccumulatorFactory":
        """Create a compatible accumulator factory."""
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        len_factory = self.len_estimator.accumulator_factory()
        return IntegerMultinomialAccumulatorFactory(
            min_val=min_val,
            max_val=max_val,
            name=self.name,
            keys=self.keys,
            len_factory=len_factory,
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray, Optional[SS0]]
    ) -> "IntegerMultinomialDistribution":
        """Estimate a model from category and length sufficient statistics."""
        len_dist = (
            self.len_dist
            if self.len_dist is not None
            else self.len_estimator.estimate(nobs, suff_stat[2])
        )

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(
                suff_stat[0],
                (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs,
                len_dist=len_dist,
                name=self.name,
                keys=self.keys,
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

            return IntegerMultinomialDistribution(
                min_val,
                (count_vec + pseudo_count_per_level) / adjusted_nobs,
                len_dist=len_dist,
                name=self.name,
                keys=self.keys,
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

            return IntegerMultinomialDistribution(
                min_val,
                count_vec / (count_vec.sum()),
                len_dist=len_dist,
                name=self.name,
                keys=self.keys,
            )
        return IntegerMultinomialDistribution(
            suff_stat[0],
            suff_stat[1] / (suff_stat[1].sum()),
            len_dist=len_dist,
            name=self.name,
            keys=self.keys,
        )


class IntegerMultinomialDataEncoder(DataSequenceEncoder):
    """Encode sparse integer multinomial observations into flat arrays.

    Attributes:
        len_encoder (DataSequenceEncoder): DataSequenceEncoder for encoding number of
            trials in each iid
            observation of integer multinomial. Defaults to NullDataEncoder() if None is
            passed.

    """

    def __init__(
        self, len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()
    ) -> None:
        """Initialize with an encoder for total counts.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder
                for encoding the number of trials
                in each iid observation of integer multinomial.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a representation containing the length encoder."""
        return (
            "IntegerMultinomialDataEncoder(len_encoder=" + str(self.len_encoder) + ")"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same length encoder."""
        if isinstance(other, IntegerMultinomialDataEncoder):
            return self.len_encoder == other.len_encoder
        return False

    def seq_encode(
        self, x: Sequence[Sequence[Tuple[int, float]]]
    ) -> "IntegerMultinomialEncodedDataSequence":
        """Encode ``N`` sparse count observations.

        Args:
            x (Sequence[Sequence[Tuple[int, float]]]): A sequence of iid integer
                multinomial observations in the form
                of Sequence of Tuple(s) containing integer-category and float valued
                number of successes.

        Returns:
            Encoded tuple ``(N, indices, counts, values, encoded_totals)``.
            The three flat arrays have shape ``(M,)``, where ``M`` is the
            total number of supplied category/count pairs. ``indices[j]`` maps
            ``counts[j]`` and ``values[j]`` to an input observation; encoded
            totals represent an array of shape ``(N,)``.

        """
        idx: List[int] = []
        cnt: List[float] = []
        val: List[int] = []
        tcnt: List[float] = []

        for i, y in enumerate(x):
            cc = 0.0
            for z in y:
                idx.append(i)
                cnt.append(z[1])
                val.append(z[0])
                cc += z[1]
            tcnt.append(cc)

        sz = len(x)
        idx_arr = np.asarray(idx, dtype=np.int32)
        cnt_arr = np.asarray(cnt, dtype=np.float64)
        val_arr = np.asarray(val, dtype=np.int32)
        tcnt_arr = np.asarray(tcnt, dtype=np.int32)

        tcnt_enc = self.len_encoder.seq_encode(tcnt_arr)

        return IntegerMultinomialEncodedDataSequence(
            data=(sz, idx_arr, cnt_arr, val_arr, tcnt_enc)
        )


class IntegerMultinomialEncodedDataSequence(EncodedDataSequence):
    """Contain flattened sparse integer multinomial observations.

    Attributes:
        data (E): Encoded sequence of integer multinomial observations.

    """

    def __init__(
        self, data: Tuple[int, np.ndarray, np.ndarray, np.ndarray, EncodedDataSequence]
    ):
        """Store an ``(N, indices, counts, values, encoded_totals)`` tuple.

        Args:
            data (E): Encoded sequence of integer multinomial observations.


        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded tuple."""
        return f"IntegerMultinomialEncodedDataSequence(data={self.data})"
