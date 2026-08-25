r"""Model one edit between two unordered subsets of a finite integer universe.

An observation is a pair ``(before, after)`` of collections representing sets of unique
integers from ``0`` through ``N - 1``. Collection order is ignored. For each integer,
the edit independently chooses absence or presence in ``after`` conditional on absence
or presence in ``before``. The joint probability is the product of those ``N``
transition probabilities and ``init_dist``'s probability for ``before``.

Rows of ``log_edit_pmat`` correspond to integers. Two-column input gives log
probabilities for ``present | absent`` and ``present | present``; their complements are
constructed. Four-column input is ordered as ``absent | absent``, ``absent | present``,
``present | absent``, and ``present | present``. The initial-set child distribution is
used independently for scoring, sampling, encoding, accumulation, and estimation.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import exp, maxrandint
from dmx.stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
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

T = Tuple[Union[Sequence[int], np.ndarray], Union[Sequence[int], np.ndarray]]
SS1 = TypeVar("SS1")  ## suff-stat of init_dist


class IntegerBernoulliEditDistribution(SequenceEncodableProbabilityDistribution):
    """Model a pair of unordered integer sets with independent edit transitions.

    Attributes:
        name (Optional[str]): Name for object.
        init_dist (SequenceEncodableProbabilityDistribution): Initial probability
            distribution.
        num_vals (int): Number of values in the distribution.
        orig_log_edit_pmat (np.ndarray): Original log probabilities for the edit matrix.
        log_edit_pmat (np.ndarray): Log probabilities for the edit matrix, expanded to 4
            columns.
        log_nsum (float): Logarithmic sum of probabilities for missing values.
        log_dvec (np.ndarray): Difference vector for log probabilities.
        keys (Optional[str]): Keys for parameters of distribution.
    """

    def __init__(
        self,
        log_edit_pmat: Union[Sequence[Tuple[float, float]], np.ndarray],
        init_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Defines the Integer Bernoulli Edit Set Distribution.

        Args:
            log_edit_pmat (Union[Sequence[Tuple[float, float]], np.ndarray]): Log
                probabilities for the edit matrix.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Initial
                distribution. Defaults to NullDistribution().
            name (Optional[str]): Name for object.
            keys (Optional[str]): Keys for parameters of distribution.
        """
        super().__init__()
        num_vals = len(log_edit_pmat)
        self.name = name
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.num_vals = num_vals

        pmat = np.asarray(log_edit_pmat, dtype=np.float64).copy()
        if pmat.shape[1] == 2:
            log_pmat = np.zeros((num_vals, 4), dtype=np.float64)
            log_pmat[:, 0] = np.log1p(
                -np.exp(pmat[:, 0])
            )  # p_mat(missing | missing) = 1 - p_mat(present | missing)
            log_pmat[:, 1] = np.log1p(
                -np.exp(pmat[:, 1])
            )  # p_mat(missing | present) = 1 - p_mat(present | present)
            log_pmat[:, 2] = pmat[:, 0]  # p_mat(present | missing)
            log_pmat[:, 3] = pmat[:, 1]  # p_mat(present | present)
        else:
            log_pmat = pmat

        self.orig_log_edit_pmat = pmat
        self.log_edit_pmat = log_pmat
        self.log_nsum = self.log_edit_pmat[
            np.isfinite(self.log_edit_pmat[:, 0]), 0
        ].sum()  # sum [ln p_mat(missing | missing)]
        self.log_dvec = (
            self.log_edit_pmat[:, 1:] - self.log_edit_pmat[:, 0, None]
        )  # ln p_mat (?? | ??) - ln p_mat(missing | missing)
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s1 = repr(list(map(list, self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist)
        s3 = repr(self.keys)
        s4 = repr(self.name)

        return (
            f"IntegerBernoulliEditDistribution({s1}, init_dist={s2}, keys={s3}, "
            f"name={s4})"
        )

    def density(self, x: T) -> float:
        """Evaluate the joint probability of a ``(before, after)`` set pair."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: T) -> float:
        """Evaluate the joint log probability of a ``(before, after)`` set pair."""
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        in10 = np.isin(xx1, xx0, invert=False)  # xx0 \cap xx1
        in01 = np.isin(xx0, xx1, invert=True)  # xx0 \cap xx1

        yy = np.ones(len(xx1), dtype=int)
        yy[in10] = 2
        rv = self.log_nsum  # ln p_mat(missing | missing) for the empty set
        rv += np.sum(
            self.log_dvec[xx1[in10], 2]
        )  # ln p_mat(present | present) same stuff that was there
        rv += np.sum(
            self.log_dvec[xx1[~in10], 1]
        )  # ln p_mat(present | missing) new additions
        rv += np.sum(
            self.log_dvec[xx0[in01], 0]
        )  # ln p_mat(missing | present) stuff to remove
        # rv = ln p_mat(x[1] | x[0])

        # rv = ln p_mat(x[1] | x[0]) + ln(p_mat(x[0]) = ln p_mat(x[0], x[1])
        rv += self.init_dist.log_density(x[0])

        return float(rv)

    def seq_log_density(
        self, x: "IntegerBernoulliEditEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate joint log probabilities for encoded set-pair observations."""
        if not isinstance(x, IntegerBernoulliEditEncodedDataSequence):
            raise TypeError(
                "IntegerBernoulliEditEncodedDataSequence required for "
                "seq_log_density()."
            )
        sz, idx, xs, ys, _ym, init_enc = x.data
        rv = np.bincount(idx, weights=self.log_dvec[xs, ys], minlength=sz)
        rv += self.log_nsum
        rv += self.init_dist.seq_log_density(init_enc)

        return np.asarray(rv, dtype=float)

    def sampler(self, seed: Optional[int] = None) -> "IntegerBernoulliEditSampler":
        """Create a sampler for initial sets and one-step edits."""
        return IntegerBernoulliEditSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerBernoulliEditEstimator":
        """Create an estimator and delegate initial-set estimation to the child."""
        return IntegerBernoulliEditEstimator(
            self.num_vals,
            init_estimator=self.init_dist.estimator(),
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Create an encoder using the initial distribution's child encoder."""
        return IntegerBernoulliEditDataEncoder(
            init_encoder=self.init_dist.dist_to_encoder()
        )


class IntegerBernoulliEditSampler(DistributionSampler):
    """Sampler for the Integer Bernoulli Edit Set Distribution.

    Attributes:
        rng (RandomState): Random number generator for sampling.
        dist (IntegerBernoulliEditDistribution): The distribution to sample from.
        init_rng (DistributionSampler): Initial sampler for the distribution.
        next_rng (RandomState): Random number generator for subsequent samples.
    """

    def __init__(
        self, dist: IntegerBernoulliEditDistribution, seed: Optional[int] = None
    ):
        """Sampler for the Integer Bernoulli Edit Set Distribution.

        Args:
            dist (IntegerBernoulliEditDistribution): The distribution to sample from.
            seed (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(dist, seed)
        self.init_rng = dist.init_dist.sampler(self.rng.randint(0, maxrandint))
        self.next_rng = np.random.RandomState(self.rng.randint(0, maxrandint))

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[List[int], List[int]]], Tuple[List[int], List[int]]]:
        """Draw one initial/edited set pair, or ``size`` independent pairs."""
        if size is None:
            temp = self.rng.rand(self.dist.num_vals)
            temp = np.log(temp)
            mask = np.zeros(self.dist.num_vals, dtype=bool)
            prev_ob = np.asarray(self.init_rng.sample(), dtype=int)

            mask[temp <= self.dist.log_edit_pmat[:, 2]] = True
            mask[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

            return list(prev_ob), list(np.flatnonzero(mask))
        rv: List[Tuple[List[int], List[int]]] = []
        for _ in range(size):
            sample = self.sample()
            assert isinstance(sample, tuple)
            rv.append(sample)
        return rv

    def sample_given(self, x: Sequence[Sequence[int]]) -> List[int]:
        """Samples from the distribution given a prior subset.

        Args:
            x (Sequence[Sequence[int]]): Prior subset.

        Returns:
            List[int]: Sampled subset.
        """
        temp = self.rng.rand(self.dist.num_vals)
        np.log(temp, out=temp)
        rv = np.zeros(self.dist.num_vals, dtype=bool)
        prev_ob = np.asarray(x[-1], dtype=int)

        rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

        return list(np.flatnonzero(rv))


class IntegerBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate removal, addition, retention, and initial-set statistics.

    For each integer, the three explicit columns count ``absent | present``,
    ``present | absent``, and ``present | present``. The complementary
    ``absent | absent`` count is derived from total observation weight. Initial sets
    are also forwarded to ``init_acc``.

    Attributes:
        pcnt (np.ndarray): Counts for sufficient statistics.
        keys (Optional[str]): Keys for parameters of distribution.
        num_vals (int): Number of values in the distribution.
        init_acc (SequenceEncodableStatisticAccumulator): Initial accumulator object.
        tot_sum (float): Total sum of weights.
    """

    def __init__(
        self,
        num_vals: int,
        init_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize an empty edit accumulator and its initial-set child."""
        del name
        self.pcnt = np.zeros((num_vals, 3), dtype=np.float64)
        self.keys = keys
        self.num_vals = num_vals
        self.init_acc = init_acc if init_acc is not None else NullAccumulator()
        self.tot_sum = 0.0

    def update(
        self, x: T, weight: float, estimate: Optional[IntegerBernoulliEditDistribution]
    ) -> None:
        """Add one weighted set-pair observation."""
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.update(
            x[0], weight, estimate.init_dist if estimate is not None else None
        )

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize from one set pair and delegate its initial set."""
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.initialize(x[0], weight, rng)

    def seq_update(
        self,
        x: "IntegerBernoulliEditEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[IntegerBernoulliEditDistribution],
    ) -> None:
        """Add weighted encoded edit categories and initial sets."""
        assert estimate is not None
        _sz, idx, xs, _ys, ym, init_enc = x.data

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_update(init_enc, weights, estimate.init_dist)

    def seq_initialize(
        self,
        x: "IntegerBernoulliEditEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize from weighted encoded edits and initial sets."""
        _sz, idx, xs, _ys, ym, init_enc = x.data

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_initialize(init_enc, weights, rng)

    def combine(
        self, suff_stat: Tuple[np.ndarray, float, Optional[SS1]]
    ) -> "IntegerBernoulliEditAccumulator":
        """Merge edit counts, total weight, and child statistics."""
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        self.init_acc.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[np.ndarray, float, Optional[Any]]:
        """Return edit counts, total weight, and initial-set statistics."""
        return self.pcnt, self.tot_sum, self.init_acc.value()

    def from_value(
        self, x: Tuple[np.ndarray, float, Optional[SS1]]
    ) -> "IntegerBernoulliEditAccumulator":
        """Restore edit and child statistics from a serialized value."""
        self.pcnt = x[0]
        self.tot_sum = x[1]
        self.init_acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed edit statistics and then delegate to the child."""
        if self.keys is not None:
            if self.keys in stats_dict:
                temp = stats_dict[self.keys]
                stats_dict[self.keys] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.keys] = (self.pcnt, self.tot_sum)

        self.init_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed edit statistics and then delegate to the child."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.keys]

        self.init_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Create an edit encoder using the child accumulator's encoder."""
        return IntegerBernoulliEditDataEncoder(
            init_encoder=self.init_acc.acc_to_encoder()
        )


class IntegerBernoulliEditAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating Integer Bernoulli Edit Accumulators.

    Attributes:
        keys (Optional[str]): Keys for parameters of distribution.
        init_factory (StatisticAccumulatorFactory): Initial factory for creating
            accumulators.
        num_vals (int): Number of values in the distribution.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        num_vals: int,
        init_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Factory for creating Integer Bernoulli Edit Accumulators.

        Args:
            num_vals (int): Number of values in the distribution.
            init_factory (Optional[StatisticAccumulatorFactory]): Initial accumulator
                factory. Defaults to NullAccumulatorFactory().
            keys (Optional[str]): Keys for parameters of distribution.
            name (Optional[str]): Name for object.
        """
        self.keys = keys
        self.init_factory = (
            init_factory if init_factory is not None else NullAccumulatorFactory()
        )
        self.num_vals = num_vals
        self.name = name

    def make(self) -> "IntegerBernoulliEditAccumulator":
        """Create an empty edit accumulator with a new child accumulator."""
        return IntegerBernoulliEditAccumulator(
            self.num_vals,
            init_acc=self.init_factory.make(),
            keys=self.keys,
            name=self.name,
        )


class IntegerBernoulliEditEstimator(ParameterEstimator):
    """Estimator for the Integer Bernoulli Edit Set Distribution.

    Attributes:
        num_vals (int): Number of values in the distribution.
        keys (Optional[str]): Keys for parameters of distribution.
        pseudo_count (Optional[float]): Pseudo-count for smoothing.
        suff_stat (Optional[np.ndarray]): Sufficient statistics for estimation.
        name (Optional[str]): Name for object.
        min_prob (float): Minimum probability value.
        init_est (ParameterEstimator): Initial estimator object.
    """

    def __init__(
        self,
        num_vals: int,
        init_estimator: Optional[ParameterEstimator] = NullEstimator(),
        min_prob: float = 1.0e-128,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Estimator for the Integer Bernoulli Edit Set Distribution.

        Args:
            num_vals (int): Number of values in the distribution.
            init_estimator (Optional[ParameterEstimator]): Initial estimator. Defaults
                to NullEstimator().
            min_prob (float): Minimum probability value. Defaults to 1.0e-128.
            pseudo_count (Optional[float]): Pseudo-count for smoothing. Defaults to
                None.
            suff_stat (Optional[np.ndarray]): Sufficient statistics. Defaults to None.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Keys for parameters of distribution.
        """
        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob
        self.init_est = (
            init_estimator if init_estimator is not None else NullEstimator()
        )

    def accumulator_factory(self) -> "IntegerBernoulliEditAccumulatorFactory":
        """Create a compatible factory using the initial-set child estimator."""
        return IntegerBernoulliEditAccumulatorFactory(
            self.num_vals,
            self.init_est.accumulator_factory(),
            name=self.name,
            keys=self.keys,
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, float, Optional[SS1]]
    ) -> "IntegerBernoulliEditDistribution":
        """Estimate transition probabilities and the initial-set distribution.

        ``nobs`` is ignored. The three explicit transition counts plus total weight
        determine all four transition outcomes for every integer. ``init_est`` receives
        the third sufficient-statistic component independently.
        """
        init_dist = self.init_est.estimate(None, suff_stat[2])
        count_mat, tot_sum, _ = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:

            p = self.pseudo_count
            s = self.suff_stat

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p * (s[:, 1] + s[:, 3]))
            log_s0 = np.log(s0 + p * (s[:, 0] + s[:, 2]))

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + p * s[:, 0]) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + p * s[:, 1]) - log_s1
            log_pmat[:, 2] = np.log(count_mat[:, 1] + p * s[:, 2]) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + p * s[:, 3]) - log_s1

        elif self.pseudo_count is not None and self.suff_stat is None:

            p = self.pseudo_count

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p / 2.0)
            log_s0 = np.log(s0 + p / 2.0)

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 2] = np.log(count_mat[:, 1] + (p / 4.0)) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + (p / 4.0)) - log_s1
            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + (p / 4.0)) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + (p / 4.0)) - log_s1

        else:

            if suff_stat[1] == 0:
                log_pmat = np.zeros((self.num_vals, 4), dtype=np.float64) + np.log(0.5)

            elif (self.min_prob is not None) and (self.min_prob > 0):

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                log_pmat.fill(np.log(self.min_prob))

                if np.any(s0 != 0):
                    log_pmat[:, 0] = np.log(
                        np.maximum((s0 - count_mat[:, 1]) / s0, self.min_prob)
                    )
                    log_pmat[:, 2] = np.log(
                        np.maximum(count_mat[:, 1] / s0, self.min_prob)
                    )

                if np.any(s1 != 0):
                    log_pmat[:, 1] = np.log(
                        np.maximum(count_mat[:, 0] / s1, self.min_prob)
                    )
                    log_pmat[:, 3] = np.log(
                        np.maximum(count_mat[:, 2] / s1, self.min_prob)
                    )

            else:

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) / s0)
                log_pmat[:, 1] = np.log(count_mat[:, 0] / s1)
                log_pmat[:, 2] = np.log(count_mat[:, 1] / s0)
                log_pmat[:, 3] = np.log(count_mat[:, 2] / s1)

        return IntegerBernoulliEditDistribution(
            log_pmat, init_dist=init_dist, name=self.name
        )


class IntegerBernoulliEditDataEncoder(DataSequenceEncoder):
    """Encode set pairs as flattened removal, addition, and retention events.

    Attributes:
        init_encoder (DataSequenceEncoder): Initial encoder for the distribution.
    """

    def __init__(self, init_encoder: DataSequenceEncoder) -> None:
        """Data encoder for the Integer Bernoulli Edit Set Distribution.

        Args:
            init_encoder (DataSequenceEncoder): Initial encoder for the distribution.
        """
        self.init_encoder = init_encoder

    def __str__(self) -> str:
        """String representation of the data encoder."""
        return (
            "IntegerBernoulliEditDataEncoder(init_encoder="
            + str(self.init_encoder)
            + ")"
        )

    def __eq__(self, other: object) -> bool:
        """Checks equality between two encoders.

        Args:
            other (object): Another encoder object.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, IntegerBernoulliEditDataEncoder):
            return other.init_encoder == self.init_encoder
        return False

    def seq_encode(self, x: Sequence[T]) -> "IntegerBernoulliEditEncodedDataSequence":
        """Encode a batch of unordered ``(before, after)`` set pairs.

        Args:
            x (Sequence[T]): Set pairs containing unique integers in the modeled
                universe.

        Returns:
            IntegerBernoulliEditEncodedDataSequence: Flattened transition events plus
                the child encoding of every initial set.
        """
        idx: List[int] = []
        xs: List[int] = []
        ys: List[int] = []
        pre: List[Union[Sequence[int], np.ndarray]] = []

        for i, xx in enumerate(x):
            pre.append(xx[0])

            xx0 = np.asarray(xx[0], dtype=int)
            xx1 = np.asarray(xx[1], dtype=int)

            to_add = np.isin(xx1, xx0, invert=False)
            to_rem = np.isin(xx0, xx1, invert=True)

            new_x = np.concatenate([xx0[to_rem], xx1[~to_add], xx1[to_add]])

            new_i = np.concatenate(
                [
                    np.full(int(np.sum(to_rem)), 0, dtype=np.int32),
                    np.full(int(np.sum(~to_add)), 1, dtype=np.int32),
                    np.full(int(np.sum(to_add)), 2, dtype=np.int32),
                ]
            )

            idx.extend([i] * len(new_x))
            xs.extend(list(new_x))
            ys.extend(list(new_i))

        idx_arr = np.asarray(idx, dtype=np.int32)
        xs_arr = np.asarray(xs, dtype=np.int32)
        ys_arr = np.asarray(ys, dtype=np.int32)
        ym = (
            np.flatnonzero(ys_arr == 0),
            np.flatnonzero(ys_arr == 1),
            np.flatnonzero(ys_arr == 2),
        )

        init_enc = self.init_encoder.seq_encode(pre)

        return IntegerBernoulliEditEncodedDataSequence(
            data=(len(x), idx_arr, xs_arr, ys_arr, ym, init_enc)
        )


class IntegerBernoulliEditEncodedDataSequence(EncodedDataSequence):
    """Store flattened edit events and encoded initial sets.

    Attributes:
        data (Tuple[int, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray,
        np.ndarray, np.ndarray], EncodedDataSequence]):
            Encoded data containing size, indices, values, and other metadata.
    """

    def __init__(
        self,
        data: Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            EncodedDataSequence,
        ],
    ):
        """Encoded data sequence for the Integer Bernoulli Edit Set Distribution.

        Args:
            data: Tuple containing the batch size, flattened observation indices,
                edited integer values, transition-category indices, positions grouped
                by category, and the child encoding of every initial set.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded tuple."""
        return f"IntegerBernoulliEditEncodedDataSequence(data={self.data})"
