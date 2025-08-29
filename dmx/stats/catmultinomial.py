""""Create, estimate, and sample from a MultinomialDistribution.

Defines the MultinomialDistribution, MultinomialSampler, MultinomialAccumulatorFactory, MultinomialAccumulator,
MultinomialEstimator, and the MultinomialDataEncoder classes for use with dmx-learn.

Let P_dist(V_k) be a distribution for a countable set of discrete observations of values V_k of type T. Denote

    p_k = P_dist(V_k),

as the probability of success for value V_k. Then sum_{k=0}^{inf} p_k = 1. Let x = (x_0, x_1,....,x_{n-1}) be a
multinomial observation for a 'n' trials, where each x_i = (V_j, n_j) for some value V_j in the observation space and
n_j is the associated number of success for the value. (note: sum n_j = n). Then, denoting p_j = p_mat(V_j), we have
log-density:

    log(p_mat(x)) = log(n!) - sum_{j=0}^{n-1} n_j * log(p_j) - log(n_j!) + log(P_len(n)),

where P_len(n) is a distribution for the number of trials in the multinomial having support on the non-negative
integers.

The multinomial is assumed to have data type: Sequence[Tuple[T, float]], where T is the data type of the 'categories'.

"""
from __future__ import annotations
import numpy as np
from numpy.random import RandomState
from dmx.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence
from dmx.arithmetic import maxrandint
from dmx.stats.null_dist import NullDistribution, NullEstimator, NullAccumulator, NullAccumulatorFactory
from typing import Optional, Sequence, Tuple, Any, Union, Dict, TypeVar

T = TypeVar('T') ## Generic data type for value.

SS1 = TypeVar('SS1') ## suff stat type for dist
SS2 = TypeVar('SS2') ## suff stat type for len_dist


class MultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Multinomial distribution over a countable support with optional distribution for number of trials.

    Args:
        dist (SequenceEncodableProbabilityDistribution): Distribution with at most a countable support.
        len_dist (Optional[SequenceEncodableProbabilityDistribution], optional): Distribution for the number of trials. Defaults to NullDistribution().
        len_normalized (bool, optional): Take geometric mean of the density of observation. Defaults to False.
        name (Optional[str], optional): Name for the object instance. Defaults to None.
        keys (Optional[str], optional): Keys for merging sufficient statistics. Defaults to None.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Distribution with at most a countable support.
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the number of trials.
        len_normalized (bool): Take geometric mean of the density of observation.
        name (Optional[str]): Name for the object instance.
        keys (Optional[str]): Keys for merging sufficient statistics.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
        len_normalized: bool = False,
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initializes a MultinomialDistribution object.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution with at most a countable support.
            len_dist (Optional[SequenceEncodableProbabilityDistribution], optional): Distribution for the number of trials. Defaults to NullDistribution().
            len_normalized (bool, optional): Take geometric mean of the density of observation. Defaults to False.
            name (Optional[str], optional): Name for the object instance. Defaults to None.
            keys (Optional[str], optional): Keys for merging sufficient statistics. Defaults to None.
        """
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.len_normalized = len_normalized
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Returns string representation of the object instance.

        Returns:
            str: String representation.
        """
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return (
            f"MultinomialDistribution({s1}, len_dist={s2}, "
            f"len_normalized={s3}, name={s4}, keys={s5})"
        )

    def density(self, x: Sequence[Tuple[T, float]]) -> float:
        """Returns the density of multinomial evaluated at observation x.

        Args:
            x (Sequence[Tuple[T, float]]): Tuples of observed multinomial values and successes such that the successes sum to the number of trials.

        Returns:
            float: Density evaluated at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Tuple[T, float]]) -> float:
        """Returns the log-density of multinomial evaluated at observation x.

        Args:
            x (Sequence[Tuple[T, float]]): Tuples of observed multinomial values and successes such that the successes sum to the number of trials.

        Returns:
            float: Log-density evaluated at x.
        """
        rv = 0.0
        cc = 0.0
        for i in range(len(x)):
            rv += self.dist.log_density(x[i][0]) * x[i][1]
            cc += x[i][1]

        if self.len_normalized and len(x) > 0:
            rv /= cc

        rv += self.len_dist.log_density(cc)

        return rv

    def seq_log_density(self, x: 'MultinomialEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (MultinomialEncodedDataSequence): Encoded sequence.

        Returns:
            np.ndarray: Log-densities for each observation.

        Raises:
            Exception: If input is not a MultinomialEncodedDataSequence.
        """
        if not isinstance(x, MultinomialEncodedDataSequence):
            raise Exception(
                "MultinomialDistribution.seq_log_density() requires MultinomialEncodedDataSequence."
            )

        idx, icnt, inz, enc_seq, enc_nseq, enc_w, enc_ww = x.data

        ll = self.dist.seq_log_density(enc_seq)
        ll_sum = np.bincount(idx, weights=ll * enc_w, minlength=len(icnt))

        if self.len_normalized:
            ll_sum *= icnt

        if enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def sampler(self, seed: Optional[int] = None) -> 'MultinomialSampler':
        """Create a MultinomialSampler object from this distribution.

        Args:
            seed (Optional[int], optional): Seed for sampling. Defaults to None.

        Returns:
            MultinomialSampler: Sampler for this distribution.

        Raises:
            Exception: If len_dist is a NullDistribution.
        """
        if isinstance(self.len_dist, NullDistribution):
            raise Exception(
                "len_dist must not be a SequenceEncodableProbabilityDistribution with support of non-negative integers."
            )
        return MultinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'MultinomialEstimator':
        """Create a MultinomialEstimator object from this distribution.

        Args:
            pseudo_count (Optional[float], optional): Re-weight member sufficient statistics when estimating from aggregated data.

        Returns:
            MultinomialEstimator: Estimator for this distribution.
        """
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        dist_est = self.dist.estimator(pseudo_count=pseudo_count)

        return MultinomialEstimator(
            dist_est,
            len_estimator=len_est,
            len_normalized=self.len_normalized,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys
        )

    def dist_to_encoder(self) -> 'MultinomialDataEncoder':
        """Get a data encoder for this distribution.

        Returns:
            MultinomialDataEncoder: Encoder for this distribution.
        """
        return MultinomialDataEncoder(
            encoder=self.dist.dist_to_encoder(),
            len_encoder=self.len_dist.dist_to_encoder()
        )


class MultinomialSampler(DistributionSampler):
    """MultinomialSampler object for sampling from multinomial distribution.

    Attributes:
         dist (MultinomialDistribution): An instance of a MultinomialDistribution object.
         rng (RandomState): RandomState with seed set if passed.
         dist_sampler (DistributionSampler): DistributionSampler object for sampling category values.
         len_sampler (DistributionSampler): DistributionSampler object for sampling number of trials in multinomial.

    """

    def __init__(self, dist: MultinomialDistribution, seed: Optional[int] = None) -> None:
        """MultinomialSampler object.

        Args:
            dist (MultinomialDistribution): An instance of a MultinomialDistribution object.
            seed (Optional[int]): Set the seed for sampling.

        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None)\
            -> Union[Sequence[Sequence[Tuple[Any, float]]], Sequence[Tuple[Any, float]]]:
        """Draw samples from multinomial distribution.

        Note: If len_sampler can draw n=0, an empty list is returned for that sample.

        Args:
            size (Optional[int]): Number of iid samples to draw from multinomial.

        Returns:
            Union[Sequence[Sequence[Tuple[Any, float]]], Sequence[Tuple[Any, float]]]: Sequence of 'size' iid
            observations if size is not None, else a single multinomial sample.

        """
        if size is None:
            n = self.len_sampler.sample()
            rv = dict()
            for i in range(n):
                v = self.dist_sampler.sample()
                if v in rv:
                    rv[v] += 1
                else:
                    rv[v] = 1
            return list(rv.items())

        else:
            return [self.sample() for i in range(size)]


class MultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """MultinomialAccumulator object.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for category values.
        len_normalized (bool): Take geometric mean of density.
        len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for the number of trials in
            each observation, defaults to the NullAccumulator.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        _init_rng (bool): True if RandomState objects have been initialized
        _len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
        _acc_rng (Optional[RandomState]): List of RandomState objects for initializing category accumulator.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, len_normalized: bool,
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """MultinomialAccumulator object.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for category values.
            len_normalized (bool): Take geometric mean of density.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator object for the
                number of trials in each observation.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        """
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.name = name
        self.keys = keys
        self.len_normalized = len_normalized

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: Optional[RandomState] = None
        self._acc_rng: Optional[RandomState] = None

    def update(
        self,
        x: Sequence[Tuple[T, float]],
        weight: float,
        estimate: Optional['MultinomialDistribution']
    ) -> None:
        """Update the accumulator with a new observation.

        Args:
            x (Sequence[Tuple[T, float]]): Observation as a sequence of (category, count) tuples.
            weight (float): Weight for the observation.
            estimate (Optional[MultinomialDistribution]): Current estimate of the distribution, or None.
        """
        cc = [u[1] for u in x]
        ss = sum(cc)

        if estimate is None:
            w = weight / ss if (self.len_normalized and ss > 0) else weight
            for i in range(len(x)):
                self.accumulator.update(x[i][0], w * x[i][1], None)
            self.len_accumulator.update(ss, weight, None)
        else:
            w = weight / ss if (self.len_normalized and ss > 0) else weight
            for i in range(len(x)):
                self.accumulator.update(x[i][0], w * x[i][1], estimate.dist)
            self.len_accumulator.update(ss, weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set RandomState member variables for initialize and seq_initialize consistency.

        Args:
            rng (RandomState): RandomState object used to set member RandomState objects.

        Returns:
            None

        """
        rng_seeds = rng.randint(maxrandint, size=2)
        self._len_rng = RandomState(seed=rng_seeds[0])
        self._acc_rng = RandomState(seed=rng_seeds[1])
        self._init_rng = True

    def initialize(
        self,
        x: Sequence[Tuple[T, float]],
        weight: float,
        rng: RandomState
    ) -> None:
        """Initialize the accumulator with a new observation and random state.

        Args:
            x (Sequence[Tuple[T, float]]): Observation as a sequence of (category, count) tuples.
            weight (float): Weight for the observation.
            rng (RandomState): Random state for initialization.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        cc = [u[1] for u in x]
        ss = sum(cc)
        w = weight / ss if self.len_normalized and ss > 0 else weight

        for i in range(len(x)):
            self.accumulator.initialize(x[i][0], w * x[i][1], self._acc_rng)

        self.len_accumulator.initialize(ss, weight, self._len_rng)

    def seq_update(
        self,
        x: 'MultinomialEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['MultinomialDistribution']
    ) -> None:
        """Update the accumulator with a sequence of encoded observations.

        Args:
            x (MultinomialEncodedDataSequence): Encoded sequence of observations.
            weights (np.ndarray): Array of weights for each observation.
            estimate (Optional[MultinomialDistribution]): Current estimate of the distribution, or None.
        """
        idx, icnt, inz, enc_seq, enc_nseq, enc_w, enc_ww = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]
        w *= enc_w

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)
        self.len_accumulator.seq_update(enc_nseq, weights * enc_ww, estimate.len_dist if estimate is not None else None)

    def seq_initialize(
        self,
        x: 'MultinomialEncodedDataSequence',
        weights: np.ndarray,
        rng: RandomState
    ) -> None:
        """Initialize the accumulator with a sequence of encoded observations and random state.

        Args:
            x (MultinomialEncodedDataSequence): Encoded sequence of observations.
            weights (np.ndarray): Array of weights for each observation.
            rng (RandomState): Random state for initialization.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        idx, icnt, inz, enc_seq, enc_nseq, enc_w, enc_ww = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]
        w = w * enc_w

        self.accumulator.seq_initialize(enc_seq, w, self._acc_rng)
        self.len_accumulator.seq_initialize(enc_nseq, weights * enc_ww, self._len_rng)

    def combine(
        self,
        suff_stat: Tuple[SS1, Optional[SS2]]
    ) -> 'MultinomialAccumulator':
        """Combine this accumulator with another sufficient statistic.

        Args:
            suff_stat (Tuple[SS1, Optional[SS2]]): Sufficient statistics to combine.

        Returns:
            MultinomialAccumulator: Self after combining.
        """
        self.accumulator.combine(suff_stat[0])
        self.len_accumulator.combine(suff_stat[1])
        return self

    def value(self) -> Tuple[Any, Optional[Any]]:
        """Get the value of the sufficient statistics.

        Returns:
            Tuple[Any, Optional[Any]]: Value of the accumulator and length accumulator.
        """
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(
        self,
        x: Tuple[SS1, Optional[SS2]]
    ) -> 'MultinomialAccumulator':
        """Set the accumulator from a value.

        Args:
            x (Tuple[SS1, Optional[SS2]]): Value to set.

        Returns:
            MultinomialAccumulator: Self after setting value.
        """
        self.accumulator.from_value(x[0])
        self.len_accumulator.from_value(x[1])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary of statistics by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of statistics.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.accumulator.key_merge(stats_dict)
        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's value from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of statistics.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.accumulator.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'MultinomialDataEncoder':
        """Get a data encoder for this accumulator.

        Returns:
            MultinomialDataEncoder: Encoder for this accumulator.
        """
        return MultinomialDataEncoder(
            encoder=self.accumulator.acc_to_encoder(),
            len_encoder=self.len_accumulator.acc_to_encoder()
        )


class MultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """MultinomialAccumulatorFactory object for creating MultinomialAccumulator objects.

    Attributes:
        est_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the value distribution.
        len_normalized (bool): If true, geometric mean of density is taken.
        len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for number of trials.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

    """

    def __init__(self, est_factory: StatisticAccumulatorFactory, len_normalized: bool,
                 len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 name: Optional[str]  = None,
                 keys: Optional[str] = None) -> None:
        """MultinomialAccumulatorFactory object for creating MultinomialAccumulator objects.

        Args:
            est_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the value distribution.
            len_normalized (bool): If true, geometric mean of density is taken.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for number of trials.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        """
        self.est_factory = est_factory
        self.len_normalized = len_normalized
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self) -> 'MultinomialAccumulator':

        return MultinomialAccumulator(
            accumulator=self.est_factory.make(), 
            len_normalized=self.len_normalized, 
            len_accumulator=self.len_factory.make(), 
            name=self.name,
            keys=self.keys)


class MultinomialEstimator(ParameterEstimator):
    """MultinomialEstimator object for estimating MultinomialDistribution objects from aggregated data.

    Attributes:
        estimator (ParameterEstimator): ParameterEstimator for distribution of values.
        len_estimator (ParameterEstimator): ParameterEstimator for the number of trials, defaults to
            the NullEstimator if None is passed.
        pseudo_count (Optional[float]): Regularizer estimator and len_estimator. 
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): If None, distribution for number of trials
            will be estimated from 'len_estimator'.
        len_normalized (Optional[bool]): Take geometric mean of density.
        name (Optional[str]): Name of object instance.
        keys (Optional[str]): Keys of object instance for merging sufficient statistics.

    """

    def __init__(self, 
                 estimator: ParameterEstimator, 
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[float] = None,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
                 len_normalized: Optional[bool] = False,
                 name: Optional[str] = None,  keys: Optional[str] = None) -> None:
        """MultinomialEstimator object.

        Args:
            estimator (ParameterEstimator): ParameterEstimator for distribution of values.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator for the number of trials.
            pseudo_count (Optional[float]): Regularizer estimator and len_estimator. 
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Set distribution for the number of trials.
            len_normalized (Optional[bool]): Take geometric mean of density.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys to object instance for merging sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("MultinomialEstimator requires keys to be of type 'str'.")

        self.estimator = estimator
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.len_dist = len_dist
        self.len_normalized = len_normalized
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'MultinomialAccumulatorFactory':

        est_factory = self.estimator.accumulator_factory()
        len_factory = self.len_estimator.accumulator_factory()
        return MultinomialAccumulatorFactory(est_factory=est_factory, 
                                             len_normalized=self.len_normalized,
                                             name=self.name,
                                             len_factory=len_factory, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[SS1, Optional[SS2]]) -> 'MultinomialDistribution':
        """Estimate a MultinomialDistribution object from aggregated data contained in arg 'suff_stat'.

        Args:
            nobs (Optional[float]): Number of observations used in aggregation of 'suff_stat'.
            suff_stat (Tuple[SS1, Optional[SS2]]): Tuple of sufficient statistics for distribution of values and
                trial distribution.

        Returns:
            MultinomialDistribution: Estimate from sufficient statistics.

        """
        len_dist = self.len_estimator.estimate(nobs, suff_stat[1]) if self.len_dist is None else self.len_dist
        dist = self.estimator.estimate(nobs, suff_stat[0])

        return MultinomialDistribution(dist=dist, len_dist=len_dist, len_normalized=self.len_normalized,
                                       name=self.name)


class MultinomialDataEncoder(DataSequenceEncoder):
    """MultinomialDataEncoder object for encoding sequences of iid multinomial observations.

    Note: Arg encoders[0] must encoder data type T of multinomial, and encoders[1] must have support on the
    positive integers.

    Attributes:
         encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the
            multinomial value encoder. Must be data type T.
        len_encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the trial size of the multinomial.

    """

    def __init__(self, encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder) -> None:
        """MultinomialDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the
                multinomial value encoder. Must be data type T.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the trial size of the multinomial.

        """
        self.encoder = encoder
        self.len_encoder = len_encoder

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultinomialDataEncoder):
            return other.len_encoder == self.len_encoder
        else:
            return False

    def __str__(self) -> str:
        return 'MultinomialDataEncoder(len_encoder=' + str(self.len_encoder) + ')'

    def seq_encode(self, x: Sequence[Sequence[Tuple[T, float]]]) -> 'MultinomialEncodedDataSequence':
        """Encode a sequence of iid observations of multinomial distribution for use with vectorized functions.

        Returns a tuple of size 7 containing:
            rv1 (ndarray[int]): Observation index of sequence values.
            rv2 (ndarray[float]): Trial size for each observation.
            rv3 (ndarray[float]): Non-zero trial size indices.
            rv4 (EncodedDataSequence): Sequence encoded flattened list of values from x.
            rv5 (Optional[EncodedDataSequence]): Sequence encoded flatted list of trial sizes.
            rv6 (np.ndarray[float]): Flattened array of counts for values.
            rv7 (ndarray[float]): Flattened array of trial sizes.

        Args:
            x (Sequence[Sequence[Tuple[T, float]]]): Sequence of iid observations of multinomial distributions.

        Returns:
            MultinomialEncodedDataSequence: See above.

        """
        tx = []
        nx = []
        tidx = []
        cc = []
        ccc = []

        for i in range(len(x)):
            nx.append(len(x[i]))
            aa = 0
            for j in range(len(x[i])):
                tidx.append(i)
                tx.append(x[i][j][0])
                cc.append(x[i][j][1])
                aa += x[i][j][1]
            ccc.append(aa)

        rv1 = np.asarray(tidx, dtype=int)
        rv2 = np.asarray(ccc, dtype=float)
        rv3 = (rv2 != 0)
        rv6 = np.asarray(cc, dtype=float)
        rv7 = np.asarray(ccc, dtype=float)

        rv2[rv3] = 1.0 / rv2[rv3]
        # rv2[rv3] = 1.0

        rv4 = self.encoder.seq_encode(tx)

        if self.len_encoder is not None:
            rv5 = self.len_encoder.seq_encode(ccc)
        else:
            rv5 = None

        return MultinomialEncodedDataSequence(data=(rv1, rv2, rv3, rv4, rv5, rv6, rv7))

class MultinomialEncodedDataSequence(EncodedDataSequence):
    """MultinomialEncodedDataSequence object for MultinomialDistribution.

    Data 'x' is a tuple of size 7 containing:
            x[0] (ndarray[int]): Observation index of sequence values.
            x[1] (ndarray[float]): Trial size for each observation.
            x[2] (ndarray[float]): Non-zero trial size indices.
            x[3] (EncodedDataSequence): Sequence encoded flattened list of values from x.
            x[4] (Optional[EncodedDataSequence]): Sequence encoded flatted list of trial sizes.
            x[5] (np.ndarray[float]): Flattened array of counts for values.
            x[6] (ndarray[float]): Flattened array of trial sizes.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray, np.ndarray, EncodedDataSequence, Optional[EncodedDataSequence], np.ndarray, np.ndarray]): See above.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, EncodedDataSequence, Optional[EncodedDataSequence], np.ndarray, np.ndarray]):
        """MultinomialEncodedDataSequence object.

        Args:
            data (Tuple[np.ndarray, np.ndarray, np.ndarray, EncodedDataSequence, Optional[EncodedDataSequence], np.ndarray, np.ndarray]): See above.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'MultinomialEncodedDataSequence(data={self.data}'

