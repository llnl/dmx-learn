r"""Provide fixed-lag Markov chains over a finite integer state space.

An observation is a finite state sequence :math:`x=(x_0,\ldots,x_{n-1})` with
:math:`x_i\in\{0,\ldots,K-1\}`.  For lag :math:`L` and :math:`n\geq L`,

.. math::

   p(x) = p_N(n)\,p_0(x_{0:L})
          \prod_{i=0}^{n-L-1}p(x_{i+L}\mid x_{i:i+L}).

``init_dist`` models the initial length-``lag`` block, while row
``ravel_multi_index(context, [num_values] * lag)`` of ``cond_dist`` models the
next state.  Scalar density evaluation and sampling interpret ``len_dist`` as
the distribution of ``n``.  Accumulator updates and encoded operations instead
pass ``max(n - lag + 1, 0)`` to the length component; random accumulator
initialization passes ``n - lag`` for sequences of at least ``lag`` states and
does not update it for shorter sequences.  A sequence shorter than ``lag`` has
no initial or transition term, and sampling such a requested length returns the
empty sequence.  These historical semantics are intentionally unchanged.

The module provides the distribution, sampler, estimator, sufficient-statistic
accumulator, and vectorized data encoder.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
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

SS1 = TypeVar("SS1")  ## suff stat of init
SS2 = TypeVar("SS2")  ## suff-stat of length


class IntegerMarkovChainDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a fixed-lag chain on states ``0`` through ``num_values - 1``.

    Each observation is a complete integer state sequence.  Sequence duration is
    supplied by ``len_dist``; there is no terminal state.

    Attributes:
        num_values (int): Total number of values in support.
        cond_dist (Array-like): Should be num_vals ** lag by num_vals with transition
            probabilities for each
            lagged length tuple (v_0,v_1,..,v_{lag}).
        lag (int): Lag length for conditional density.
        init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
            distribution for initial states
            of Markov chain (with length lag). Should be a distribution compatible with
            Sequences.
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
            distribution for the length of
            observations.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Set keys for merging sufficient statistics, including the
            sufficient statistics of
            init_dist and len_dist.

    """

    def __init__(
        self,
        num_values: int,
        cond_dist: Union[List[List[float]], np.ndarray],
        lag: int = 1,
        init_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize a fixed-lag integer Markov-chain distribution.

        Args:
            num_values (int): Total number of values in support.
            cond_dist (Array-like): Should be num_vals ** lag by num_vals with
                transition probabilities for each
                lagged length tuple (v_0,v_1,..,v_{lag}).
            lag (int): Lag length for conditional density.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                distribution for initial states
                of Markov chain (with length lag). Should be a distribution compatible
                with Sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                distribution for the length of
                observations.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics, including
                the sufficient statistics of
                init_dist and len_dist.

        """
        super().__init__()
        self.num_values = num_values
        self.cond_dist = np.asarray(cond_dist)
        self.lag = lag
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of object instance."""
        s1 = repr(self.num_values)
        s2 = repr(self.cond_dist.tolist())
        s3 = repr(self.lag)
        s4 = repr(self.init_dist) if self.init_dist is None else str(self.init_dist)
        s5 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.keys)

        return (
            f"IntegerMarkovChainDistribution({s1}, {s2}, lag={s3}, init_dist={s4}, "
            f"len_dist={s5}, name={s6}, keys={s7})"
        )

    def density(self, x: Sequence[int]) -> float:
        """Density of integer Markov chain evaluated at x.

        See log_density() for details.

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            float: Density evaluated at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Evaluate the scalar log density of an integer state sequence.

        For a sequence of length ``n >= lag``, this sums the initial-block log
        density, the log conditional probability of each later state given its
        preceding ``lag`` states, and ``len_dist.log_density(n)``.  A shorter
        sequence contributes only ``len_dist.log_density(n)``.

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            float: Log-density evaluated at x.

        """
        rv = 0.0
        lag = self.lag

        if len(x) >= lag:

            m_shape = [self.num_values] * lag
            rv += self.init_dist.log_density(x[:lag])

            for i in range(len(x) - lag):
                idx = np.ravel_multi_index(x[i : (i + lag)], m_shape)
                rv += np.log(self.cond_dist[idx, x[i + lag]])

        rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(self, x: "IntegerMarkovChainEncodedDataSequence") -> np.ndarray:
        """Evaluate log densities for an encoded batch of integer state sequences."""
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x.data

        left_idx = [
            np.ravel_multi_index(u[0], [self.num_values] * self.lag)
            for u in u_seq_values
        ]
        right_idx = np.asarray([u[1] for u in u_seq_values])
        temp_prob = np.log(self.cond_dist[left_idx, right_idx])
        temp_prob = temp_prob[u_seq_idx]

        rv = np.bincount(seq_idx, weights=temp_prob, minlength=len(seq_len))

        if self.init_dist is not None:
            rv[init_idx] += self.init_dist.seq_log_density(init_enc)

        if self.len_dist is not None and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerMarkovChainSampler":
        """Create a sampler for complete integer state sequences."""
        return IntegerMarkovChainSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerMarkovChainEstimator":
        """Create an estimator using this model's lag and component estimators."""
        init_est = self.init_dist.estimator()
        len_est = self.len_dist.estimator()

        return IntegerMarkovChainEstimator(
            num_values=self.num_values,
            lag=self.lag,
            init_estimator=init_est,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerMarkovChainDataEncoder":
        """Create an encoder for batches of integer state sequences."""
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist.dist_to_encoder()
        return IntegerMarkovChainDataEncoder(
            lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder
        )


class IntegerMarkovChainSampler(DistributionSampler):
    """Sample complete sequences from a fixed-lag integer Markov chain.

    Attributes:
        dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
        rng (RandomState): RandomState object with seed set if passed.
        trans_sampler (RandomState): RandomState object for sampling transitions.

    """

    def __init__(
        self, dist: IntegerMarkovChainDistribution, seed: Optional[int]
    ) -> None:
        """Initialize an integer Markov-chain sampler.

        Args:
            dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
            seed (Optional[int]): Set the seed for random sampling.

        """
        super().__init__(dist, seed)
        rng = self.rng
        seeds = rng.randint(0, maxrandint, size=3)

        self.dist = dist
        self.rng = rng
        self.trans_sampler = np.random.RandomState(seeds[0])

        if isinstance(self.dist.init_dist, NullDistribution):
            raise RuntimeError(
                "IntegerMarkovChainSampler requires init_dist for "
                "IntegerMarkovDistribution."
            )
        self.init_sampler = dist.init_dist.sampler(seeds[1])

        if isinstance(dist.len_dist, NullDistribution):
            raise RuntimeError(
                "IntegerMarkovChainSampler requires len_dist for "
                "IntegerMarkovDistribution."
            )
        self.len_sampler = dist.len_dist.sampler(seeds[2])

    def single_sample(self) -> Sequence[int]:
        """Returns a single sample from the integer Markov chain distribution."""
        cnt = int(self.len_sampler.sample())
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag

        if cnt >= lag:
            rv = [int(v) for v in self.init_sampler.sample()]  ## must return a list
            for _ in range(lag, cnt):
                idx = np.ravel_multi_index(rv[-lag:], m_shape)
                rv.append(
                    int(self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :]))
                )
            return rv
        return []

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Sequence[int]], Sequence[int]]:
        """Draw iid samples from an integer Markov chain distribution.

        Args:
            size (Optional[int]): If None, size is taken to be 0.

        Returns:
            Sequence[int] if size is None, else List[Sequence[int]] with length equal to
            size.

        """
        if size is not None:
            return [self.single_sample() for i in range(size)]
        return self.single_sample()

    def sample_given(self, x: Sequence[int]) -> int:
        """Sample from the Markov chain conditioned on a given value 'x'.

        Args:
            x (Sequence[int]): Sample from Markov chain conditioned on observing 'x'.

        Returns:
            Single sample transition from integer Markov chain.

        """
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag
        idx = np.ravel_multi_index(x[-lag:], m_shape)

        return int(self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :]))


class IntegerMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted statistics for fixed-lag integer chains.

    The sufficient statistic contains transition counts keyed by
    ``(context_tuple, next_state)``, followed by the initial-block and length
    component statistics.  Only sequences of at least ``lag`` states contribute
    an initial block, and only later states contribute transitions.

    Attributes:
        lag (int): The lag for the Markov chain.
        trans_count_map (Dict[Tuple[Sequence[int], int], float]): Dictionary for
            tracking transition counts.
        init_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the
            initial distribution. Should
            be a sequence compatible accumulator with support on the integers. Defaults
            to the NullAccumulator.
        len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the
            length of the observed
            sequences. Should be a sequence compatible accumulator with support on the
            non-negative integers.
            Defaults to the NullAccumulator.
        max_value (int): Largest value encountered when accumulating sufficient
            statistics.
        keys (Optional[str]): Set key for merging sufficient statistics with objects
            possessing matching key.
        name (Optional[str]): Set name for object.

        _init_rng (bool): True if RandomState objects for accumulator have been
            initialized.
        _acc_rng (Optional[RandomState]): RandomState object for initializing the init
            accumulator.
        _len_rng (Optional[RandomState]): RandomState object for initializing the length
            accumulator.

    """

    def __init__(
        self,
        lag: int,
        init_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
        len_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an integer Markov-chain accumulator.

        Args:
            lag (int): The lag for the Markov chain.
            init_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional
                accumulator for the initial
                distribution.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional
                accumulator for the length
                of the observed sequences.
            keys (Optional[str]): Set key for merging sufficient statistics with objects
                possessing matching key.
            name (Optional[str]): Set name for object.

        """
        self.lag = lag
        self.trans_count_map: Dict[Tuple[Tuple[int, ...], int], float] = {}
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator()
        )
        self.init_accumulator = (
            init_accumulator if init_accumulator is not None else NullAccumulator()
        )
        self.max_value = -1
        del name
        self.keys = keys

        self._acc_rng: Optional[RandomState] = None
        self._len_rng: Optional[RandomState] = None
        self._init_rng = False

    def update(
        self,
        x: Sequence[int],
        weight: float,
        estimate: Optional[IntegerMarkovChainDistribution],
    ) -> None:
        """Add a weighted integer state sequence to the statistics."""
        lag = self.lag
        self.len_accumulator.update(
            max(len(x) - lag + 1, 0),
            weight,
            estimate.len_dist if estimate is not None else None,
        )

        if len(x) >= lag:
            self.init_accumulator.update(
                x[:lag], weight, estimate.init_dist if estimate is not None else None
            )

        for i in range(len(x) - lag):
            entry = (tuple(x[i : (i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

    def _rng_initialize(self, rng: RandomState) -> None:

        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState) -> None:
        """Initialize statistics from a weighted integer state sequence."""
        if not self._init_rng:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        assert self._len_rng is not None

        lag = self.lag

        if len(x) >= lag:
            self.len_accumulator.initialize(len(x) - lag, weight, self._len_rng)
            self.init_accumulator.initialize(x[:lag], weight, self._acc_rng)

        for i in range(len(x) - lag):
            entry = (tuple(x[i : (i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

    def seq_update(
        self,
        x: "IntegerMarkovChainEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[IntegerMarkovChainDistribution],
    ) -> None:
        """Add an encoded weighted batch to the statistics."""
        _seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x.data

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_update(
            init_enc,
            weights[init_idx],
            estimate.init_dist if estimate is not None else None,
        )

        self.len_accumulator.seq_update(
            len_enc, weights, estimate.len_dist if estimate is not None else None
        )

    def seq_initialize(
        self,
        x: "IntegerMarkovChainEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Initialize statistics from an encoded weighted batch."""
        if not self._init_rng:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        assert self._len_rng is not None

        _seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x.data

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_initialize(init_enc, weights[init_idx], self._acc_rng)
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

    def combine(
        self,
        suff_stat: Tuple[
            Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]
        ],
    ) -> "IntegerMarkovChainAccumulator":
        """Merge another set of sufficient statistics into this accumulator."""
        for k, v in suff_stat[0].items():
            self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        if suff_stat[1] is not None:
            self.init_accumulator.combine(suff_stat[1])

        if suff_stat[2] is not None:
            self.len_accumulator.combine(suff_stat[2])

        return self

    def value(
        self,
    ) -> Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[Any], Optional[Any]]:
        """Return transition, initial-block, and length statistics."""
        return (
            self.trans_count_map,
            self.init_accumulator.value(),
            self.len_accumulator.value(),
        )

    def from_value(
        self,
        x: Tuple[
            Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]
        ],
    ) -> "IntegerMarkovChainAccumulator":
        """Replace the accumulated sufficient statistics with ``x``."""
        self.trans_count_map = x[0]
        if x[1] is not None:
            self.init_accumulator = self.init_accumulator.from_value(x[1])

        if x[2] is not None:
            self.len_accumulator = self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics into entries selected by configured keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from entries selected by configured keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerMarkovChainDataEncoder":
        """Create an encoder compatible with this accumulator."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        init_encoder = self.init_accumulator.acc_to_encoder()
        return IntegerMarkovChainDataEncoder(
            lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder
        )


class IntegerMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for fixed-lag integer Markov chains.

    Attributes:
        lag (int): Length of lag in Markov chain.
        init_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory object
            for the init distribution.
            Should be compatible with sequences of integers. Defaults to
            NullAccumulatorFactory if None.
        len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory object
            for the length of Markov
            chain sequence. Requires support on non-negative integers. Defaults to
            NullAccumulatorFactory if None.
        keys (Optional[str]): Set key for merging sufficient statistics, including the
            sufficient statistics of
            init_dist and len_dist.
        name (Optional[str]): Set name for object.

    """

    def __init__(
        self,
        lag: int,
        init_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an integer Markov-chain accumulator factory.

        Args:
            lag (int): Length of lag in Markov chain.
            init_factory (Optional[StatisticAccumulatorFactory]): Optional
                StatisticAccumulatorFactory object for the
                init distribution. Should be compatible with sequences of integers.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional
                StatisticAccumulatorFactory object for the
                length of Markov chain sequence. Should have support on non-negative
                integers.
            keys (Optional[str]): Set keys for merging sufficient statistics, including
                the sufficient statistics of
                init_dist and len_dist.
            name (Optional[str]): Set name for object.

        """
        self.lag = lag
        self.init_factory = (
            init_factory if init_factory is not None else NullAccumulatorFactory()
        )
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerMarkovChainAccumulator":
        """Create a new empty integer Markov-chain accumulator."""
        init_acc = self.init_factory.make()
        len_acc = self.len_factory.make()
        return IntegerMarkovChainAccumulator(
            self.lag, init_acc, len_acc, keys=self.keys, name=self.name
        )


class IntegerMarkovChainEstimator(ParameterEstimator):
    """Estimate a fixed-lag integer chain from aggregated statistics.

    Transition counts are arranged into a dense ``K**lag`` by ``K`` matrix and
    normalized row-wise.  ``pseudo_count``, when supplied, is added to every
    matrix cell before normalization.  Initial-block and length distributions
    are either held fixed by ``init_dist`` and ``len_dist`` or estimated from
    their corresponding component statistics.  During estimation, the matrix
    state count is inferred as one plus the largest integer present in a
    transition key; the configured ``num_values`` is not used for that step.

    Attributes:
        num_values (int): Number of values in Markov chain support.
        lag (int): Length of conditional dependence.
        init_estimator (ParameterEstimator): Optional ParameterEstimator object
            compatible with
            sequences of integers. Defaults to NullEstimator.
        len_estimator (ParameterEstimator): ParameterEstimator object compatible with
            the non-negative integers.
            Defaults to the NullEstimator.
        init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed,
            init_dist is fixed and not
            estimated. Must be compatible with sequences of integers.
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed,
            len_dist is fixed and not
            estimated. Must be compatible with non-negative integers.
        pseudo_count (Optional[float]): If passed sufficient statistics are re-weighted
            in estimation step.
        name (Optional[str]): Set name to object instance.
        key (Optional[str]): Set key for merging sufficient statistics, including the
            sufficient statistics of
            init_dist and len_dist.

    """

    def __init__(
        self,
        num_values: int,
        lag: int = 1,
        init_estimator: Optional[ParameterEstimator] = NullEstimator(),
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        init_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
        len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
        pseudo_count: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a fixed-lag integer Markov-chain estimator.

        Args:
            num_values (int): Number of values in Markov chain support.
            lag (int): Length of conditional dependence.
            init_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator
                object compatible with
                sequences of integers.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator
                object compatible with the
                non-negative integers.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed,
                init_dist is fixed and not
                estimated. Must be compatible with sequences of integers.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed,
                len_dist is fixed and not
                estimated. Must be compatible with non-negative integers.
            pseudo_count (Optional[float]): If passed sufficient statistics are
                re-weighted in estimation step.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics, including
                the sufficient statistics of
                init_dist and len_dist.

        """
        self.num_values = num_values
        self.lag = lag
        self.init_estimator = (
            init_estimator if init_estimator is not None else NullEstimator()
        )
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.init_dist = init_dist
        self.len_dist = len_dist
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "IntegerMarkovChainAccumulatorFactory":
        """Create a factory for compatible sufficient-statistic accumulators."""
        len_factory = self.len_estimator.accumulator_factory()
        init_factory = self.init_estimator.accumulator_factory()
        return IntegerMarkovChainAccumulatorFactory(
            self.lag, init_factory, len_factory, keys=self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[
            Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]
        ],
    ) -> "IntegerMarkovChainDistribution":
        """Estimate a distribution from transition and component statistics."""
        trans_count_map, init_ss, len_ss = suff_stat
        lag = self.lag

        len_dist = (
            self.len_dist
            if self.len_dist is not None
            else self.len_estimator.estimate(None, len_ss)
        )
        init_dist = (
            self.init_dist
            if self.init_dist is not None
            else self.init_estimator.estimate(None, init_ss)
        )

        num_values = 1 + max(
            observed_value
            for context, value in trans_count_map.keys()
            for observed_value in (*context, value)
        )

        cond_mat = np.zeros((num_values**lag, num_values), dtype=np.float32)

        vv = list(trans_count_map.items())
        yidx = np.asarray(
            [np.ravel_multi_index(u[0], [num_values] * lag) for u, _ in vv]
        )
        xidx = np.asarray([u[1] for u, _ in vv])
        zidx = np.asarray([u[1] for u in vv])

        cond_mat[yidx, xidx] = zidx

        if self.pseudo_count is not None:
            cond_mat += self.pseudo_count

        cond_mat /= cond_mat.sum(axis=1, keepdims=True)

        return IntegerMarkovChainDistribution(
            num_values,
            cond_mat,
            init_dist=init_dist,
            lag=lag,
            len_dist=len_dist,
            name=self.name,
        )


class IntegerMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode batches of fixed-lag integer state sequences.

    The encoding records sequence indices for initial blocks and transitions,
    deduplicates ``(context, next_state)`` transitions, and delegates initial
    blocks and effective lengths to their component encoders.

    Attributes:
        lag (int): Integer valued length of lag.
        init_encoder (DataSequenceEncoder): DataSequenceEncoder object for initial
            lagged value. Should be a
            DataSequenceEncoder for a Sequence of distribution with support on integers.
        len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of
            observed sequences. Should be
            a DataSequenceEncoder with support on the integers.

    """

    def __init__(
        self,
        lag: int,
        init_encoder: DataSequenceEncoder = NullDataEncoder(),
        len_encoder: DataSequenceEncoder = NullDataEncoder(),
    ) -> None:
        """Initialize an integer Markov-chain data encoder.

        Args:
            lag (int): Integer valued length of lag.
            init_encoder (DataSequenceEncoder): DataSequenceEncoder object for initial
                lagged value.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of
                observed sequences.

        """
        self.lag = lag
        self.init_encoder = init_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a representation of the encoder."""
        rv = "IntegerMarkovChainDataEncoder(len_encoder=" + str(self.len_encoder)
        rv += ",init_encoder=" + str(self.init_encoder) + ",lag=" + str(self.lag) + ")"
        return rv

    def __eq__(self, other: object) -> bool:
        """Return whether two encoders have equal lag and component encoders."""
        if isinstance(other, IntegerMarkovChainDataEncoder):
            c0 = other.init_encoder == self.init_encoder
            c1 = other.len_encoder == self.len_encoder
            c2 = self.lag == other.lag
            if c0 and c1 and c2:
                return True
            return False
        return False

    def seq_encode(
        self, x: List[Sequence[int]]
    ) -> "IntegerMarkovChainEncodedDataSequence":
        """Encode independent observations from an integer Markov chain.

        Args:
            x: Integer-valued Markov-chain observations.

        Returns:
            Encoded sequence lengths, observation indexes, unique transitions,
            transition indexes, initial values, and sequence lengths.
        """
        lag = self.lag

        _cnt = len(x)
        lens = np.asarray([len(u) for u in x])
        lag_cnt = (lens >= lag).sum()
        step_cnt = np.maximum(lens - lag, 0).sum()

        init_entries = np.zeros(lag_cnt, dtype=object)
        seq_entries = np.zeros(step_cnt, dtype=object)

        init_idx: List[int] = []
        seq_idx: List[int] = []
        seq_len: List[int] = []

        i0 = 0
        i1 = 0

        for i, x_i in enumerate(x):
            xx = x_i
            seq_len.append(max(len(xx) - lag + 1, 0))

            if len(xx) < lag:
                continue

            init_idx.append(i)
            init_entries[i0] = tuple(xx[:lag])
            i0 += 1

            for j in range(len(xx) - lag):
                seq_idx.append(i)
                seq_entries[i1] = (tuple(xx[j : (j + lag)]), xx[j + lag])
                i1 += 1

        u_seq_values, u_seq_idx = np.unique(seq_entries, return_inverse=True)

        init_idx_arr = np.asarray(init_idx, dtype=np.int32)
        seq_idx_arr = np.asarray(seq_idx, dtype=np.int32)
        seq_len_arr = np.asarray(seq_len, dtype=np.int32)

        len_enc = self.len_encoder.seq_encode(seq_len_arr)
        init_enc = self.init_encoder.seq_encode(init_entries)

        rv_enc = (
            seq_len_arr,
            init_idx_arr,
            seq_idx_arr,
            u_seq_idx,
            u_seq_values,
            init_enc,
            len_enc,
        )

        return IntegerMarkovChainEncodedDataSequence(data=rv_enc)


class IntegerMarkovChainEncodedDataSequence(EncodedDataSequence):
    """Store an encoded batch of fixed-lag integer-chain observations.

    Notes:
        E = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        EncodedDataSequence, EncodedDataSequence]

    Attributes:
        data (E): Encoded sequence of integer Markov chain observations.

    """

    def __init__(
        self,
        data: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            EncodedDataSequence,
            EncodedDataSequence,
        ],
    ):
        """Initialize an encoded integer Markov-chain batch.

        Args:
            data (E): Encoded sequence of integer Markov chain observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation of the encoded batch."""
        return f"IntegerMarkovChainEncodedDataSequence(data={self.data})"
