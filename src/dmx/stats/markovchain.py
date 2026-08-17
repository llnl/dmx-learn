"""Create, estimate, and sample from a Markov chain.

Defines the MarkovChainDistribution, MarkovChainDistributionSampler,
MarkovChainDistributionAccumulatorFactory,
MarkovChainDistributionAccumulator, MarkovChainDistributionEstimator, and the
MarkovChainDistributionDataEncoder
classes for use with dmx-learn.

The assumed data type for the stats-space is T.

The density of Markov chain is given by for sequence of length n,
x=[x[0],x[1],...,x[n-1]]

    p_mat(x) = p_mat(x[0])*p_mat(x[1]|x[0])*...*p_mat(x[n-1]|x[n-2])*P_len(n)

where p_mat(x[i+1]|x[i]) is the transition probability, p_mat(x[0]) is the
init-probability, and P_len(n) is given
by the length distribution density.

Note if len(x) = 0, only log(P_len(0)) is returned.

"""

from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.random import RandomState
from scipy.sparse import dok_matrix

from dmx.arithmetic import log, maxrandint, one, zero
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

T = TypeVar("T")  ### state type
T1 = TypeVar("T1")  ### Type for length distribution sufficient statsitics value.
suff_stat_type = Tuple[Dict[T, float], Dict[T, Dict[T, float]], Optional[Any]]
enc_data_type = Tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any
]


class MarkovChainDistribution(  # pylint: disable=too-many-instance-attributes
    SequenceEncodableProbabilityDistribution, Generic[T]
):
    """MarkovChainDistribution object defining a Markov chain compatible with data type
    T.

    Attributes:
        init_prob_map (Dict[T, float]): Probability of each initial values of data type
            T.
        transition_map (Dict[T, Dict[T, float]]): Transition probability map.
        len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length
            distribution for length of
            observation sequence.
        default_value (float): Default probability for value outside support.
        name (Optional[str]): Set name to MarkovChainDistribution object.
        all_vals (Set[T]): Set of all values in state-space.
        loginit_prob_map (Dict[T, float]): Dictionary mapping initial state value to log
            probability.
        log_transition_map (Dict[T, Dict[T, float]]): Dictionary mapping state to state
            transition
            log-probabilities.
        log_dv (float): Log default value.
        log_dtv (float): Log of default value scaled by number of state-values + 1.
        log1p_dv (float): Log of 1 plus default_value.
        key_map (Dict[T, int]): Maps each state-value in all_vals to integer [1,
            len(all_vals)+1]
        inv_key_map (List[T]): List of all state-values (keys).
        num_keys (int): Number of state-values (len(keys)).
        init_log_pvec (ndarray): Log-probabilities of each initial value. Entry 0, is
            log_dv. (len == num_keys+1).
        trans_log_pvec (dok_matrix): Dictionary of keys for sparse log transition
            probabilities.
        keys (Optional[str] = None): Set keys for distribution parameters

    """

    def __init__(
        self,
        init_prob_map: Dict[T, float],
        transition_map: Dict[T, Dict[T, float]],
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        default_value: float = 0.0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """MarkovChainDistribution object.

        Args:
            init_prob_map (Dict[T, float]): Probability of each initial values of data
                type T.
            transition_map (Dict[T, Dict[T, float]]): Transition probability map.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length
                distribution for length of
                observation sequence.
            default_value (float): Default probability for value outside support.
            name (Optional[str]): Set name to MarkovChainDistribution object.
            keys (Optional[str] = None): Set keys for distribution parameters

        """
        self.name = name
        super().__init__()
        self.init_prob_map = init_prob_map
        self.transition_map = transition_map
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

        self.all_vals = (
            set(init_prob_map.keys())
            .union({v for u in transition_map.values() for v in u.keys()})
            .union(transition_map.keys())
        )
        self.loginit_prob_map = {
            u[0]: -np.inf if u[1] == 0.0 else log(u[1]) for u in init_prob_map.items()
        }

        self.log_transition_map = dict(
            (key, dict((u[0], log(u[1])) for u in transition_map[key].items()))
            for key in transition_map.keys()
        )

        self.default_value = max(min(default_value, 1.0), 0.0)
        self.log_dv = -np.inf if default_value == 0.0 else log(self.default_value)
        self.log_dtv = (
            -np.inf
            if default_value == 0.0
            else (log(default_value) - np.log(len(self.all_vals) + 1))
        )
        self.log1p_dv = log(one + self.default_value)

        num_keys = len(self.all_vals)

        val_keys = list(self.all_vals)
        sidx = np.argsort(np.asarray(val_keys, dtype=object))
        val_keys = [val_keys[i] for i in sidx]

        self.key_map = {val_keys[i]: i + 1 for i in range(num_keys)}
        self.inv_key_map = val_keys
        self.num_keys = num_keys

        self.init_log_pvec = np.zeros(num_keys + 1)
        self.trans_log_pvec = dok_matrix((num_keys + 1, num_keys + 1))

        for k1, v1 in init_prob_map.items():
            self.init_log_pvec[self.key_map.get(k1, 0)] = (
                -np.inf if v1 == 0.0 else np.log(v1)
            )

        for k1, trans_probs in transition_map.items():
            k1_idx = self.key_map.get(k1, 0)
            for k2, v2 in trans_probs.items():
                self.trans_log_pvec[k1_idx, self.key_map.get(k2, 0)] = (
                    -np.inf if v2 == 0 else np.log(v2)
                )

        self.init_log_pvec[0] = self.log_dv
        self.trans_log_pvec[:, 0] = self.log_dv
        self.trans_log_pvec[0, :] = self.log_dv - np.log(num_keys + 1)
        self.keys = keys

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarkovChainDistribution):
            return False

        names = [
            "name",
            "keys",
            "init_prob_map",
            "transition_map",
            "len_dist",
            "default_value",
        ]
        for x in names:
            try:
                v0 = getattr(other, x)
                v1 = getattr(self, x)
            except AttributeError:
                return False

            if not np.all(v0 == v1):
                return False
        return True

    def __str__(self) -> str:
        s1 = dict(sorted(self.init_prob_map.items(), key=lambda u: repr(u[0])))
        s1 = {k: float(v) for k, v in s1.items()}
        temp = sorted(self.transition_map.items(), key=lambda u: repr(u[0]))
        s2 = repr(
            {k: dict(sorted(v.items(), key=lambda u: repr(u[0]))) for k, v in temp}
        )
        s3 = str(self.len_dist)
        s4 = repr(self.default_value)
        s5 = repr(self.name)
        s6 = repr(self.keys)

        return (
            f"MarkovChainDistribution({s1}, {s2}, len_dist={s3}, default_value={s4}, "
            f"name={s5}, keys={s6})"
        )

    def density(self, x: List[T]) -> float:
        """Return density of MarkovChainDistribution at observed sequence x.

        Returns exponential of log_density(x). See log_density() for details.

        Args:
            x (List[T]): An observed Markov chain sequence of data type T.

        Returns:
            float: Density of Markov chain at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: List[T]) -> float:
        """Return log-density of MarkovChainDistribution at observed sequence x.

        Args:
            x (List[T]): An observed Markov chain sequence of data type T.

        Returns:
            float: Log-density of Markov chain at x.

        """
        if len(x) == 0:
            rv = 0.0
        else:
            rv = self.loginit_prob_map.get(x[0], self.log_dv) - self.log1p_dv

            for i in range(1, len(x)):
                if x[i - 1] in self.log_transition_map:
                    rv += (
                        self.log_transition_map[x[i - 1]].get(x[i], self.log_dv)
                        - self.log1p_dv
                    )
                else:
                    rv += self.log_dtv - self.log1p_dv

        rv += self.len_dist.log_density(len(x))

        return float(rv)

    def seq_log_density(self, x: "MarkovChainEncodedDataSequence") -> np.ndarray:
        if not isinstance(x, MarkovChainEncodedDataSequence):
            raise TypeError(
                "MarkovChainEncodedDataSequence required for seq_log_density()."
            )

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x.data

        loc_key_map = np.asarray([self.key_map.get(u, 0) for u in inv_key_map])

        temp = (
            self.trans_log_pvec[loc_key_map[prev_x], loc_key_map[next_x]]
            .toarray()
            .flatten()
            - self.log1p_dv
        )
        rv = np.bincount(idx1, weights=temp, minlength=sz)
        rv[idx0] += self.init_log_pvec[loc_key_map[init_x]] - self.log1p_dv

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> "MarkovChainSampler":
        return MarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "MarkovChainEstimator":

        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        return MarkovChainEstimator(
            pseudo_count=pseudo_count,
            len_estimator=len_est,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "MarkovChainDataEncoder":
        len_encoder = self.len_dist.dist_to_encoder()
        return MarkovChainDataEncoder(len_encoder=len_encoder)


class MarkovChainSampler(DistributionSampler, Generic[T]):
    """MarkovChainSampler object for sampling from Markov chain.

    Attributes:
        rng (RandomState): RandomState obejct for setting seed of random sampler.
        init_prob (Tuple[List[T], List[float]): Tuple of initial state-values and
            probabilities.
        trans_prob (Dict[T, Tuple[List[T], List[float]]]): Dictionary mapping transition
            probabilties from state i
            to state j.
        len_sampler (DistributionSampler): Sample length of Markov chain sequence. Must
            be a DistributionSampler
            with support on non-negative integers.

    """

    def __init__(
        self, dist: "MarkovChainDistribution[T]", seed: Optional[int] = None
    ) -> None:
        """MarkovChainSampler object.

        Args:
            dist (MarkovChainDistribution): Instance of MarkovChainDistribution object
                to sample from.
            seed (Optional[int]): Set seed of random number generator for sampling from
                Markov chain.

        """
        super().__init__(dist, seed)

        loc_trans = list(dist.init_prob_map.items())
        loc_probs = [v[1] for v in loc_trans]
        loc_keys = [v[0] for v in loc_trans]

        self.init_prob: Tuple[List[T], List[float]] = (loc_keys, loc_probs)

        self.trans_prob: Dict[T, Tuple[List[T], List[float]]] = {}
        for k, v in dist.transition_map.items():
            loc_trans = list(v.items())
            loc_probs = [v[1] for v in loc_trans]
            loc_keys = [v[0] for v in loc_trans]
            self.trans_prob[k] = (loc_keys, loc_probs)

        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def _choice(self, levels: List[T], probs: List[float]) -> T:
        rv = self.rng.choice(np.asarray(levels, dtype=object), p=probs)
        return cast(T, rv.item() if hasattr(rv, "item") else rv)

    def _sample_single(self) -> List[T]:
        cnt = int(self.len_sampler.sample())
        rv: List[T] = []

        if cnt >= 1:
            rv.append(self._choice(self.init_prob[0], self.init_prob[1]))

        for i in range(1, cnt):
            curr_k, curr_p = self.trans_prob[rv[i - 1]]
            rv.append(self._choice(curr_k, curr_p))

        return rv

    def sample(self, size: Optional[int] = None) -> Union[List[T], List[List[T]]]:
        """Draw iid samples from Markov chain distribution.

        If size is None, sample N from len_sampler() and return a List[T] of length N,
        where T is the data type of
        the Markov chain. If size > 0, return a list of length size, containing List[T]
        data types.

        Args:
            size (Optional[int]): Number of samples to draw. Draws 1 sample if None.

        Returns:
            List[T] or List[List[T]], depending on size arg.

        """
        if size is not None:
            return [self._sample_single() for i in range(size)]

        return self._sample_single()

    def sample_seq(
        self, size: Optional[int] = None, v0: Optional[T] = None
    ) -> Union[T, List[T]]:
        """Sample a Markov chain sequence of length 'size' conditioned on initial state
        'v0'.

        If size is None, draw a sequence of length 1, returning as type T.

        If size is not None, draw a sequence of length size, returning as type List[T].

        If v0 is None, v0 is sampled from member variable 'init_prob'.

        Args:
            size (Optional[int]): Length of Markov chain sequence to sample.
            v0 (Optional[T]): Initial state of Markov chain sequence to sample from.

        Returns:
            T or List[T] depending on arg size.

        """
        if size is not None:

            rv: List[Any] = [None] * size

            prev_val = v0

            if size > 0 and prev_val is None:
                rv[0] = self._choice(self.init_prob[0], self.init_prob[1])
                prev_val = rv[0]

            for i in range(1, size):

                if prev_val not in self.trans_prob:
                    break

                levels, probs = self.trans_prob[prev_val]
                rv[i] = self._choice(levels, probs)
                prev_val = rv[i]

            return cast(List[T], rv)

        prev_val = v0

        if prev_val is None:
            single = self._choice(self.init_prob[0], self.init_prob[1])
        else:
            levels, probs = self.trans_prob[prev_val]
            single = self._choice(levels, probs)

        return single


class MarkovChainAccumulator(SequenceEncodableStatisticAccumulator, Generic[T]):
    """MarkovChainAccumulator object for accumulating sufficient statistics from
    observed data.

    Attributes:
        init_count_map (Dict[T, float]): Dictionary for accumulating weighted counts of
            initial states.
        trans_count_map (Dict[T, Dict[T, float]]): Dictionary for accumulating weighted
            counts of state to state
            transitions
        len_accumulator (SequenceEncodableStatisticAccumulator):
            SequenceEncodableStatisticAccumulator
            object for accumulating sufficient statistics of length distribution for
            length of Markov sequences.
            Set to NullAccumulator() if no length distribution is to be estimated.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Keys for merging sufficient statistics of
            MarkovChainAccumulator.

    """

    def __init__(
        self,
        len_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """MarkovChainAccumulator object.

        Args:
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]):
                SequenceEncodableStatisticAccumulator
                object for accumulating sufficient statistics of length distribution for
                length of Markov sequences.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for merging sufficient statistics of
                MarkovChainAccumulator.

        """
        self.init_count_map: Dict[T, float] = {}
        self.trans_count_map: Dict[T, Dict[T, float]] = {}
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator()
        )
        self.name = name
        self.keys = keys

    def update(
        self, x: List[T], weight: float, estimate: MarkovChainDistribution
    ) -> None:
        if x is not None:
            self.len_accumulator.update(len(x), weight, estimate.len_dist)

        if x is not None and len(x) != 0:
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = {}

                self.trans_count_map[x0][u] = (
                    self.trans_count_map[x0].get(u, zero) + weight
                )
                x0 = u

    def initialize(self, x: List[T], weight: float, rng: RandomState) -> None:
        if x is not None:
            self.len_accumulator.initialize(len(x), weight, rng)

        if x is not None and len(x) != 0:
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = {}

                self.trans_count_map[x0][u] = (
                    self.trans_count_map[x0].get(u, zero) + weight
                )
                x0 = u

    def seq_initialize(
        self, x: "MarkovChainEncodedDataSequence", weights: np.ndarray, rng: RandomState
    ) -> None:
        _sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x.data
        self.len_accumulator.seq_initialize(len_enc, weights, rng)

        init_count = np.bincount(init_x, weights=weights[idx0])

        for i, init_count_i in enumerate(init_count):
            v = init_count_i
            if v != 0:
                self.init_count_map[inv_key_map[i]] = (
                    self.init_count_map.get(inv_key_map[i], 0.0) + v
                )
        # ------------- slow and sparse...

        for i, prev_x_i in enumerate(prev_x):
            k1 = inv_key_map[prev_x_i]
            k2 = inv_key_map[next_x[i]]
            ww = weights[idx1[i]]

            if k1 not in self.trans_count_map:
                self.trans_count_map[k1] = {k2: ww}
            else:
                m = self.trans_count_map[k1]
                m[k2] = m.get(k2, 0.0) + ww

        # ------------- slow and sparse...

    def seq_update(
        self,
        x: "MarkovChainEncodedDataSequence",
        weights: np.ndarray,
        estimate: MarkovChainDistribution,
    ) -> None:

        _sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x.data

        init_count = np.bincount(init_x, weights=weights[idx0])

        for i, init_count_i in enumerate(init_count):
            v = init_count_i
            if v != 0:
                self.init_count_map[inv_key_map[i]] = (
                    self.init_count_map.get(inv_key_map[i], 0.0) + v
                )
        # ------------- slow and sparse...

        for i, prev_x_i in enumerate(prev_x):
            k1 = inv_key_map[prev_x_i]
            k2 = inv_key_map[next_x[i]]
            ww = weights[idx1[i]]

            if k1 not in self.trans_count_map:
                self.trans_count_map[k1] = {k2: ww}
            else:
                m = self.trans_count_map[k1]
                m[k2] = m.get(k2, 0.0) + ww

        # ------------- slow and sparse...
        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(self, suff_stat: suff_stat_type) -> "MarkovChainAccumulator":

        for item in suff_stat[0].items():
            self.init_count_map[item[0]] = (
                self.init_count_map.get(item[0], 0.0) + item[1]
            )

        for trans_item in suff_stat[1].items():
            if trans_item[0] not in self.trans_count_map:
                self.trans_count_map[trans_item[0]] = {}

            item_map = self.trans_count_map[trans_item[0]]
            for elem in trans_item[1].items():
                item_map[elem[0]] = item_map.get(elem[0], 0.0) + elem[1]

        self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> suff_stat_type:

        return self.init_count_map, self.trans_count_map, self.len_accumulator.value()

    def from_value(self, x: suff_stat_type) -> "MarkovChainAccumulator":

        self.init_count_map = x[0]
        self.trans_count_map = x[1]
        self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, "MarkovChainAccumulator"]) -> None:

        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())

            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, "MarkovChainAccumulator"]) -> None:

        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "MarkovChainDataEncoder":
        len_encoder = self.len_accumulator.acc_to_encoder()
        return MarkovChainDataEncoder(len_encoder=len_encoder)


class MarkovChainAccumulatorFactory(StatisticAccumulatorFactory, Generic[T]):
    """MarkovChainAccumulatorFactory object for creating MarkovChainAccumulator objects.

    Attributes:
        len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory object
            for the length distribution
            of Markov chain sequences. Set to NullAccumulatorFactory if no length
            distribution is to be estimated.
        name (Optional[str]): Name for object.
        keys (Optional[str]): Keys for merging sufficient statistics of
            MarkovChainAccumulator.

    """

    def __init__(
        self,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """MarkovChainAccumulatorFactory object.

        Args:
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory
                object for the length distribution
                of Markov chain sequences.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Set keys for merging sufficient statistics of
                MarkovChainAccumulator.

        """
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self) -> "MarkovChainAccumulator[T]":
        len_acc = self.len_factory.make()
        return MarkovChainAccumulator(
            len_accumulator=len_acc, keys=self.keys, name=self.name
        )


class MarkovChainEstimator(ParameterEstimator, Generic[T]):
    """MarkovChainEstimator object for estimating MarkovChainDistribution object from
    aggregated data.

    Notes:
        With ``pseudo_count=None``, initial-state and transition probabilities are
        normalized from observed counts only. With a pseudo-count, the estimator builds
        a smoothing support from observed initial states, observed transition targets,
        and any values supplied in ``levels``. Unseen states receive pseudo-count mass
        only if they are included in that support, so pass ``levels`` when a known
        state space should be available during initialization or sparse fitting.

        The length distribution is estimated by ``len_estimator`` and is not smoothed
        by the Markov-chain pseudo-count.

    Attributes:
        pseudo_count (Optional[float]): Smoothing mass for initial-state and transition
            probabilities.
        levels (Optional[Iterable[T]]): Optional state values to include in the
            smoothing support.
        len_estimator (ParameterEstimator): NullEstimator if no length distribution is
            to be estimated.
        name (Optional[str]): Name for instance of MarkovChainEstimator.
        keys (Optional[str]): Keys for merging sufficient statistics of
            MarkovChainAccumulator objects.

    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        levels: Optional[Iterable[T]] = None,
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """MarkovChainEstimator object.

        Args:
            pseudo_count (Optional[float]): Smoothing mass for initial-state and
                transition probabilities. If set, mass is spread uniformly over the
                support from observed states plus ``levels``.
            levels (Optional[Iterable[T]]): Optional state values to include in the
                smoothing support even if they are not observed.
            len_estimator (Optional[ParameterEstimator]): ParameterEstimator for length
                of Markov sequences.
            name (Optional[str]): Set a name for instance of MarkovChainEstimator.
            keys (Optional[str]): Set keys for merging sufficient statistics of
                MarkovChainAccumulator objects.

        """
        self.name = name
        self.pseudo_count = pseudo_count
        self.levels = levels
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.keys = keys

    def accumulator_factory(self) -> "MarkovChainAccumulatorFactory[T]":
        return MarkovChainAccumulatorFactory(
            len_factory=self.len_estimator.accumulator_factory(),
            keys=self.keys,
            name=self.name,
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: suff_stat_type
    ) -> "MarkovChainDistribution[T]":
        if self.pseudo_count is not None:
            return self.estimate1(nobs, suff_stat)
        return self.estimate0(nobs, suff_stat)

    def estimate0(
        self, nobs: Optional[float], suff_stat: suff_stat_type
    ) -> "MarkovChainDistribution":
        """Estimate MarkovChainDistribution from aggregated sufficient statistics from
        observed data.

        Maximum likelihood estimates for initial state probabilities, transition
        probabilities, and the length
        distribution are obtained directly from aggregated data in 'suff_stat'.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their aggregated
            counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition
            counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator.
            (Assumed type T1).

        Args:
            nobs (Optional[float]): Number of observations. Passed to estimate1() or
                estimate2().
            suff_stat: Seed above for details.

        Returns:
            MarkovChainDistribution object.

        """
        temp_sum = sum(suff_stat[0].values())
        init_prob_map = {k: v / temp_sum for k, v in suff_stat[0].items()}

        trans_map: Dict[T, Dict[T, float]] = {}

        for key, tmap in suff_stat[1].items():
            temp_sum = sum(tmap.values())
            if temp_sum > 0:
                trans_map[key] = {k: v / temp_sum for k, v in tmap.items()}

        len_dist = self.len_estimator.estimate(nobs, suff_stat[2])

        return MarkovChainDistribution(
            init_prob_map, trans_map, len_dist=len_dist, name=self.name
        )

    def estimate1(
        self, nobs: Optional[float], suff_stat: suff_stat_type
    ) -> "MarkovChainDistribution[T]":
        """Estimate MarkovChainDistribution from aggregated sufficient statistics from
        observed data.

        Maximum likelihood estimates for initial state probabilities, transition
        probabilities, and the length
        distribution are obtained by a weighted aggregation of sufficient statistics in
        'suff_stat', and member
        variables of MarkovChainEstimator object.

        The smoothing support is the union of observed initial states, observed
        transition targets, and ``levels``. States absent from that union do not receive
        pseudo-count mass.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their aggregated
            counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition
            counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator.
            (Assumed type T1).

        Args:
            nobs (Optional[float]): Number of observations. Passed to estimate1() or
                estimate2().
            suff_stat: Seed above for details.

        Returns:
            MarkovChainDistribution object.

        """
        trans_map: Dict[T, Dict[T, float]] = {}
        init_prob_map: Dict[T, float] = {}
        def_val = 0.0

        all_keys = set(suff_stat[0].keys())
        for u in suff_stat[1].values():
            all_keys.update(u.keys())
        if self.levels is not None:
            all_keys.update(self.levels)

        temp_sum = sum(suff_stat[0].values())
        p_cnt0 = self.pseudo_count if self.pseudo_count is not None else 0.0
        p_cnt1 = p_cnt0 / len(all_keys)

        if (temp_sum + p_cnt0) > 0:
            init_prob_map = {
                k: (suff_stat[0].get(k, 0.0) + p_cnt1) / (temp_sum + p_cnt0)
                for k in all_keys
            }

        a_sum = temp_sum
        for key, tmap in suff_stat[1].items():
            temp_sum = sum(tmap.values())
            a_sum += temp_sum
            if (temp_sum + p_cnt0) > 0:
                trans_map[key] = {
                    k: (tmap.get(k, 0.0) + p_cnt1) / (temp_sum + p_cnt0)
                    for k in all_keys
                }

        len_dist = self.len_estimator.estimate(nobs, suff_stat[2])

        if a_sum > 0:
            def_val = p_cnt0 / a_sum

        return MarkovChainDistribution(
            init_prob_map,
            trans_map,
            len_dist=len_dist,
            default_value=def_val,
            name=self.name,
        )


class MarkovChainDataEncoder(DataSequenceEncoder, Generic[T]):
    """MarkovChainDataEncoder used for sequence encoding data for use with vectorized
    'seq_' functions.

    Attributes:
          len_encoder (DataSequenceEncoder): DataSequenceEncoder object that has support
              on non-negative integers.
            Is set to NullDataEncoder() if no length distribution is to be estimated.
    """

    def __init__(self, len_encoder: DataSequenceEncoder = NullDataEncoder()) -> None:
        """MarkovChainDataEncoder object.

        Args:
            len_encoder (DataSequenceEncoder): DataSequenceEncoder with data type int.

        """
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        return "MarkovChainDataEncoder(len_encoder=" + str(self.len_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MarkovChainDataEncoder):
            return other.len_encoder == self.len_encoder
        return False

    def seq_encode(self, x: List[List[T]]) -> "MarkovChainEncodedDataSequence":
        """Sequence encoding a sequence of iid Markov chain observations with data type
        T.

        The returned value is (rv) is a Tuple of length 8 with entries:

            rv[0] (int): Number of total observations (number of Markov sequences).
            rv[1] (ndarray[int]): Sequence index for initial state observations.
            rv[2] (ndarray[int]): Sequence index for non-initial state observations in a
            sequence greater than len 1.
            rv[3] (ndarray[int]): Numpy array of observations index in inv_key_map for
            initial states.
            rv[4] (ndarray[int]): State-to-state index value of inv_key_map for initial
            state value.
            rv[5] (ndarray[int]): State-to-state index value of inv_key_map for
            transition.
            rv[6] (ndarray[T]): Maps integer index value to value in state-space (T).
            rv[7] (Optional[T1]): Encoded sequence of lengths from len_encoder. None if
            no length distributon to be
                estimated.

        Args:
            x (List[List[T]]): Sequence of iid observations of Markov chain sequences.

        Returns:
            MarkovChainEncodedDataSequence

        """

        init_entries: List[int] = []
        pair_entries: List[Tuple[int, int]] = []
        entries_idx0: List[int] = []
        entries_idx1: List[int] = []
        obs_cnt: List[int] = []
        key_map: Dict[T, int] = {}

        for i, x_i in enumerate(x):
            entry = x_i
            obs_cnt.append(len(entry))

            if len(entry) == 0:
                continue

            if entry[0] not in key_map:
                key_map[entry[0]] = len(key_map)

            prev_idx = key_map[entry[0]]
            init_entries.append(prev_idx)
            entries_idx0.append(i)

            for j in range(1, len(entry)):

                if entry[j] not in key_map:
                    key_map[entry[j]] = len(key_map)
                next_idx = key_map[entry[j]]

                pair_entries.append((prev_idx, next_idx))
                entries_idx1.append(i)
                prev_idx = next_idx

        obs_cnt_arr = np.asarray(obs_cnt, dtype=np.int32)
        init_entries_arr = np.asarray(init_entries, dtype=np.int32)
        pair_entries_arr = np.asarray(pair_entries, dtype=np.int32).reshape((-1, 2))
        entries_idx0_arr = np.asarray(entries_idx0, dtype=np.int32)
        entries_idx1_arr = np.asarray(entries_idx1, dtype=np.int32)

        inv_key_map_list: List[Any] = [None] * len(key_map)
        for k, v in key_map.items():
            inv_key_map_list[v] = k
        inv_key_map = np.asarray(inv_key_map_list, dtype=object)

        len_enc = self.len_encoder.seq_encode(obs_cnt_arr)

        rv_enc = (
            len(x),
            entries_idx0_arr,
            entries_idx1_arr,
            init_entries_arr,
            pair_entries_arr[:, 0],
            pair_entries_arr[:, 1],
            inv_key_map,
            len_enc,
        )

        return MarkovChainEncodedDataSequence(data=rv_enc)


class MarkovChainEncodedDataSequence(EncodedDataSequence):
    """MarkovChainEncodedDataSequence object for vectorized function calls.

    Notes:
        E = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, EncodedDataSequence]

    Attributes:
        data (E): Encoded sequence of markov chain iid observations.

    """

    def __init__(
        self,
        data: Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            EncodedDataSequence,
        ],
    ):
        """MarkovChainEncodedDataSequence object.

        Args:
            data (E): Encoded sequence of markov chain iid observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f"MarkovChainEncodedDataSequence(data={self.data})"
