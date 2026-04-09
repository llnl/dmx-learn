""""Create, estimate, and sample from a hidden markov model with integer emission distributions"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numba
import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import *
from dmx.arithmetic import maxrandint
from dmx.stats.markovchain import MarkovChainDistribution
from dmx.stats.mixture import MixtureDistribution
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

T = TypeVar("T")
T1 = TypeVar("T1")  # Emission suff-stat type
T2 = TypeVar("T2")  # Len suff-stat type

E0 = Tuple[
    Tuple[
        int,
        List[Tuple[int, int]],
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        EncodedDataSequence,
    ],
    EncodedDataSequence,
    EncodedDataSequence,
]
E = Tuple[np.ndarray, np.ndarray, EncodedDataSequence]


class IntegerHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden Markov Model distribution with integer emission distributions.

    A hidden Markov model (HMM) with integer emission distributions. This class implements
    a probability distribution over sequences of integers, where the probability of each
    integer in the sequence depends on a hidden state that follows a Markov process.

    If a length distribution for the HMM sequence is included, it must have data type int
    with support of non-negative integers.

    Attributes:
        pmat (np.ndarray): Emission probability matrix where pmat[i,j] is the probability
            of emitting integer i from state j.
        log_pmat (np.ndarray): Log of emission probability matrix.
        n_words (int): Number of possible integer emissions (vocabulary size).
        n_states (int): Number of hidden states.
        w (np.ndarray): Initial state probabilities.
        log_w (np.ndarray): Initial state log-probabilities.
        transitions (np.ndarray): 2-d array of hidden state transition probabilities
            with shape (n_states, n_states).
        log_transitions (np.ndarray): Log of transition probabilities.
        terminal_values (Optional[Set[T]]): Set of values that terminate the HMM sequence
            when sampling.
        name (Optional[str]): Name of the distribution instance.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution for sequence
            lengths. Defaults to NullDistribution.
        keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states,
            transitions counts, and emission distributions.
    """

    def __init__(
        self,
        pmat: Union[List[List[float]], np.ndarray],
        w: Union[Sequence[float], np.ndarray],
        transitions: Union[List[List[float]], np.ndarray],
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        terminal_values: Optional[Set[T]] = None,
    ) -> None:
        """Initialize an IntegerHiddenMarkovModelDistribution.

        Args:
            pmat: Emission probability matrix where pmat[i,j] is the probability of
                emitting integer i from state j.
            w: Initial state probabilities.
            transitions: 2-d array of hidden state transition probabilities with shape
                (n_states, n_states).
            len_dist: Distribution for sequence lengths. Defaults to NullDistribution.
            name: Name of the distribution instance. Defaults to None.
            keys: Keys for initial states, transitions counts, and emission distributions.
                Defaults to (None, None, None).
            terminal_values: Set of values that terminate the HMM sequence when sampling.
                Defaults to None.
        """

        with np.errstate(divide="ignore"):

            if not isinstance(pmat, np.ndarray):
                pmat = np.ndarray(pmat)

            self.pmat = pmat
            self.log_pmat = np.log(pmat)
            self.n_words = pmat.shape[0]
            self.n_states = pmat.shape[1]
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(transitions, (self.n_states, self.n_states))
            self.log_transitions = np.log(self.transitions)

        self.terminal_values = terminal_values
        self.name = name
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys

    def __str__(self) -> str:
        s1 = repr(self.pmat.tolist())
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.terminal_values)
        s7 = repr(self.keys)

        return (
            "IntegerHiddenMarkovModelDistribution(pmat=pmat, w=%s, transitions=%s, taus=%s, len_dist=%s, name=%s, terminal_values=%s, "
            "keys=%s)" % (s1, s2, s3, s4, s5, s6, s7)
        )

    def density(self, x: Sequence[int]) -> float:
        """Returns the density of HMM for an observed sequence x.

        See 'IntegerHiddenMarkovDistribution.log_density()' for details.

        Args:
            x (Sequence[int]): Observed sequence of HMM emissions.

        Returns:
            float: Density of HMM for observed sequence x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[int]) -> float:
        """Returns the log-density of HMM for observed sequence x.

        Args:
            x (Sequence[int]): Observed sequence of HMM emissions.

        Returns:
            float: Log-density of observed HMM sequence x.

        """
        if x is None or len(x) == 0:
            return self.len_dist.log_density(0)

        # Initialize: log_alpha = log(w) + log(P(x[0] | state))
        log_alpha = self.log_w + self.log_pmat[x[0], :]

        # Forward recursion
        for t in range(1, len(x)):
            # Vectorized log-sum-exp over previous states
            # log_alpha_new[s] = log(sum_s' exp(log_alpha[s'] + log_A[s', s]))
            log_terms = log_alpha[:, np.newaxis] + self.log_transitions  # (S, S)
            max_vals = np.max(log_terms, axis=0)  # (S,)
            log_alpha = max_vals + np.log(np.sum(np.exp(log_terms - max_vals), axis=0))

            # Add emission probabilities
            log_alpha += self.log_pmat[x[t], :]

        # Final probability
        max_val = np.max(log_alpha)
        ret_val = max_val + np.log(np.sum(np.exp(log_alpha - max_val)))

        ret_val += self.len_dist.log_density(len(x))

        return ret_val

    def seq_log_density(
        self, x: "IntegerHiddenMarkovEncodedDataSequence"
    ) -> np.ndarray:

        # enc_data is just numpy array of integers

        if not isinstance(x, IntegerHiddenMarkovEncodedDataSequence):
            raise Exception("Requires IntegerHiddenMarkovEncodedDataSequence.")

        enc_data, tz, len_enc = x.data

        ll_ret = vec.zeros(len(tz) - 1)
        log_pmat = self.log_pmat

        fast_log_density(
            xs=enc_data,
            tz=tz,
            log_pmat=log_pmat,
            log_A=self.log_transitions,
            log_pi=self.log_w,
            out=ll_ret,
        )

        ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def seq_posterior(
        self, x: "IntegerHiddenMarkovEncodedDataSequence"
    ) -> List[np.ndarray]:
        """Compute posterior distribution for each latent state of a sequence.

        Args:
            x (IntegerHiddenMarkovEncodedDataSequence): Numba encoded sequence of HMM observations.

        Returns:
            List[np.ndarray]: A list of posterior probabilities for each latent state for each observation sequence.

        """

        if not isinstance(x, IntegerHiddenMarkovEncodedDataSequence):
            raise Exception(
                "Requires IntegerHiddenMarkovEncodedDataSequence for seq_posterior"
            )

        enc_data, tz, len_enc = x.data

        tot_cnt = len(enc_data)
        seq_cnt = len(tz) - 1
        num_states = self.n_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        weights = np.ones(seq_cnt, dtype=np.float64)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        pr_obs += self.log_pmat[enc_data]

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(
            num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc
        )

        return [alphas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: Sequence[int]) -> np.ndarray:
        """Returns the viterbi sequence for an HMM observation.

        Args:
            x (Sequence[int]): Single HMM sequence.

        Returns:
            np.ndarray of most likely state sequence
        """
        if len(x) == 0:
            return np.array([], dtype=int)

        num_states = len(self.log_w)
        sz = len(x)

        # Initialize
        log_delta = self.log_w + self.log_pmat[x[0], :]
        psi = np.zeros((len(x), num_states), dtype=int)

        # Store log_delta at each time step for backtracking
        all_log_delta = np.zeros((sz, num_states))
        all_log_delta[0, :] = log_delta

        # Forward pass (vectorized)
        for t in range(1, sz):
            # Compute all transition scores: (S_prev, S_curr)
            log_trans_probs = log_delta[:, np.newaxis] + self.log_transitions  # (S, S)

            # Find best previous state for each current state
            psi[t, :] = np.argmax(log_trans_probs, axis=0)
            log_delta = np.max(log_trans_probs, axis=0) + self.log_pmat[x[t], :]
            all_log_delta[t, :] = log_delta

        # Backward pass
        states = np.zeros(sz, dtype=int)
        states[sz - 1] = np.argmax(all_log_delta[sz - 1, :])

        for t in range(sz - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def sampler(self, seed: Optional[int] = None) -> "IntegerHiddenMarkovSampler":
        if isinstance(self.len_dist, NullDistribution) and self.terminal_values is None:
            raise Exception(
                "IntegerHiddenMarkovSampler requires len_dist with support on non-negative integers, or terminal_"
                "values to be set."
            )

        return IntegerHiddenMarkovSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerHiddenMarkovEstimator":

        len_est = (
            None
            if self.len_dist is None
            else self.len_dist.estimator(pseudo_count=pseudo_count)
        )

        return IntegerHiddenMarkovEstimator(
            max_val=self.n_words,
            pseudo_count=(pseudo_count, pseudo_count),
            len_estimator=len_est,
            name=self.name,
            keys=self.keys,
            use_numba=self.use_numba,
        )

    def dist_to_encoder(self) -> "IntegerHiddenMarkovDataEncoder":

        len_encoder = self.len_dist.dist_to_encoder()

        return IntegerHiddenMarkovDataEncoder(len_encoder=len_encoder)


class IntegerHiddenMarkovSampler(DistributionSampler):
    """HiddenMarkovSampler object for sampling from HMM.

    If 'dist.len_dist' is set, samples HMM sequences with sequence lengths generated from 'len_dist'. If
    'dist.len_dist' is NullDistribution, 'dist.terminal_values' is must be set. Samples are generated until
    a terminal value is reached.

    Attributes:
        num_states (int): Number of hidden states in 'dist' object.
        dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
        rng (RandomState): RandomState object with seed set for sampling.
        obs_samplers (List[DistributionSampler]): List of DistributionSampler objects corresponding to the emission
            distributions of 'dist'. Taken to be MixtureSampler objects if 'dist.has_topics' is True.
        len_sampler (Optional[DistributionSampler]): DistributionSampler object with data type int and support on
            non-negative integers for sampling HMM observation sequence lengths.
        terminal_set (Optional[Set[T]]): Set of values to terminate HMM sampling when calling 'sample_seq()'.
        state_sampler (MarkovChainSampler): MarkovChainSampler for sampling states of HMM.

    """

    def __init__(
        self, dist: "IntegerHiddenMarkovModelDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize an IntegerHiddenMarkovSampler.

        Args:
            dist: IntegerHiddenMarkovModelDistribution object instance to sample from.
            seed: Seed for the random number generator. Defaults to None.
        """
        self.pmat = dist.pmat
        self.range = range(dist.n_words)
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(
                seed=self.rng.randint(0, maxrandint)
            )
        else:
            self.len_sampler = None

        if dist.terminal_values is None:
            self.terminal_set = None
        else:
            self.terminal_set = set(dist.terminal_values)

        t_map = {
            i: {k: dist.transitions[i, k] for k in range(dist.n_states)}
            for i in range(dist.n_states)
        }
        p_map = {i: dist.w[i] for i in range(dist.n_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(
            seed=self.rng.randint(0, maxrandint)
        )

    def sample_int(self, i: int) -> int:
        """Sample an integer obs from a given state.
        Args:
            i (int): State index
        Returns:
            integer observation
        """

        rv: int = self.rng.choice(self.range, p=self.pmat[:, i]).item()
        return rv

    def sample_seq(
        self, size: Optional[int] = None
    ) -> Union[list[int], list[list[int]]]:
        """Sample iid HMM sequences.

        If size is None, 1 sample is drawn and a List[T] is returned. If size > 0, 'size' samples are drawn and a List
        of length 'size' with HMM sequences (List[T]) is returned.

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.

        Returns:
            List[T] or List[List[T]] depending on size arg.

        """
        if size is None:
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)
            obs_seq = [self.sample_int(i) for i in range(n)]

            return obs_seq

        else:
            n = self.len_sampler.sample(size=size)
            state_seq = [self.state_sampler.sample_seq(size=nn) for nn in n]
            obs_seq = [[self.sample_int(j) for j in nn] for nn in state_seq]

            return obs_seq

    def sample_terminal(self, terminal_set: Set[T]) -> List[int]:
        """Sample an HMM sequence, until a terminal value is samples from the emission distribution.

        Args:
            terminal_set (Set[int]): Set values to terminate the HMM sequence.

        Returns:
            List[int] with length determined by samples to reach the first terminating value.

        """
        z = self.state_sampler.sample_seq()
        rv = [self.sample_int(z)]

        while rv[-1] not in terminal_set:
            z = self.state_sampler.sample_seq(v0=z)
            rv.append(self.sample_int(z))

        return rv

    def sample(self, size: Optional[int] = None) -> Union[list[int], list[list[int]]]:
        """Draw iid samples from HMM.

        If a 'len_sampler' is set, call 'sample_seq()' (See HiddenMarkovSampler.sample_seq() for details).
        If 'len_sampler' is the NullDistributionSampler(), 'sample_terminal()' is called. (See
        HiddenMarkovSampler.sample_terminal() for details).

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.

        Returns:
            List[int] or List[List[int]] depending on arg size.

        """
        if self.len_sampler is not None:
            return self.sample_seq(size=size)

        elif self.terminal_set is not None:
            if size is None:
                return self.sample_terminal(self.terminal_set)
            else:
                return [self.sample_terminal(self.terminal_set) for i in range(size)]

        else:
            raise RuntimeError(
                "IntegerHiddenMarkovSampler requires either a length distribution or terminal value set."
            )


class IntegerHiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):
    """IntegerHiddenMarkovAccumulator object for aggregating sufficient statistics from HMM observations.

    Attributes:
        num_words (int): Number of words / max val of vocab.
        wcnts (np.ndarray): Suff-stat for the topic distributions
        num_states (int): Total number of hidden states.
        init_counts (ndarray): Track gamma_i(0), or first time point gamma for each component in Baum-Welch.
        trans_counts (ndarray): 2-d matrix tracking transition updates from Baum-Welch
            (sum_t psi_ij(t) / sum_t gamma_i(t)).
        state_counts (ndarray): Expected number of times state is observed in sequence from t=0 to t=T-2.
        len_accumulator (SequenceEncodableStatisticAccumulator): SequenceEncodableStatisticAccumulator
            object for the length distribution. Set to NullAccumulator is None is passed.
        init_key (Optional[str]): Key for initial states.
        trans_key (Optional[str]): Key for state transitions.
        state_key (Optional[str]): Key for emission accumulators..
        name (Optional[str]): Name for object.

        _init_rng (bool): True if RandomState objects have been initialized
        _len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
        _idx_rng (Optional[RandomState]): RandomState for initializing initial state draws.

    """

    def __init__(
        self,
        num_words: int,
        num_states: int,
        len_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
        keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        name: Optional[str] = None,
    ) -> None:
        """Initialize an IntegerHiddenMarkovAccumulator.

        Args:
            num_words: Number of words / max val of vocabulary.
            num_states: Number of latent states.
            len_accumulator: Accumulator for the length distribution. Defaults to NullAccumulator.
            use_numba: Whether to use numba for sequence encodings. Defaults to False.
            keys: Keys for initial states, transition counts, and emission accumulators.
                Defaults to (None, None, None).
            name: Name for the object instance. Defaults to None.
        """
        self.num_words = num_words
        self.num_states = num_states

        self.wcnts = np.zeros((self.num_words, self.num_states))

        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator()
        )

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.name = name

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: Optional[RandomState] = None
        self._idx_rng: Optional[RandomState] = None

    def update(
        self,
        x: Sequence[int],
        weight: float,
        estimate: IntegerHiddenMarkovModelDistribution,
    ) -> None:
        """Update the accumulator with a single HMM observation sequence.

        Args:
            x (Sequence[int]): Observed sequence.
            weight (float): Weight for the observation.
            estimate (HiddenMarkovModelDistribution): Current HMM distribution estimate.
        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for accumulator components.

        Args:
            rng (RandomState): Random number generator.
        """

        rng_seeds = rng.randint(maxrandint, size=2)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._init_rng = True

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a single HMM observation sequence.

        Args:
            x (Sequence[int]): Observed sequence.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator.
        """

        if not self._init_rng:
            self._rng_initialize(rng)

        n = len(x)

        self.len_accumulator.initialize(n, weight, self._len_rng)

        if n > 0:

            idx = self._idx_rng.choice(self.num_states, size=n)
            xs = np.asarray(x, dtype=int)
            self.init_counts[idx[0]] += weight
            self.state_counts[idx[0]] += weight
            self.wcnts[xs, idx] += weight

            if n > 1:
                for i in range(1, n):
                    self.trans_counts[idx[i - 1], idx[i]] += weight
                    self.state_counts[idx[i]] += weight

    def seq_initialize(
        self,
        x: "IntegerHiddenMarkovEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Vectorized initialization for encoded HMM data.

        Args:
            x (HiddenMarkovEncodedDataSequence): Encoded HMM data sequence.
            weights (np.ndarray): Weights for each sequence.
            rng (np.random.RandomState): Random number generator.
        """
        enc_data, tz, len_enc = x.data

        tot_cnt = len(enc_data)

        if not self._init_rng:
            self._rng_initialize(rng)
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

        icnts_buff = np.zeros_like(self.init_counts)
        scnts_buff = np.zeros_like(self.state_counts)
        tcnts_buff = np.zeros_like(self.trans_counts)
        wcnts_buff = np.zeros_like(self.wcnts)

        states = self._idx_rng.randint(low=0, high=self.num_states, size=tot_cnt)

        fast_seq_initialize(
            xs=enc_data,
            tz=tz,
            states=states,
            weights=weights,
            icnts=icnts_buff,
            scnts=scnts_buff,
            tcnts=tcnts_buff,
            wcnts=wcnts_buff,
        )

        # aggregate results
        self.wcnts += wcnts_buff
        self.trans_counts += tcnts_buff
        self.init_counts += icnts_buff
        self.state_counts += scnts_buff

    # TODO: Update from here down
    def seq_update(
        self,
        x: "IntegerHiddenMarkovEncodedDataSequence",
        weights: np.ndarray,
        estimate: IntegerHiddenMarkovModelDistribution,
    ) -> None:

        enc_data, tz, len_enc = x.data

        init_pvec = estimate.w
        tran_mat = estimate.transitions
        pmat = estimate.pmat

        init_counts = np.zeros_like(self.init_counts)
        trans_counts = np.zeros_like(self.trans_counts)
        emit_counts = np.zeros_like(self.wcnts)

        numba_baum_welch(
            xs=enc_data,
            tz=tz,
            weights=weights,
            init_pvec=init_pvec,
            tran_mat=tran_mat,
            pmat=pmat,
            init_counts=init_counts,
            tran_counts=trans_counts,
            emit_counts=emit_counts,
        )

        self.wcnts += emit_counts
        self.trans_counts += trans_counts
        self.init_counts += init_counts

        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(
        self,
        suff_stat: Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]
        ],
    ) -> "IntegerHiddenMarkovAccumulator":
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[T2]]): Sufficient statistics to combine.

        Returns:
            IntegerHiddenMarkovAccumulator: Self after combining.
        """

        init_counts, state_counts, trans_counts, wcnts, len_acc_value = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts
        self.wcnts += wcnts

        self.len_accumulator.combine(len_acc_value)

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]: Sufficient statistics.
        """
        len_val = self.len_accumulator.value()

        return (
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            self.wcnts,
            len_val,
        )

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[T2]]
    ) -> "IntegerHiddenMarkovAccumulator":
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[T2]]): Sufficient statistics.

        Returns:
            IntegerHiddenMarkovAccumulator: Self after setting values.
        """

        init_counts, state_counts, trans_counts, wcnts, len_acc = x

        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts
        self.wcnts = wcnts

        self.len_accumulator.from_value(len_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """

        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_counts
            else:
                stats_dict[self.init_key] = self.init_counts

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_counts
            else:
                stats_dict[self.trans_key] = self.trans_counts

        if self.state_key is not None:
            if self.state_key in stats_dict:
                stats_dict[self.state_key] += self.wcnts
            else:
                stats_dict[self.state_key] = self.wcnts

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

        return None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """

        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.wcnts = stats_dict[self.state_key]

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

        return None

    def acc_to_encoder(self) -> "IntegerHiddenMarkovDataEncoder":
        """Return a IntegerHiddenMarkovDataEncoder for this accumulator.

        Returns:
            IntegerHiddenMarkovDataEncoder: Encoder object.
        """
        len_encoder = self.len_accumulator.acc_to_encoder()

        return IntegerHiddenMarkovDataEncoder(len_encoder=len_encoder)


class IntegerHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating IntegerHiddenMarkovAccumulator objects.

    Attributes:
        num_words: Number of words / max val of vocabulary.
        num_states: Number of latent states.
        len_factory (StatisticAccumulatorFactory): Factory for the length distribution.
        keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states, transitions, and emissions.
        name (Optional[str]): Name for the object.
    """

    def __init__(
        self,
        num_words: int,
        num_states: int,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initializes HiddenMarkovAccumulatorFactory.

        Args:
            num_words: Number of words / max val of vocabulary.
            num_states: Number of latent states.
            factories: Factories for emission distributions.
            len_factory: Factory for the length distribution. Defaults to NullAccumulatorFactory().
            use_numba: Whether to use Numba for 'seq_' calls.
            keys: Keys for initial states, transitions, and emissions.
            name: Name for the object.
        """
        self.num_words = num_words
        self.num_states = num_states
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory
        self.name = name

    def make(self) -> "IntegerHiddenMarkovAccumulator":
        """Creates a new IntegerHiddenMarkovAccumulator.

        Returns:
            IntegerHiddenMarkovAccumulator: A new accumulator instance.
        """
        len_acc = self.len_factory.make()
        return IntegerHiddenMarkovAccumulator(
            num_words=self.num_words,
            num_states=self.num_states,
            len_accumulator=len_acc,
            keys=self.keys,
            name=self.name,
        )


class IntegerHiddenMarkovEstimator(ParameterEstimator):
    """Estimator for IntegerHiddenMarkovDistribution from aggregated sufficient statistics.

    Attributes:
        num_words (int): Size of vocabulary
        num_states (int): Number of latent states.
        estimators (List[ParameterEstimator]): Estimators for emission distributions.
        len_estimator (ParameterEstimator): Estimator for length distribution.
        pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo counts for initial states, transitions, and emissions.
        name (Optional[str]): Name for the object instance.
        keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states, transitions, and emissions.
    """

    def __init__(
        self,
        num_words: int,
        num_states: int,
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        pseudo_count: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = (None, None, None),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initializes IntegerHiddenMarkovEstimator.

        Args:
            num_words: Vocabular size.
            num_states: Number of latent states.
            len_estimator: Estimator for length distribution.
            pseudo_count: Pseudo counts for initial states and transitions.
            name: Name for the object instance.
            keys: Keys for initial states, transitions, and emissions.

        Raises:
            TypeError: If keys is not a tuple of three optional strings.
        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 3
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError(
                "IntegerHiddenMarkovEstimator requires keys (Tuple[Optional[str], Optional[str], Optional[str]])."
            )

        self.num_states = num_states
        self.num_words = num_words
        self.pseudo_count = (
            pseudo_count if pseudo_count is not None else (None, None, None)
        )
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.name = name

    def accumulator_factory(self) -> IntegerHiddenMarkovAccumulatorFactory:
        """Returns a factory for IntegerHiddenMarkovAccumulator.

        Returns:
            IntegerHiddenMarkovAccumulatorFactory: The accumulator factory.
        """
        len_factory = self.len_estimator.accumulator_factory()
        return IntegerHiddenMarkovAccumulatorFactory(
            num_words=self.num_words,
            num_states=self.num_states,
            len_factory=len_factory,
            keys=self.keys,
            name=self.name,
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[T2]],
    ) -> "IntegerHiddenMarkovModelDistribution":
        """Estimates a IntegerHiddenMarkovModelDistribution from sufficient statistics.

        Args:
            nobs: Number of observations.
            suff_stat: Sufficient statistics tuple.

        Returns:
            IntegerHiddenMarkovModelDistribution: The estimated distribution.
        """
        init_counts, state_counts, trans_counts, wcnts, len_ss = suff_stat

        num_states = len(init_counts)
        num_words = wcnts.shape[0]
        len_dist = self.len_estimator.estimate(nobs, len_ss)

        if self.pseudo_count[0] is not None:
            p1 = self.pseudo_count[0] / float(num_states)
            w = init_counts + p1
            w /= w.sum()
        else:
            w = init_counts / init_counts.sum()

        if self.pseudo_count[1] is not None:
            p2 = self.pseudo_count[1] / float(num_states * num_states)
            transitions = trans_counts + p2
            row_sum = transitions.sum(axis=1, keepdims=True)
            transitions /= row_sum
        else:
            row_sum = trans_counts.sum(axis=1, keepdims=True)
            bad_rows = row_sum.flatten() == 0.0

            if np.any(bad_rows):
                good_rows = ~bad_rows
                transitions = np.zeros_like(trans_counts, dtype=np.float64)
                transitions[good_rows, :] += (
                    trans_counts[good_rows, :] / row_sum[good_rows]
                )
            else:
                transitions = trans_counts / row_sum

        # Emissions are integer categoricals
        if self.pseudo_count[2] is not None:
            pseudo_count_per_level = self.pseudo_count[2] / num_words
            adjusted_state_nobs = (
                np.sum(wcnts, axis=0, keepdims=True) + self.pseudo_count[2]
            )
            pmat = (wcnts + pseudo_count_per_level) / adjusted_state_nobs
        else:
            pmat = wcnts / wcnts.sum(axis=0, keepdims=True)

        return IntegerHiddenMarkovModelDistribution(
            pmat=pmat,
            w=w,
            transitions=transitions,
            len_dist=len_dist,
            name=self.name,
            terminal_values=None,
        )


class IntegerHiddenMarkovDataEncoder(DataSequenceEncoder):
    """IntegerHiddenMarkovDataEncoder object for encoding sequences of iid HMM observations.

    Attributes:
        len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length of sequences.
            Should have support of non-negative integers. Set to NullDataEncoder if None.

    """

    def __init__(
        self, len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()
    ) -> None:
        """IntegerHiddenMarkovDataEncoder object.

        Attributes:
            len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length of sequences.
                Should have support of non-negative integers. Set to NullDataEncoder if None.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:

        return f"IntegerHiddenMarkovDataEncoder(len_encoder={str(self.len_encoder)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerHiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

    def seq_encode(
        self, x: Sequence[Sequence[int]]
    ) -> "IntegerHiddenMarkovEncodedDataSequence":
        """Sequence encode sequences of iid HMM observations.

        If use_numba is False, calls IntegerHiddenMarkovDataEncoder._seq_encode(x). (See '_seq_encode' for details).

        Args:
            x (Sequence[Sequence[int]]): A sequence of iid observations from an HMM distribution of type T.

        Returns:
            IntegerHiddenMarkovEncodedDataSequence: with numba_enc=True if use_numba=True.

        """

        xs = np.asarray([o for xs in x for o in xs], dtype=int)

        tz = np.zeros(len(x) + 1, dtype=int)
        tz[1:] = np.asarray([len(xs) for xs in x], dtype=int)

        len_enc = self.len_encoder.seq_encode(tz[1:])

        tz = np.cumsum(tz, dtype=int)

        return IntegerHiddenMarkovEncodedDataSequence(data=(xs, tz, len_enc))


class IntegerHiddenMarkovEncodedDataSequence(EncodedDataSequence):
    """IntegerHiddenMarkovEncodedDataSequence for vectorized calls.

    Notes:
        E = Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], EncodedDataSequence]

    Attributes:
        data (E): Encoded HMM sequences for numpy or numba 'seq_' calls.

    """

    def __init__(self, data: E) -> None:
        """IntegerHiddenMarkovEncodedDataSequence for vectorized calls.

        Notes:
            E = Tuple[Tuple[np.ndarray, np.ndarray, EncodedDataSequence]

        Args:
            data (E): Encoded HMM sequences for numpy or numba 'seq_' calls.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f"IntegerHiddenMarkovEncodedDataSequence(data={self.data})"


@numba.njit("float64(float64[:])")
def logsumexp_numba(arr):
    """Numerically stable logsumexp for 1D array."""
    max_val = np.max(arr)
    return max_val + np.log(np.sum(np.exp(arr - max_val)))


@numba.njit("void(float64[:,:], float64[:])")
def logsumexp_2d_rows(arr, result):
    """Logsumexp along axis=1 for 2D array."""
    for i in range(arr.shape[0]):
        result[i] = logsumexp_numba(arr[i])


@numba.njit(
    "void(int64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])"
)
def forward_backward(xs_seq, init_pvec, tran_mat, pmat, log_alpha, log_beta):
    """
    Forward-backward algorithm for a single sequence.

    Args:
        xs_seq: numpy array of shape (T,) containing emission observations
        init_pvec: numpy array of shape (S,) initial state distribution
        tran_mat: numpy array of shape (S, S) transition matrix
        pmat: numpy array of shape (W, S) emission matrix P(W=i | Z=z)
        log_alpha: pre-allocated output array (T, S) for log forward probabilities
        log_beta: pre-allocated output array (T, S) for log backward probabilities
    """
    T = len(xs_seq)
    S = len(init_pvec)

    # Forward pass (in log space for numerical stability)

    # Initialize: alpha_1(z) = pi(z) * P(x_1 | z)
    for j in range(S):
        log_alpha[0, j] = np.log(init_pvec[j]) + np.log(pmat[xs_seq[0], j])

    # Recursion: alpha_t(z) = sum_z' alpha_{t-1}(z') * A(z'->z) * P(x_t | z)
    temp = np.empty(S)
    for t in range(1, T):
        for j in range(S):
            for i in range(S):
                temp[i] = log_alpha[t - 1, i] + np.log(tran_mat[i, j])
            log_alpha[t, j] = logsumexp_numba(temp) + np.log(pmat[xs_seq[t], j])

    # Backward pass

    # Initialize: beta_T(z) = 1 for all z (log(1) = 0)
    for j in range(S):
        log_beta[T - 1, j] = 0.0

    # Recursion: beta_t(z) = sum_z' A(z->z') * P(x_{t+1} | z') * beta_{t+1}(z')
    for t in range(T - 2, -1, -1):
        for i in range(S):
            for j in range(S):
                temp[j] = (
                    np.log(tran_mat[i, j])
                    + np.log(pmat[xs_seq[t + 1], j])
                    + log_beta[t + 1, j]
                )
            log_beta[t, i] = logsumexp_numba(temp)


@numba.njit(
    "void(int64[:], int64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:])",
    parallel=True,
    fastmath=True,
)
def numba_baum_welch(
    xs, tz, weights, init_pvec, tran_mat, pmat, init_counts, tran_counts, emit_counts
):
    """
    E-step: Compute sufficient statistics for Baum-Welch algorithm.
    Uses parallel processing across sequences.

    Args:
        xs: numpy array int32 of emission observations
        tz: numpy array int64 containing start/end indices (len(tz)-1 sequences)
        init_pvec: numpy array float64 of shape (S,) initial state distribution
        tran_mat: numpy array float64 of shape (S, S) transition matrix
        pmat: numpy array float64 of shape (W, S) emission matrix
        init_counts: output array float64 of shape (S,) for initial state counts
        tran_counts: output array float64 of shape (S, S) for transition counts
        emit_counts: output array float64 of shape (W, S) for emission counts
    """
    S = len(init_pvec)
    W = pmat.shape[0]
    num_sequences = len(tz) - 1

    # Zero out the output arrays
    for j in range(S):
        init_counts[j] = 0.0

    for i in range(S):
        for j in range(S):
            tran_counts[i, j] = 0.0

    for i in range(W):
        for j in range(S):
            emit_counts[i, j] = 0.0

    # Create a lock for thread-safe updates
    lock = numba.objmode(None, "none")()

    # Process each sequence in parallel
    for seq_idx in numba.prange(num_sequences):
        start = tz[seq_idx]
        end = tz[seq_idx + 1]
        T = end - start

        # Extract sequence view
        xs_seq = xs[start:end]
        weight_loc = weights[seq_idx]

        # Local accumulators for this sequence
        local_init = np.zeros(S)
        local_tran = np.zeros((S, S))
        local_emit = np.zeros((W, S))

        # Allocate arrays for forward-backward
        log_alpha = np.zeros((T, S))
        log_beta = np.zeros((T, S))

        # Run forward-backward
        forward_backward(xs_seq, init_pvec, tran_mat, pmat, log_alpha, log_beta)

        # Compute gamma: P(Z_t = z | X, theta)
        log_gamma = log_alpha + log_beta

        # Normalize each row
        log_normalizers = np.empty(T)
        logsumexp_2d_rows(log_gamma, log_normalizers)

        for t in range(T):
            for j in range(S):
                log_gamma[t, j] -= log_normalizers[t]

        gamma = np.exp(log_gamma)

        # Compute xi and accumulate transition counts
        log_xi = np.empty((S, S))
        log_xi_flat = np.empty(S * S)

        for t in range(T - 1):
            # Compute log_xi
            for i in range(S):
                for j in range(S):
                    log_xi[i, j] = (
                        log_alpha[t, i]
                        + np.log(tran_mat[i, j])
                        + np.log(pmat[xs_seq[t + 1], j])
                        + log_beta[t + 1, j]
                    )

            # Flatten for normalization
            for i in range(S):
                for j in range(S):
                    log_xi_flat[i * S + j] = log_xi[i, j]

            log_norm = logsumexp_numba(log_xi_flat)

            # Accumulate transition counts
            for i in range(S):
                for j in range(S):
                    local_tran[i, j] += np.exp(log_xi[i, j] - log_norm) * weight_loc

        # Accumulate initial state counts (from first position)
        for j in range(S):
            local_init[j] = gamma[0, j] * weight_loc

        # Accumulate emission counts
        for t in range(T):
            obs = xs_seq[t]
            for j in range(S):
                local_emit[obs, j] += gamma[t, j] * weight_loc

        # Add local counts to global using a critical section for thread-safety
        with numba.objmode():
            # Critical section - only one thread can execute this at a time
            for j in range(S):
                init_counts[j] += local_init[j]

            for i in range(S):
                for j in range(S):
                    tran_counts[i, j] += local_tran[i, j]

            for i in range(W):
                for j in range(S):
                    emit_counts[i, j] += local_emit[i, j]


@numba.njit(
    "void(int64, int64[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], "
    "float64[:,:])",
    parallel=True,
    fastmath=True,
)
def numba_baum_welch2(
    num_states: int,
    tz: np.ndarray,
    prob_mat: np.ndarray,
    init_pvec: np.ndarray,
    tran_mat: np.ndarray,
    weights: np.ndarray,
    alpha_loc: np.ndarray,
    xi_acc: np.ndarray,
    pi_acc: np.ndarray,
) -> None:
    """Parallelized Baum-Welch forward-backward algorithm for HMM parameter estimation.

    Args:
        num_states (int): Number of hidden states.
        tz (np.ndarray): Cumulative sum of sequence lengths.
        prob_mat (np.ndarray): Observation likelihoods for each state.
        init_pvec (np.ndarray): Initial state probabilities.
        tran_mat (np.ndarray): State transition matrix.
        weights (np.ndarray): Sequence weights.
        alpha_loc (np.ndarray): Forward probabilities.
        xi_acc (np.ndarray): Accumulator for transition probabilities.
        pi_acc (np.ndarray): Accumulator for initial state probabilities.
    """
    for n in numba.prange(len(tz) - 1):

        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):

            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):

            sp1 = s + 1

            for j in range(num_states):
                beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

            xi_buff_sum = 0
            gamma_buff = 0
            beta_sum = 0
            for i in range(num_states):

                temp_beta = 0
                for j in range(num_states):
                    temp = tran_mat[i, j] * beta_buff[j]
                    temp_beta += temp
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_loc[s0, i]


@numba.njit(
    "void(int64[:], int64[:], float64[:], int64[:], float64[:], float64[:], float64[:,:], float64[:,:])",
    fastmath=True,
)
def fast_seq_initialize(
    xs: np.ndarray,
    tz: np.ndarray,
    weights: np.ndarray,
    states: np.ndarray,
    icnts: np.ndarray,
    scnts: np.ndarray,
    tcnts: np.ndarray,
    wcnts: np.ndarray,
) -> None:
    """Process data sequentially to ensure deterministic results.

    Args:
        xs: Flattened array of observations
        tz: Cumulative indices marking sequence boundaries
        weights: Weights for each sequence
        states: States for each observation
        icnts: Initial state counts buffer with shape (num_states)
        scnts: State counts buffer with shape (num_states)
        tcnts: Transition counts buffer with shape (num_states, num_states)
        wcnts: Word counts buffer with shape (num_words, num_states)
    """
    num_words, num_states = wcnts.shape
    num_sequences = len(tz) - 1

    # Process data sequentially to ensure deterministic results
    for n in range(num_sequences):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        local_tran = np.zeros((num_states, num_states))
        local_init = np.zeros(num_states)
        local_wcnts = np.zeros((num_words, num_states))

        for i in range(s0, s1):
            # Update buffers for this sequence directly
            local_wcnts[xs[i], states[i]] += weights[n]

            if i == s0:
                local_init[states[i]] += weights[n]
            else:
                local_tran[states[i - 1], states[i]] += weights[n]

        # Add local counts to global accumulation
        for j in range(num_states):
            icnts[j] += local_init[j]

        for i in range(num_states):
            for j in range(num_states):
                tcnts[i, j] += local_tran[i, j]

        for i in range(num_words):
            for j in range(num_states):
                wcnts[i, j] += local_wcnts[i, j]


@numba.njit(
    "void(int64[:], int64[:], float64[:,:], float64[:], float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def fast_log_density(
    xs: np.ndarray,
    tz: np.ndarray,
    log_pmat: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Compute log density of sequences for an HMM with categorical emissions.

    Parameters:
    -----------
    xs : flattened array of observations
    tz : cumulative indices marking sequence boundaries
    log_pmat : array (V, S) - log emission probabilities
    log_pi : array (S,) - log initial state probabilities
    log_A : array (S, S) - log transition probabilities
    out : array (n,) - output log densities
    """
    num_states = log_pi.shape[0]

    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]
        sz = s1 - s0

        if sz == 0:
            out[n] = 0.0
            continue

        # Allocate forward variables (reuse across time steps)
        log_alpha_prev = np.empty(num_states)
        log_alpha_curr = np.empty(num_states)

        # Initial step
        x0 = xs[s0]
        for s in range(num_states):
            log_alpha_prev[s] = log_pi[s] + log_pmat[x0, s]

        # Forward recursion
        for t in range(1, sz):
            xt = xs[s0 + t]

            for s in range(num_states):
                # Compute log-sum-exp more efficiently
                max_val = log_alpha_prev[0] + log_A[0, s]
                for s_prev in range(1, num_states):
                    val = log_alpha_prev[s_prev] + log_A[s_prev, s]
                    if val > max_val:
                        max_val = val

                sum_exp = 0.0
                for s_prev in range(num_states):
                    sum_exp += np.exp(
                        log_alpha_prev[s_prev] + log_A[s_prev, s] - max_val
                    )

                log_alpha_curr[s] = max_val + np.log(sum_exp) + log_pmat[xt, s]

            # Swap buffers
            log_alpha_prev, log_alpha_curr = log_alpha_curr, log_alpha_prev

        # Final probability (log-sum-exp over final states)
        max_val = log_alpha_prev[0]
        for s in range(1, num_states):
            if log_alpha_prev[s] > max_val:
                max_val = log_alpha_prev[s]

        sum_exp = 0.0
        for s in range(num_states):
            sum_exp += np.exp(log_alpha_prev[s] - max_val)

        out[n] = max_val + np.log(sum_exp)
