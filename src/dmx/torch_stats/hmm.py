""""Create, estimate, and sample from a hidden markov model with K emission distributions (i.e. K states).

Defines the HierarchicalMixtureDistribution, HierarchicalMixtureSampler, HierarchicalMixtureEstimatorAccumulatorFactory,
HierarchicalMixtureEstimatorAccumulator, HierarchicalMixtureEstimator, and the HierarchicalMixtureDataEncoder classes
for use with pysparkplug.

Data type: Sequence[T] (determined by emission distributions).

Consider an observation x = (x_1, x_2, ..., x_T) where x_i is of data type T. Assume Z = (Z_1, ..., Z_T) is an
unobserved sequence of hidden states taking on values {1,2,..,K}. A K state hidden markov model can be written as
hierarchical model as follows:

For t = 1,2,..,T, the emission distributions are given by
    (1) P_1(X_t = x_t | Z_t = k), for k = {1,2,...,K}.

The state transitions are given by the K by K matrix formed from
    (2) p_mat(Z_t = i | Z_{t-1} = j), for i, j = {2,3,..,K}.

The initial state distribution is given by weights
    (3) p_mat(Z_1=k) = pi_k, for k = {1,2,...,K}, where sum_k pi_k = 1.0

If included, the length of the hidden markov model sequences is modeled through
    (4) P_len(T), where P_len() is a distribution with support on non-negative integers.

Note that P_1() in (1) must be a distribution compatible with type T data. p_mat() in (2) is a 2-d numpy array of 2-d
list of floats where the rows sum to 1.0. (3) is represented by a numpy array of list of floats that sum to 1.

"""

import torch as tn
import numpy as np
from numpy.random import RandomState
import dmx.torch_utils.vector as vec
from dmx.utils.arithmetic import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence
from dmx.stats.markovchain import MarkovChainDistribution
from dmx.torch_stats.mixture import MixtureDistribution
from dmx.torch_stats.null_dist import NullDistribution, NullAccumulatorFactory, NullEstimator, NullDataEncoder, \
    NullAccumulator

from dmx.utils.arithmetic import maxrandint

from typing import List, Any, Tuple, Sequence, Union, Optional, TypeVar, Set, Dict

T = TypeVar('T')
T1 = TypeVar('T1')  # Emission suff-stat type
T2 = TypeVar('T2')  # Len suff-stat type
E = Tuple[Tuple[int, int, tn.tensor, List[tn.tensor], tn.tensor, tn.tensor, tn.tensor, TorchEncodedSequence], TorchEncodedSequence]


class HiddenMarkovModelDistribution(TorchProbabilityDistribution):

    def __init__(self, topics: Sequence[TorchProbabilityDistribution],
                 w: Union[Sequence[float], np.ndarray],
                 transitions: Union[List[List[float]], np.ndarray],
                 taus: Optional[Union[List[List[float]], np.ndarray]] = None,
                 len_dist: Optional[TorchProbabilityDistribution] = None,
                 terminal_values: Optional[Set[T]] = None,
                 device: Optional[tn.device] = None) -> None:
        """HiddenMarkovModelDistribution object defining HMM compatible with data type T.

        Defines an HMM with emission distributions in 'topics' (all must have the same data type T). If a length
        distribution for the length of HMM sequence is included, it must have data type int with support of non-negative
        integers.


        Args:
            topics (Sequence[TorchProbabilityDistribution]): Emission distributions all having type T.
            w (Union[Sequence[float], np.ndarray]): Initial state probabilities.
            transitions (Union[List[List[float]], np.ndarray]): 2-d array of hidden state transition probabilities.
            taus (Optional[Union[Sequence[float], np.ndarray]]): Emission distributions are a Mixture over topics.
                Hidden states govern transitions between mixture weights.
            len_dist (Optional[TorchProbabilityDistribution]):
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
            device (Optional[tn.device]): Device for tensor calculations.

        Attributes:
            topics (Sequence[TorchProbabilityDistribution]): Emission distributions all having type T.
            n_topics (int): Number of emission distributions.
            n_states (int): Number of hidden states.
            w (Tensor): Initial state probabilities.
            log_w (Tensor): Initial state log-probabilities.
            transitions (Tensor): 2-d tensor of hidden state transition probabilities. (n_states by n_states).
            log_transitions (Tensor): Log of above.
            taus (Optional[Tensor]): Emission distributions are a Mixture over topics. Hidden states govern
                transitions between mixture weights.
            log_taus (Optional[tn.Tensor]): Log probabilties of taus above.
            has_topics (bool): True if taus is passed.
            len_dist (Optional[TorchProbabilityDistribution]):
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.

        """
        super().__init__(device)
        self.topics = topics
        self.n_topics = len(topics)
        self.n_states = len(w)
        self.w = vec.tensor(w, device=self._device)
        self.log_w = tn.log(self.w)

        if not isinstance(transitions, np.ndarray):
            transitions = np.asarray(transitions, dtype=float)

        self.transitions = np.reshape(transitions, (self.n_states, self.n_states))
        self.transitions = tn.tensor(transitions, device=device)
        self.log_transitions = tn.log(self.transitions)
        self.terminal_values = terminal_values

        self.len_dist = len_dist if len_dist is not None else NullDistribution(device=self._device)

        if taus is not None:
            self.taus = vec.tensor(taus, device=self._device)
            self.log_taus = tn.log(self.taus)
            self.has_topics = True
        else:
            self.taus = None
            self.has_topics = False

    def to(self, device: tn.device) -> None:
        for dist in self.topics:
            dist.to(device)

        self.w = self.w.to(device)
        self.transitions.to(device)

        self.log_w = tn.log(self.w)
        self.log_transitions = tn.log(self.transitions)

        if self.taus is not None:
            self.taus = self.taus.to(device)
            self.log_taus = tn.log(self.taus)

        self._device = device

    def __repr__(self) -> str:
        """Returns string representation of HiddenMarkovDistribution instance."""
        s1 = ','.join(map(str, self.topics))
        s2 = repr(self.w.data.cpu().tolist())
        s3 = repr([u for u in self.transitions.data.cpu().tolist()])
        if self.taus is None:
            s4 = repr(None)
        else:
            s4 = repr([u for u in self.taus.data.cpu().tolist()])
        s5 = str(self.len_dist)
        s6 = repr(self.terminal_values)

        return 'HiddenMarkovModelDistribution([%s], %s, %s, %s, len_dist=%s, terminal_values=%s)' % (s1, s2, s3, s4, s5, s6)

    def density(self, x: List[T]) -> float:
        """Returns the density of HMM for an observed sequence x.

        See 'HiddenMarkovDistribution.log_density()' for details.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Density of HMM for observed sequence x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: List[T]) -> float:
        """Returns the log-density of HMM for observed sequence x.

        Density for a sequence of length N is given by recursively evaluating the conditional density,

            p_mat(x_mat(0),x_mat(1),....,x_mat(t)) = p_mat(x_mat(t)|x_mat(0),...,x_mat(t-1)) = p_mat(x_mat(t)|Z(t))*p_mat(Z(t)|Z(t-1))*p_mat(Z(t-1)|x_mat(0),....,x_mat(t-1))

        for t = 1,2,...,N-1. p_mat(Z(0)) is given by 'w', p_mat(x_mat(t)|Z(t)) is given by emission distribution 'topics' for
        t = 0,1,...,N-1.

        The returned density is given by

            p_mat(x_mat) = p_mat(x_mat(0),x_mat(1),....,x_mat(t))*P_len(N).

        where P_len(N) is the length distribution 'len_dist', if assigned.
        Note: All calculations are done on the log scale with log-sum-exp used to prevent numerical underflow.

        If 'has_topics' is true, 'weighed_log_sum_exp' and 'log_sum' calls from dmx.utils.vector are used to handle
        the emission distributions being treated as mixture distributions with weights 'log_taus'.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Log-density of observed HMM sequence x.

        """
        if x is None or len(x) == 0:
            return self.len_dist.log_density(0)  # this will return 0.0 if NullDistribution()

        enc_data = self.dist_to_encoder().seq_encode([x], device=self._device)

        return float(self.seq_log_density(enc_data)[0])

    def seq_log_density(self, x: 'HiddenMarkovTorchSequence') -> tn.Tensor:

        num_states = self.n_states

        if not isinstance(x, HiddenMarkovTorchSequence):
            raise Exception('HiddenMarkovTorchSequence required for `seq_` function calls.')

        if x.device != self.model_device():
            raise Exception('HiddenMarkovTorchSequence must be on same device as model.')

        (tot_cnt, max_len, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x.data
        w = self.w
        a_mat = self.transitions

        num_seq = int(idx_mat.shape[0])

        good = idx_mat >= 0

        pr_obs = vec.zeros((tot_cnt, num_states), device=self._device)
        ll_ret = vec.zeros(num_seq, device=self._device)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

        pr_max0, _ = pr_obs.max(dim=1, keepdim=True)
        pr_obs -= pr_max0
        tn.exp(pr_obs, out=pr_obs)

        # Vectorized alpha pass
        band0 = idx_bands[:, 0]
        band1 = idx_bands[:, 1]

        alphas_prev = tn.multiply(pr_obs[band0[0]:band1[0], :], w)
        temp = alphas_prev.sum(dim=1, keepdim=True)
        # temp2 = temp.copy()
        # temp2[temp2 == 0] = 1.0
        alphas_prev /= temp

        tn.log(temp, out=temp)
        temp2 = pr_max0[band0[0]:band1[0], 0]
        ll_ret[good[:, 0]] += temp[:, 0] + temp2

        for i in range(1, max_len):
            band = idx_bands[i]
            has_next_loc = has_next[i - 1]

            alphas_next = tn.matmul(alphas_prev[has_next_loc, :], a_mat)
            alphas_next *= pr_obs[band[0]:band[1], :]
            pr_max = alphas_next.sum(dim=1, keepdim=True)
            # pr_max2 = pr_max.copy()
            # pr_max2[pr_max2 == 0] = 1.0
            alphas_next /= pr_max
            alphas_prev = alphas_next

            tn.log(pr_max, out=pr_max)
            temp2 = pr_max0[band0[i]:band1[i], 0]
            ll_ret[good[:, i]] += pr_max[:, 0] + temp2

        # nz = len_vec != 0
        # ll_ret[nz] /= len_vec[nz]

        ll_ret[tn.isnan(ll_ret)] = -tn.inf

        if self.len_dist is not None:
            ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def viterbi(self, x: List[T]) -> tn.Tensor:
        nn = len(x)
        num_states = self.n_states

        v = tn.zeros((nn, num_states), device=self._device)
        ptr = tn.zeros(nn, device=self._device)
        pr_obs = tn.zeros((nn, num_states), device=self._device)
        enc_x = self.topics[0].dist_to_encoder().seq_encode(x, device=self._device)

        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

        v[0, :] += pr_obs[0, :] + self.log_w

        for t in range(1, nn):
            temp = tn.zeros((num_states, num_states), device=self._device)
            temp += tn.reshape(v[t-1, :], (num_states, 1))
            temp += self.log_transitions
            temp += tn.reshape(pr_obs[t, :], (1, num_states))
            temp, _ = temp.max(dim=0, keepdim=False)
            v[t, :] += temp

        for t in range(nn-1, -1, -1):
            ptr[t] = tn.argmax(v[t, :])

        return ptr

    def sampler(self, seed: Optional[int] = None) -> 'HiddenMarkovSampler':
        """Create a HiddenMarkovSampler object with seed passed.

        Note: Throws exception if 'len_dist'and 'terminal_values' are not set.

        If len_dist is set, it should be a SequenceEncodableProbabilityDistribution with data type int and support on
        non-negative integers.

        Args:
            seed (Optional[int]): Set seed for random sampling.

        Returns:
            HiddenMarkovSampler object.

        """
        if isinstance(self.len_dist, NullDistribution) and self.terminal_values is None:
            raise Exception('HiddenMarkovSampler requires len_dist with support on non-negative integers, or terminal_'
                            'values to be set.')

        return HiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HiddenMarkovEstimator':
        """Create HiddenMarkovEstimator for estimating HiddenMarkovDistribution objects from aggregated sufficient
            statistics.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics of HiddenMarkovDistribution object
                instance.

        Returns:
            HiddenMarkovEstimator object.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return HiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count, pseudo_count), len_estimator=len_est)

    def dist_to_encoder(self) -> 'HiddenMarkovDataEncoder':
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return HiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder)


class HiddenMarkovSampler(DistributionSampler):

    def __init__(self, dist: 'HiddenMarkovModelDistribution', seed: Optional[int] = None) -> None:
        """HiddenMarkovSampler object for sampling from HMM.

        If 'dist.len_dist' is set, samples HMM sequences with sequence lengths generated from 'len_dist'. If
        'dist.len_dist' is NullDistribution, 'dist.terminal_values' is must be set. Samples are generated until
        a terminal value is reached.

        Args:
            dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
            seed (Optional[int]): Set seed on random number generator for sampling.

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
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.has_topics:
            taus = dist.taus.data.cpu().numpy()
            self.obs_samplers = [
                MixtureDistribution(dist.topics, taus[i, :]).sampler(seed=self.rng.randint(0, maxrandint)) for i in
                range(dist.n_states)]
        else:
            self.obs_samplers = [dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in
                                 range(dist.n_states)]

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        else:
            self.len_sampler = None

        if dist.terminal_values is None:
            self.terminal_set = None
        else:
            self.terminal_set = set(dist.terminal_values)

        trans = dist.transitions.data.cpu().numpy()
        w = dist.w.data.cpu().numpy()

        t_map = {i: {k: trans[i, k] for k in range(dist.n_states)} for i in range(dist.n_states)}
        p_map = {i: w[i] for i in range(dist.n_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample_seq(self, size: Optional[int] = None) -> Union[List[Any], List[List[Any]]]:
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
            obs_seq = [self.obs_samplers[state_seq[i]].sample() for i in range(n)]

            return obs_seq

        else:
            n = self.len_sampler.sample(size=size)
            state_seq = [self.state_sampler.sample_seq(size=nn) for nn in n]
            obs_seq = [[self.obs_samplers[j].sample() for j in nn] for nn in state_seq]

            return obs_seq

    def sample_terminal(self, terminal_set: Set[T]) -> List[T]:
        """Sample an HMM sequence, until a terminal value is samples from the emission distribution.

        Args:
            terminal_set (Set[T]): Set values to terminate the HMM sequence.

        Returns:
            List[T] with length determined by samples to reach the first terminating value.

        """
        z = self.state_sampler.sample_seq()
        rv = [self.obs_samplers[z].sample()]

        while rv[-1] not in terminal_set:
            z = self.state_sampler.sample_seq(v0=z)
            rv.append(self.obs_samplers[z].sample())

        return rv

    def sample(self, size: Optional[int] = None):
        """Draw iid samples from HMM.

        If a 'len_sampler' is set, call 'sample_seq()' (See HiddenMarkovSampler.sample_seq() for details).
        If 'len_sampler' is the NullDistributionSampler(), 'sample_terminal()' is called. (See
        HiddenMarkovSampler.sample_terminal() for details).

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.

        Returns:
            List[T] or List[List[T]] depending on arg size.

        """
        if self.len_sampler is not None:
            return self.sample_seq(size=size)

        elif self.terminal_set is not None:
            if size is None:
                return self.sample_terminal(self.terminal_set)
            else:
                return [self.sample_terminal(self.terminal_set) for i in range(size)]

        else:
            raise RuntimeError('HiddenMarkovSampler requires either a length distribution or terminal value set.')


class HiddenMarkovAccumulator(TorchStatisticAccumulator):

    def __init__(self, accumulators: Sequence[TorchStatisticAccumulator],
                 len_accumulator: Optional[TorchStatisticAccumulator] = None,
                 keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
                 device: Optional[tn.device] = None) -> None:
        super().__init__(device)
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator(device)

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

    def seq_initialize(self, x: 'HiddenMarkovTorchSequence', weights: tn.Tensor, tng: tn.Generator) -> None:
        (tot_cnt, max_len, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x.data

        self.len_accumulator.seq_initialize(len_enc, weights, tng)

        non_zero_len = len_vec != 0
        weights_nz = weights[non_zero_len]
        weights_loc = weights_nz[idx_vec]

        band0 = idx_bands[:, 0]
        band1 = idx_bands[:, 1]

        idx = vec.randint(size=tot_cnt, low=0, high=self.num_states, tng=tng)

        # count the states
        self.state_counts += tn.bincount(idx, weights=weights_loc, minlength=self.num_states).cpu().detach().numpy()

        # count initial states
        b0, b1 = band0[0], band1[0]
        tmp = tn.bincount(idx[b0:b1], weights=weights_loc[b0:b1], minlength=self.num_states)
        tmp = tmp.cpu().detach().numpy()
        self.init_counts += tmp

        # Vectorized alpha pass
        idx_prev = idx[b0:b1]
        w_prev = weights_loc[b0:b1]

        tcnts = vec.zeros(self.num_states**2, device=self._device)

        for i in range(1, max_len):
            b0, b1 = band0[i], band1[i]
            has_next_loc = has_next[i - 1]
            idx_next = idx[b0:b1]

            idx0 = idx_prev[has_next_loc]*self.num_states + idx_next
            w_prev = w_prev[has_next_loc]
            tcnts += tn.bincount(idx0, weights=w_prev, minlength=self.num_states**2)

            idx_prev = idx_next

        for j in range(self.num_states):
            self.accumulators[j].seq_initialize(enc_data, tn.where(idx == j, weights_loc, 0.0), tng)

        self.trans_counts += tcnts.reshape(self.num_states, self.num_states).cpu().detach().numpy()

    def seq_update(self, x: 'HiddenMarkovTorchSequence', weights: tn.Tensor, estimate: HiddenMarkovModelDistribution) -> None:

        num_states = self.num_states
        (tot_cnt, max_len, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x.data
        w = estimate.w
        a_mat = estimate.transitions

        band0 = idx_bands[:, 0]
        band1 = idx_bands[:, 1]

        good = idx_mat >= 0

        pr_obs = vec.zeros((tot_cnt, num_states), device=self._device)
        alphas = vec.zeros((tot_cnt, num_states), device=self._device)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

        pr_max, _ = pr_obs.max(dim=1, keepdim=True)
        pr_obs -= pr_max
        tn.exp(pr_obs, out=pr_obs)

        # Vectorized alpha pass
        alphas_prev = alphas[band0[0]:band1[0], :]
        tn.multiply(pr_obs[band0[0]:band1[0], :], w, out=alphas_prev)

        # tn.multiply(pr_obs[band0[0]:band0[0], :], w, out=alphas_prev)
        pr_sum = alphas_prev.sum(dim=1, keepdim=True)
        pr_sum[pr_sum == 0.0] = 1.0
        alphas_prev /= pr_sum

        for i in range(1, max_len):
            has_next_loc = has_next[i - 1]
            alphas_next = alphas[band0[i]:band1[i], :]
            tn.matmul(alphas_prev[has_next_loc, :], a_mat, out=alphas_next)
            alphas_next *= pr_obs[band0[i]:band1[i], :]

            pr_max = alphas_next.sum(dim=1, keepdim=True)
            pr_max[pr_max == 0.0] = 1.0

            alphas_next /= pr_max
            alphas_prev = alphas_next

        prev_beta = vec.ones((band1[-1] - band0[-1], num_states), device=self._device)
        alphas[band0[-1]:band1[-1], :] /= alphas[band0[-1]:band1[-1], :].sum(dim=1, keepdim=True)

        tcnts = vec.zeros((self.num_states, self.num_states), device=self._device)

        # Vectorized beta pass
        for i in range(max_len - 2, -1, -1):
            # band1 = idx_bands[i]
            # band2 = idx_bands[i + 1]
            has_next_loc = has_next[i]

            next_b = pr_obs[band0[i+1]:band1[i+1], :]
            prev_a = alphas[band0[i]:band1[i], :]
            prev_a = prev_a[has_next_loc, :]

            prev_beta *= next_b

            prev_a = prev_a.reshape((prev_a.shape[0], prev_a.shape[1], 1))
            next_beta2 = prev_beta.reshape((prev_beta.shape[0], 1, prev_beta.shape[1]))

            xi_loc = next_beta2 * a_mat
            next_beta = xi_loc.sum(dim=2)
            next_beta_max, _ = next_beta.max(dim=1, keepdim=True)
            next_beta_max[next_beta_max == 0.0] = 1.0
            next_beta /= next_beta_max

            prev_beta = vec.ones((int(band1[i] - band0[i]), num_states), device=self._device)
            prev_beta[has_next_loc, :] = next_beta

            xi_loc *= prev_a
            xi_loc_sum = xi_loc.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)

            weights_loc = tn.reshape(weights[good[:, i + 1]], (-1, 1, 1))
            xi_loc_sum[xi_loc_sum == 0] = 1.0

            xi_loc *= weights_loc / xi_loc_sum

            temp = xi_loc.sum(dim=2)
            temp_sum = temp.sum(dim=1, keepdim=True)
            temp_sum[temp_sum == 0] = 1.0
            temp /= temp_sum

            alphas[band0[i] + has_next_loc, :] = temp

            tcnts += xi_loc.sum(dim=0)

        self.trans_counts += tcnts.cpu().detach().numpy()
        # Aggregate sufficient statistics
        for i in range(num_states):
            alphas[:, i] *= weights[idx_vec]
            self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

        self.state_counts += alphas.sum(dim=0).cpu().detach().numpy()

        temp = alphas[band0[0]:band1[0], :].sum(dim=1, keepdim=True)
        temp[temp == 0] = 1.0
        alphas[band0[0]:band1[0], :] *= tn.reshape(weights[good[:, 0]], (-1, 1)) / temp

        self.init_counts += alphas[band0[0]:band1[0], :].sum(dim=0).cpu().detach().numpy()

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]]) \
            -> 'HiddenMarkovAccumulator':
        """Combine the sufficient statistics of HiddenMarkovAccumulator with suff_stat arg.

        Sufficient statistics in suff_stat are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (Sequence[T1]): Emission distribution accumulators.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            suff_stat: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, acc_values, len_acc_value = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.accumulators[i].combine(acc_values[i])

        if len_acc_value is not None:
            self.len_accumulator.combine(len_acc_value)

        return self

    def value(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any],
                             Optional[Any]]:
        """Returns sufficient statistics of HiddenMarkovAccumulator object instance.

        Returned value rv is a Tuple containing:
            rv[0] (int): Number of hidden states.
            rv[1] (np.ndarray): Initial state counts.
            rv[2] (np.ndarray): State counts.
            rv[3] (np.ndarray): State transition counts.
            rv[4] (Sequence[T1]): Emission distribution accumulator sufficient statistics (type T1).
            rv[5] (Optional[T2]): Optional sufficient statistics of the length distribution (type T2).

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]].

        """
        len_val = self.len_accumulator.value()

        return self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple(
            [u.value() for u in self.accumulators]), len_val

    def from_value(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]])\
            -> 'HiddenMarkovAccumulator':
        """Set the sufficient statistics of HiddenMarkovAccumulator object instance to value x.

        Returned value x is a Tuple containing:
            x[0] (int): Number of hidden states.
            x[1] (np.ndarray): Initial state counts.
            x[2] (np.ndarray): State counts.
            x[3] (np.ndarayy): State transition counts.
            x[4] (List[T1]): Emission distribution accumulators.
            x[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            x: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = x
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(accumulators):
            self.accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with sufficient statistics in suff_stat that have
            matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary containing sufficient statistics for corresponding keys.

        Returns:
            None.

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
                acc = stats_dict[self.state_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

        return None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the sufficient statistics of HiddenMarkovAccumulator object with matching sufficient statistics in
            arg suff_stat that have matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.accumulators = stats_dict[self.state_key]

        for u in self.accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

        return None

    def acc_to_encoder(self) -> 'HiddenMarkovDataEncoder':
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return HiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder)


class HiddenMarkovAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, factories: Sequence[TorchStatisticAccumulatorFactory],
                 len_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        """HiddenMarkovAccumulatorFactory object for creating HiddenMarkovEstimatorAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution.
            keys (Optional[Tuple[Optional[str],Optional[str], Optional[str]]]): Set keys for initial states, state
                transitions, and the emission distributions.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution. Defaults
                to NullAccumulatorFactory().
            keys (Tuple[Optional[str],Optional[str], Optional[str]]): Set keys for initial states, state
                transitions, and the emission distributions.


        """
        self.factories = factories
        self.keys = keys if keys is None else (None, None, None)
        self.len_factory = len_factory

    def make(self, device: Optional[tn.device] = None) -> 'HiddenMarkovAccumulator':
        """Returns a HiddenMarkovAccumulator object. """
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return HiddenMarkovAccumulator([self.factories[i].make() for i in range(len(self.factories))],
                                       len_accumulator=len_acc, keys=self.keys, device=device)


class HiddenMarkovEstimator(TorchParameterEstimator):

    def __init__(self, estimators: List[TorchParameterEstimator],
                 len_estimator: Optional[TorchParameterEstimator] = None,
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        """HiddenMarkovEstimator object for estimating HiddenMarkovDistribution for aggregated sufficient statistics.

        Args:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for length distribution.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Pseudo count for initial states and
                state transitions.
            name (Optional[str]): Set name to object.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for initial states,
                transitions counts, and emission distributions.

        Attributes:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (ParameterEstimator): ParameterEstimator object for length distribution, set to NullEstimator
                if None was passed.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo count for initial states and
                state transitions. Defaults to Tuple of (None, None) if None was passed.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states, transitions counts, and
                emission distributions. Defaults to Tuple of (None, None, None).

        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()

    def accumulator_factory(self):
        """Returns an HiddenMarkovAccumulatorFactory object."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HiddenMarkovAccumulatorFactory(est_factories, len_factory)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[T1], Optional[T2]],
                 device: Optional[tn.device] = None)\
            -> 'HiddenMarkovModelDistribution':
        """Estimate HiddenMarkovModel from aggregated sufficient statistics contained in arg 'suff_stat'.

        Sufficient statistics in arg 'suff_stat' are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (List[T1]): List of Sufficient statistics for the emission distribution accumulators.
                Each having type S0.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the type for the sufficient statistics of the emission accumulators. T2 is the type for the
        length accumulator.

        If pseudo_count[0] is not None, the initial counts in 'suff_stat' is re-weighted in estimation.
        If pseudo_count[1] is not None, the transition counts in 'suff_stat' are re-weighted in estimation.


        Args:
            nobs (Optional[float]): Number of observations used in estimation.
            suff_stat: See above for details.
            device (Optional[tn.device]): Device for next model estimate.

        Returns:
            HiddenMarkovModelDistribution object.

        """
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss, device=device)
        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]

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
                transitions[good_rows, :] += trans_counts[good_rows, :] / row_sum[good_rows]
            else:
                transitions = trans_counts / row_sum

        return HiddenMarkovModelDistribution(topics=topics, w=w, transitions=transitions, taus=None, len_dist=len_dist,
                                             device=device)


class HiddenMarkovDataEncoder(TorchSequenceEncoder):

    def __init__(self, emission_encoder: TorchSequenceEncoder,
                 len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder()) -> None:
        """HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations.

        Args:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder object for the length
                of sequences. Should have support of non-negative integers.

        Attributes:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length of sequences.
                Should have support of non-negative integers. Set to NullDataEncoder if None.

        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDataEncoder object instance."""
        s = 'HiddenMarkovDataEncoder(emission_encoder=' + str(self.emission_encoder) + ','
        s += 'len_encoder=' + str(self.len_encoder) + ')'
        return s

    def __eq__(self, other: object) -> bool:
        """Check if other is equivalent to HiddenMarkovDataEncoder object instance.

        Args:
            other (Object): Object to compare to HiddenMarkovDataEncoder object instance.

        Returns:
            True if other is HiddenMarkovDataEncoder with equivalent 'len_encoder' and 'use_numba', else False.

        """
        if isinstance(other, HiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

    def seq_encode(self, x: List[List[T]], device: Optional[tn.device] = None) -> 'HiddenMarkovTorchSequence':
        cnt = len(x)
        len_vec = [len(u) for u in x]
        len_enc = self.len_encoder.seq_encode(len_vec, device=device)

        len_vec = vec.int_tensor(len_vec, device=device)
        max_len = int(tn.max(len_vec))
        # len_cnt = np.bincount(len_vec)

        seq_x = []
        idx_loc = 0
        idx_mat = vec.int_vec((cnt, max_len), device=device) - 1
        idx_bands = []
        has_next = []
        idx_vec = []

        for i in range(max_len):
            i0 = idx_loc
            has_next_loc = []
            for j in range(cnt):
                if i < len_vec[j]:
                    if i < (len_vec[j] - 1):
                        has_next_loc.append(idx_loc - i0)
                    idx_vec.append(j)
                    seq_x.append(x[j][i])
                    idx_mat[j, i] = idx_loc
                    idx_loc += 1

            has_next.append(vec.int_tensor(has_next_loc, device=device))
            idx_bands.append([i0, idx_loc])

        idx_bands = vec.int_tensor(idx_bands, device=device)
        tot_cnt = len(seq_x)
        enc_data = self.emission_encoder.seq_encode(seq_x, device=device)
        idx_vec = vec.int_tensor(idx_vec, device=device)

        return HiddenMarkovTorchSequence(data=((tot_cnt, max_len, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc), device=device)


class HiddenMarkovTorchSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[Tuple[int, int, tn.tensor, List[tn.tensor], tn.tensor, tn.tensor, tn.tensor, TorchEncodedSequence], TorchEncodedSequence], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'HiddenMarkovTorchSequence(device={repr(self.device)})'

