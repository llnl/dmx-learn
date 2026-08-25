"""Create, estimate, and sample finite-state hidden Markov models.

The torch implementation represents an observation as a Python value and an
observation sequence as ``List[T]``. Batched computations use
``HiddenMarkovTorchSequence``, which packs variable-length sequences by time
step. Model parameters and forward-backward work arrays are torch tensors;
exported HMM count statistics are NumPy arrays.

The likelihood, posterior, and estimation paths implement one emission
distribution per hidden state. The optional ``taus`` topic-mixture matrix is
used by sampling only, unlike the corresponding NumPy implementation, which
also supports topic mixtures in scalar likelihood calculations.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from math import exp
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import numpy as np
import torch as tn
from numpy.random import RandomState

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from dmx.stats.markovchain import MarkovChainDistribution
from dmx.torch_stats.mixture import MixtureDistribution
from dmx.torch_stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

T = TypeVar("T")
T1 = TypeVar("T1")  # Emission suff-stat type
T2 = TypeVar("T2")  # Len suff-stat type
E = Tuple[
    Tuple[
        int,
        int,
        tn.Tensor,
        List[tn.Tensor],
        tn.Tensor,
        tn.Tensor,
        tn.Tensor,
        TorchEncodedSequence,
    ],
    TorchEncodedSequence,
]


class HiddenMarkovModelDistribution(TorchProbabilityDistribution):
    """Represent a finite-state HMM with observations of generic type ``T``.

    For ``K`` hidden states, ``w[k]`` is the initial-state probability and
    ``transitions[i, j]`` is the probability of moving from state ``i`` to
    state ``j``. The vectorized likelihood, posterior, Viterbi, and estimation
    methods interpret ``topics[k]`` as the emission distribution for state
    ``k`` and therefore require ``len(topics) == K``. If ``taus`` is supplied,
    sampling instead uses row ``k`` as topic-mixture weights for state ``k``.

    Floating-point parameters use the dtype selected by
    ``dmx.torch_utils.vector``: normally ``torch.float64``, but
    ``torch.float32`` on MPS. They are created on ``device`` and moved by
    :meth:`to`. The length distribution is constructed on that device when it
    is omitted, but a supplied length distribution is not moved by :meth:`to`.

    Attributes:
        topics: Emission distributions, normally one per hidden state.
        n_topics: Number of emission distributions.
        n_states: Number of hidden states, inferred from ``w``.
        w: Initial-state probabilities with shape ``(K,)``.
        log_w: Elementwise logarithm of ``w``, with shape ``(K,)``.
        transitions: Transition probabilities with shape ``(K, K)``.
        log_transitions: Elementwise logarithm of ``transitions``.
        taus: Optional sampling-only topic weights, normally shape
            ``(K, n_topics)``.
        log_taus: Elementwise logarithm of ``taus``, when present.
        has_topics: Whether ``taus`` was supplied.
        len_dist: Distribution for observed sequence lengths.
        terminal_values: Optional emissions that terminate sampling when no
            usable length sampler is present.
    """

    def __init__(
        self,
        topics: Sequence[TorchProbabilityDistribution],
        w: Union[Sequence[float], np.ndarray],
        transitions: Union[List[List[float]], np.ndarray],
        taus: Optional[Union[List[List[float]], np.ndarray]] = None,
        len_dist: Optional[TorchProbabilityDistribution] = None,
        terminal_values: Optional[Set[T]] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize a hidden Markov model.

        Args:
            topics: Emission distributions. Vectorized inference expects one
                distribution per hidden state.
            w: Initial-state probabilities with shape ``(K,)``.
            transitions: Transition probabilities, reshaped to ``(K, K)``.
            taus: Optional topic-mixture weights used by the sampler only.
            len_dist: Distribution of sequence lengths. ``None`` selects a
                null distribution.
            terminal_values: Optional emissions that terminate sampling.
            device: Device for HMM parameter tensors. ``None`` selects CPU.
        """
        super().__init__(device)
        self.topics: Sequence[TorchProbabilityDistribution] = topics
        self.n_topics = len(topics)
        self.n_states = len(w)
        self.w: tn.Tensor = vec.tensor(w, device=self._device)
        self.log_w: tn.Tensor = tn.log(self.w)

        transitions_arr = np.asarray(transitions, dtype=float)
        transitions_arr = np.reshape(transitions_arr, (self.n_states, self.n_states))
        self.transitions: tn.Tensor = vec.tensor(transitions_arr, device=self._device)
        self.log_transitions: tn.Tensor = tn.log(self.transitions)
        self.terminal_values: Optional[Set[T]] = terminal_values

        self.len_dist: TorchProbabilityDistribution = (
            len_dist if len_dist is not None else NullDistribution(device=self._device)
        )
        self.taus: Optional[tn.Tensor]
        self.log_taus: Optional[tn.Tensor]

        if taus is not None:
            self.taus = vec.tensor(taus, device=self._device)
            self.log_taus = tn.log(self.taus)
            self.has_topics = True
        else:
            self.taus = None
            self.log_taus = None
            self.has_topics = False

    def to(self, device: vec.DeviceLike) -> "HiddenMarkovModelDistribution":
        """Move HMM parameters and emission distributions to a device.

        The initial and transition tensors, optional topic weights, and all
        emission distributions are moved. The length distribution is retained
        as-is. Parameter dtypes are preserved.

        Args:
            device: Target device. ``None`` retains the current model device.

        Returns:
            This distribution after the in-place move.
        """
        target_device = self._resolve_device_arg(device)
        for dist in self.topics:
            dist.to(target_device)

        self.w = self.w.to(target_device)
        self.transitions = self.transitions.to(target_device)

        self.log_w = tn.log(self.w)
        self.log_transitions = tn.log(self.transitions)

        if self.taus is not None:
            self.taus = self.taus.to(target_device)
            self.log_taus = tn.log(self.taus)

        self._device = target_device
        return self

    def __repr__(self) -> str:
        """Return a string representation of the HMM."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(self.w.data.cpu().tolist())
        s3 = repr(list(self.transitions.data.cpu().tolist()))
        if self.taus is None:
            s4 = repr(None)
        else:
            s4 = repr(list(self.taus.data.cpu().tolist()))
        s5 = str(self.len_dist)
        s6 = repr(self.terminal_values)

        return (
            f"HiddenMarkovModelDistribution([{s1}], {s2}, {s3}, {s4}, "
            f"len_dist={s5}, terminal_values={s6})"
        )

    def density(self, x: List[T]) -> float:
        """Evaluate the density of one observed sequence.

        Args:
            x: Observed emission sequence.

        Returns:
            The marginal sequence density as a Python float.
        """
        return exp(self.log_density(x))

    def log_density(self, x: List[T]) -> float:
        """Evaluate the HMM log density of one observed sequence.

        The hidden-state path is marginalized by the batched forward routine.
        An empty sequence contributes only its length log density.

        Args:
            x: Observed emission sequence.

        Returns:
            The marginal log density as a Python float.
        """
        if x is None or len(x) == 0:
            return self.len_dist.log_density(
                0
            )  # this will return 0.0 if NullDistribution()

        enc_data = self.dist_to_encoder().seq_encode([x], device=self._device)

        return float(self.seq_log_density(enc_data)[0])

    def seq_log_density(self, x: "HiddenMarkovTorchSequence") -> tn.Tensor:
        """Evaluate log densities for a batch of observation sequences.

        A scaled forward recursion marginalizes each hidden-state path. For a
        batch of ``N`` sequences, the returned tensor has shape ``(N,)`` on
        the model device and uses the model's floating-point dtype. Sequence
        length log densities are included. This path assumes one emission
        distribution per state and does not use ``taus``.

        Args:
            x: Batch encoded by :class:`HiddenMarkovDataEncoder`.

        Returns:
            Marginal log densities, one per input sequence.

        Raises:
            TypeError: If ``x`` is not a ``HiddenMarkovTorchSequence``.
        """
        num_states = self.n_states

        if not isinstance(x, HiddenMarkovTorchSequence):
            raise TypeError(
                "HiddenMarkovTorchSequence required for `seq_` function calls."
            )

        (
            tot_cnt,
            max_len,
            idx_bands,
            has_next,
            _,
            idx_mat,
            _,
            enc_data,
        ), len_enc = x.data
        w = self.w
        a_mat = self.transitions

        num_seq = int(idx_mat.shape[0])

        good = idx_mat >= 0
        good_cpu = good.cpu()

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

        alphas_prev = tn.multiply(pr_obs[band0[0] : band1[0], :], w)
        temp = alphas_prev.sum(dim=1, keepdim=True)
        # temp2 = temp.copy()
        # temp2[temp2 == 0] = 1.0
        alphas_prev /= temp

        tn.log(temp, out=temp)
        temp2 = pr_max0[band0[0] : band1[0], 0]
        ll_ret[good_cpu[:, 0].to(device=ll_ret.device)] += temp[:, 0] + temp2

        for i in range(1, max_len):
            band = idx_bands[i]
            has_next_loc = has_next[i - 1]

            alphas_next = tn.matmul(alphas_prev[has_next_loc, :], a_mat)
            alphas_next *= pr_obs[band[0] : band[1], :]
            pr_max = alphas_next.sum(dim=1, keepdim=True)
            # pr_max2 = pr_max.copy()
            # pr_max2[pr_max2 == 0] = 1.0
            alphas_next /= pr_max
            alphas_prev = alphas_next

            tn.log(pr_max, out=pr_max)
            temp2 = pr_max0[band0[i] : band1[i], 0]
            ll_ret[good_cpu[:, i].to(device=ll_ret.device)] += pr_max[:, 0] + temp2

        # nz = len_vec != 0
        # ll_ret[nz] /= len_vec[nz]

        ll_ret[tn.isnan(ll_ret)] = -tn.inf

        if self.len_dist is not None:
            ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def viterbi(self, x: List[T]) -> tn.Tensor:
        """Return per-time maximizing states from Viterbi score vectors.

        The output is a floating-point tensor of shape ``(L,)`` on the model
        device for a length-``L`` input. Each entry is the index maximizing the
        dynamic-programming score at that time. This implementation does not
        retain backpointers or perform traceback, so the result need not be the
        globally maximizing state path. The sequence-length distribution and
        optional ``taus`` matrix are not used.

        Args:
            x: One nonempty observed emission sequence.

        Returns:
            Per-time hidden-state indices stored in a floating-point tensor.
        """
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
            temp += tn.reshape(v[t - 1, :], (num_states, 1))
            temp += self.log_transitions
            temp += tn.reshape(pr_obs[t, :], (1, num_states))
            temp, _ = temp.max(dim=0, keepdim=False)
            v[t, :] += temp

        for t in range(nn - 1, -1, -1):
            ptr[t] = tn.argmax(v[t, :])

        return ptr

    def sampler(self, seed: Optional[int] = None) -> "HiddenMarkovSampler":
        """Create a seeded sampler for independent HMM sequences.

        Args:
            seed: Optional NumPy random seed.

        Returns:
            A sampler bound to this distribution.

        Raises:
            RuntimeError: If neither sequence lengths nor terminal emissions
                can stop sampling.
        """
        if isinstance(self.len_dist, NullDistribution) and self.terminal_values is None:
            raise RuntimeError(
                "HiddenMarkovSampler requires len_dist with support on "
                "non-negative integers, or terminal_"
                "values to be set."
            )

        return HiddenMarkovSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "HiddenMarkovEstimator":
        """Create an estimator with matching emission and length estimators.

        The same scalar pseudo-count is used for initial-state probabilities,
        transition probabilities, and delegated component estimators.

        Args:
            pseudo_count: Optional additive smoothing mass.

        Returns:
            An HMM parameter estimator.
        """
        len_est = (
            None
            if self.len_dist is None
            else self.len_dist.estimator(pseudo_count=pseudo_count)
        )
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return HiddenMarkovEstimator(
            comp_ests, pseudo_count=(pseudo_count, pseudo_count), len_estimator=len_est
        )

    def dist_to_encoder(self) -> "HiddenMarkovDataEncoder":
        """Create an encoder for batches of HMM observation sequences.

        Returns:
            An encoder using the first emission distribution's encoder and the
            length distribution's encoder.
        """
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder
        )


class HiddenMarkovSampler(DistributionSampler):
    """Generate observation sequences from an HMM.

    Hidden states are sampled with the NumPy-backed Markov-chain sampler. Each
    state's observation sampler is either its corresponding emission sampler
    or, when ``taus`` is present, a mixture over all topic samplers. Sampled
    values are ordinary Python objects rather than torch tensors.
    """

    def __init__(
        self, dist: "HiddenMarkovModelDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler from an HMM distribution.

        Args:
            dist: Source HMM.
            seed: Optional NumPy random seed used to seed child samplers.
        """
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.has_topics:
            assert dist.taus is not None
            taus = dist.taus.data.cpu().numpy()
            self.obs_samplers: List[DistributionSampler] = [
                MixtureDistribution(dist.topics, taus[i, :]).sampler(
                    seed=self.rng.randint(0, maxrandint)
                )
                for i in range(dist.n_states)
            ]
        else:
            self.obs_samplers = [
                dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint))
                for i in range(dist.n_states)
            ]

        if dist.len_dist is not None:
            self.len_sampler: Optional[DistributionSampler] = dist.len_dist.sampler(
                seed=self.rng.randint(0, maxrandint)
            )
        else:
            self.len_sampler = None

        if dist.terminal_values is None:
            self.terminal_set = None
        else:
            self.terminal_set = set(dist.terminal_values)

        trans = dist.transitions.data.cpu().numpy().astype(np.float64, copy=True)
        w = dist.w.data.cpu().numpy().astype(np.float64, copy=True)

        w_sum = w.sum()
        if w_sum > 0.0:
            w /= w_sum

        row_sums = trans.sum(axis=1, keepdims=True)
        good_rows = row_sums[:, 0] > 0.0
        if np.any(good_rows):
            trans[good_rows, :] /= row_sums[good_rows]
        if np.any(~good_rows):
            trans[~good_rows, :] = 1.0 / dist.n_states

        t_map = {
            i: {k: trans[i, k] for k in range(dist.n_states)}
            for i in range(dist.n_states)
        }
        p_map = {i: w[i] for i in range(dist.n_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(
            seed=self.rng.randint(0, maxrandint)
        )

    def sample_seq(
        self, size: Optional[int] = None
    ) -> Union[List[Any], List[List[Any]]]:
        """Sample independent sequences using random sequence lengths.

        Args:
            size: Number of sequences. ``None`` returns one sequence.

        Returns:
            One observation list, or a list of observation lists when ``size``
            is provided.
        """
        assert self.len_sampler is not None
        if size is None:
            n = int(self.len_sampler.sample())
            state_seq = cast(List[int], self.state_sampler.sample_seq(size=n))
            obs_seq = [self.obs_samplers[state_seq[i]].sample() for i in range(n)]

            return obs_seq

        n_values = [int(nn) for nn in self.len_sampler.sample(size=size)]
        state_seqs = [
            cast(List[int], self.state_sampler.sample_seq(size=nn)) for nn in n_values
        ]
        obs_seq = [[self.obs_samplers[j].sample() for j in nn] for nn in state_seqs]

        return obs_seq

    def sample_terminal(self, terminal_set: Set[T]) -> List[T]:
        """Sample through the first emission in a terminal set.

        Args:
            terminal_set: Emission values that stop the sequence.

        Returns:
            A sequence including its terminal emission.
        """
        z = cast(int, self.state_sampler.sample_seq())
        rv: List[T] = [self.obs_samplers[z].sample()]

        while rv[-1] not in terminal_set:
            z = cast(int, self.state_sampler.sample_seq(v0=z))
            rv.append(self.obs_samplers[z].sample())

        return rv

    def sample(self, size: Optional[int] = None) -> Union[List[Any], List[List[Any]]]:
        """Draw independent HMM observation sequences.

        Args:
            size: Number of sequences. ``None`` returns one sequence.

        Returns:
            One observation list, or a list of observation lists when ``size``
            is provided.

        Raises:
            RuntimeError: If the sampler has no stopping mechanism.
        """
        if self.len_sampler is not None:
            return self.sample_seq(size=size)

        if self.terminal_set is not None:
            if size is None:
                return self.sample_terminal(self.terminal_set)
            return [self.sample_terminal(self.terminal_set) for _ in range(size)]

        raise RuntimeError(
            "HiddenMarkovSampler requires either a length distribution or "
            "terminal value set."
        )


class HiddenMarkovAccumulator(TorchStatisticAccumulator):
    """Accumulate sufficient statistics for HMM parameter estimation.

    For ``K`` states, initial counts have shape ``(K,)``, transition counts
    have shape ``(K, K)``, and posterior state counts have shape ``(K,)``.
    These counts are NumPy ``float64`` arrays on CPU. Forward-backward work and
    posterior emission weights are torch tensors on the accumulator device;
    component and length statistics retain their own accumulator-defined
    representations.

    ``seq_initialize`` assigns states randomly, while ``seq_update`` computes
    posterior state and transition weights under a current estimate. Input
    ``weights`` has shape ``(N,)`` for ``N`` encoded sequences and is expanded
    across their observations.
    """

    def __init__(
        self,
        accumulators: Sequence[TorchStatisticAccumulator],
        len_accumulator: Optional[TorchStatisticAccumulator] = None,
        keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize an HMM sufficient-statistic accumulator.

        Args:
            accumulators: One emission accumulator per hidden state.
            len_accumulator: Sequence-length accumulator. ``None`` selects a
                null accumulator.
            keys: Optional shared keys for initial, transition, and emission
                statistics.
            device: Device for forward-backward work tensors. ``None`` selects
                CPU.
        """
        super().__init__(device)
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64
        )
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator: TorchStatisticAccumulator = (
            len_accumulator
            if len_accumulator is not None
            else NullAccumulator(device=device)
        )

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

    def seq_initialize(
        self, x: "HiddenMarkovTorchSequence", weights: tn.Tensor, tng: tn.Generator
    ) -> None:
        """Initialize sufficient statistics with random hidden states.

        Each packed observation receives a uniformly sampled state. Sequence
        weights are then accumulated into initial, state, transition,
        emission, and length statistics.

        Args:
            x: Encoded batch of ``N`` observation sequences.
            weights: Floating-point sequence weights with shape ``(N,)`` on a
                device compatible with ``x``.
            tng: Torch generator used for random state assignments.
        """
        (
            tot_cnt,
            max_len,
            idx_bands,
            has_next,
            _,
            _,
            idx_vec,
            enc_data,
        ), len_enc = x.data

        self.len_accumulator.seq_initialize(len_enc, weights, tng)

        weights_loc = weights[idx_vec]

        band0 = idx_bands[:, 0]
        band1 = idx_bands[:, 1]

        idx = vec.randint(size=tot_cnt, low=0, high=self.num_states, tng=tng)

        # count the states
        self.state_counts += (
            tn.bincount(idx, weights=weights_loc, minlength=self.num_states)
            .cpu()
            .detach()
            .numpy()
        )

        # count initial states
        b0, b1 = band0[0], band1[0]
        tmp = tn.bincount(
            idx[b0:b1], weights=weights_loc[b0:b1], minlength=self.num_states
        )
        tmp_np = tmp.cpu().detach().numpy()
        self.init_counts += tmp_np

        # Vectorized alpha pass
        idx_prev = idx[b0:b1]
        w_prev = weights_loc[b0:b1]

        tcnts = vec.zeros(self.num_states**2, device=self._device)

        for i in range(1, max_len):
            b0, b1 = band0[i], band1[i]
            has_next_loc = has_next[i - 1]
            idx_next = idx[b0:b1]

            idx0 = idx_prev[has_next_loc] * self.num_states + idx_next
            w_prev = w_prev[has_next_loc]
            tcnts += tn.bincount(idx0, weights=w_prev, minlength=self.num_states**2)

            idx_prev = idx_next

        for j in range(self.num_states):
            self.accumulators[j].seq_initialize(
                enc_data, tn.where(idx == j, weights_loc, 0.0), tng
            )

        self.trans_counts += (
            tcnts.reshape(self.num_states, self.num_states).cpu().detach().numpy()
        )

    def seq_update(
        self,
        x: "HiddenMarkovTorchSequence",
        weights: tn.Tensor,
        estimate: HiddenMarkovModelDistribution,
    ) -> None:
        """Accumulate expected sufficient statistics under an HMM estimate.

        A scaled forward-backward calculation produces posterior state and
        adjacent-state weights. It accumulates weighted expected initial
        counts, state occupancies, transition counts, emission statistics, and
        length statistics. The tensor calculations use the accumulator device;
        HMM count arrays are copied to CPU NumPy arrays.

        Args:
            x: Encoded batch of ``N`` observation sequences.
            weights: Floating-point sequence weights with shape ``(N,)``.
            estimate: Current one-emission-per-state HMM estimate.
        """
        num_states = self.num_states
        (
            tot_cnt,
            max_len,
            idx_bands,
            has_next,
            _,
            idx_mat,
            idx_vec,
            enc_data,
        ), len_enc = x.data
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
        alphas_prev = alphas[band0[0] : band1[0], :]
        tn.multiply(pr_obs[band0[0] : band1[0], :], w, out=alphas_prev)

        # tn.multiply(pr_obs[band0[0]:band0[0], :], w, out=alphas_prev)
        pr_sum = alphas_prev.sum(dim=1, keepdim=True)
        pr_sum[pr_sum == 0.0] = 1.0
        alphas_prev /= pr_sum

        for i in range(1, max_len):
            has_next_loc = has_next[i - 1]
            alphas_next = alphas[band0[i] : band1[i], :]
            tn.matmul(alphas_prev[has_next_loc, :], a_mat, out=alphas_next)
            alphas_next *= pr_obs[band0[i] : band1[i], :]

            pr_max = alphas_next.sum(dim=1, keepdim=True)
            pr_max[pr_max == 0.0] = 1.0

            alphas_next /= pr_max
            alphas_prev = alphas_next

        prev_beta = vec.ones((band1[-1] - band0[-1], num_states), device=self._device)
        alphas[band0[-1] : band1[-1], :] /= alphas[band0[-1] : band1[-1], :].sum(
            dim=1, keepdim=True
        )

        tcnts = vec.zeros((self.num_states, self.num_states), device=self._device)

        # Vectorized beta pass
        for i in range(max_len - 2, -1, -1):
            # band1 = idx_bands[i]
            # band2 = idx_bands[i + 1]
            has_next_loc = has_next[i]

            next_b = pr_obs[band0[i + 1] : band1[i + 1], :]
            prev_a = alphas[band0[i] : band1[i], :]
            prev_a = prev_a[has_next_loc, :]

            prev_beta *= next_b

            prev_a = prev_a.reshape((prev_a.shape[0], prev_a.shape[1], 1))
            next_beta2 = prev_beta.reshape((prev_beta.shape[0], 1, prev_beta.shape[1]))

            xi_loc = next_beta2 * a_mat
            next_beta = xi_loc.sum(dim=2)
            next_beta_max, _ = next_beta.max(dim=1, keepdim=True)
            next_beta_max[next_beta_max == 0.0] = 1.0
            next_beta /= next_beta_max

            prev_beta = vec.ones(
                (int(band1[i] - band0[i]), num_states), device=self._device
            )
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

        temp = alphas[band0[0] : band1[0], :].sum(dim=1, keepdim=True)
        temp[temp == 0] = 1.0
        alphas[band0[0] : band1[0], :] *= (
            tn.reshape(weights[good[:, 0]], (-1, 1)) / temp
        )

        self.init_counts += (
            alphas[band0[0] : band1[0], :].sum(dim=0).cpu().detach().numpy()
        )

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(
        self,
        suff_stat: Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]
        ],
    ) -> "HiddenMarkovAccumulator":
        """Add another HMM sufficient-statistic value in place.

        Args:
            suff_stat: Tuple returned by :meth:`value`.

        Returns:
            This accumulator after combining statistics.
        """
        (
            _,
            init_counts,
            state_counts,
            trans_counts,
            acc_values,
            len_acc_value,
        ) = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.accumulators[i].combine(acc_values[i])

        if len_acc_value is not None:
            self.len_accumulator.combine(len_acc_value)

        return self

    def value(
        self,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Optional[Any]]:
        """Return the accumulated HMM sufficient statistics.

        Returns:
            A tuple containing the number of states, initial counts of shape
            ``(K,)``, state counts of shape ``(K,)``, transition counts of
            shape ``(K, K)``, ``K`` emission-statistic values, and the optional
            length-statistic value. The three HMM count arrays are CPU NumPy
            ``float64`` arrays.
        """
        len_val = self.len_accumulator.value()

        return (
            self.num_states,
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            tuple(u.value() for u in self.accumulators),
            len_val,
        )

    def from_value(
        self,
        x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]],
    ) -> "HiddenMarkovAccumulator":
        """Replace accumulated statistics from a serialized value.

        Args:
            x: Tuple in the format returned by :meth:`value`.

        Returns:
            This accumulator after replacement.
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
        """Merge configured sufficient statistics into a shared dictionary.

        Args:
            stats_dict: Mutable mapping keyed by configured statistic names.
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
                for i, acc_item in enumerate(acc):
                    acc[i] = acc_item.combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace configured statistics from a shared dictionary.

        Args:
            stats_dict: Mapping keyed by configured statistic names.
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

    def acc_to_encoder(self) -> "HiddenMarkovDataEncoder":
        """Create an encoder compatible with this accumulator.

        Returns:
            An HMM encoder built from the first emission accumulator and the
            length accumulator.
        """
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder
        )


class HiddenMarkovAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create HMM accumulators from component accumulator factories."""

    def __init__(
        self,
        factories: Sequence[TorchStatisticAccumulatorFactory],
        len_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initialize an HMM accumulator factory.

        Args:
            factories: One emission accumulator factory per hidden state.
            len_factory: Factory for sequence-length statistics.
            keys: Optional shared keys for initial, transition, and emission
                statistics.
        """
        self.factories = factories
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory

    def make(self, device: Optional[tn.device] = None) -> "HiddenMarkovAccumulator":
        """Create an HMM accumulator.

        The HMM work tensors use ``device``. Component factories are invoked
        without a device argument and therefore retain their own defaults.

        Args:
            device: Device for HMM forward-backward work tensors.

        Returns:
            A fresh HMM accumulator.
        """
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return HiddenMarkovAccumulator(
            [factory.make() for factory in self.factories],
            len_accumulator=len_acc,
            keys=self.keys,
            device=device,
        )


class HiddenMarkovEstimator(TorchParameterEstimator):
    """Estimate HMM parameters from aggregated sufficient statistics.

    Emission estimators receive their posterior state counts as effective
    observation counts. The length estimator receives the caller's ``nobs``.
    Initial and transition probabilities are normalized from CPU NumPy count
    arrays, with optional additive pseudo-counts.
    """

    def __init__(
        self,
        estimators: List[TorchParameterEstimator],
        len_estimator: Optional[TorchParameterEstimator] = None,
        pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initialize an HMM estimator.

        Args:
            estimators: One emission estimator per hidden state.
            len_estimator: Sequence-length estimator. ``None`` selects a null
                estimator.
            pseudo_count: Optional pair of smoothing masses for initial and
                transition probabilities.
            keys: Optional shared keys for initial, transition, and emission
                statistics.
        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )

    def accumulator_factory(self) -> "HiddenMarkovAccumulatorFactory":
        """Create a factory for compatible HMM accumulators.

        Returns:
            A factory wrapping the emission and length accumulator factories.
        """
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HiddenMarkovAccumulatorFactory(est_factories, len_factory)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, List[T1], Optional[T2]
        ],
        device: Optional[tn.device] = None,
    ) -> "HiddenMarkovModelDistribution":
        """Estimate an HMM from aggregated sufficient statistics.

        Initial counts are normalized globally. Transition counts are
        normalized by row; without smoothing, a row with no transitions remains
        all zeros. A supplied initial pseudo-count is spread uniformly over
        states, and a transition pseudo-count uniformly over the full ``K`` by
        ``K`` matrix. Topic mixtures are not estimated, so the returned model
        always has ``taus=None``.

        Args:
            nobs: Effective number of sequences, forwarded to the length
                estimator.
            suff_stat: Tuple from :meth:`HiddenMarkovAccumulator.value`.
            device: Device for the returned HMM parameters and forwarded length
                estimate. ``None`` selects CPU for the HMM.

        Returns:
            The estimated one-emission-per-state HMM.
        """
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = (
            suff_stat
        )

        len_dist = self.len_estimator.estimate(nobs, len_ss, device=device)
        topics = [
            self.estimators[i].estimate(state_counts[i], topic_ss[i])
            for i in range(num_states)
        ]

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

        return HiddenMarkovModelDistribution(
            topics=topics,
            w=w,
            transitions=transitions,
            taus=None,
            len_dist=len_dist,
            device=device,
        )


class HiddenMarkovDataEncoder(TorchSequenceEncoder):
    """Encode batches of variable-length HMM observation sequences.

    A batch of ``N`` sequences is packed by time step. If ``L`` is the maximum
    sequence length and ``M`` is the total observation count, integer indexing
    tensors describe the mapping between the padded ``(N, L)`` view and the
    packed ``M`` observations. Integer tensors use the torch utility integer
    dtype, currently ``torch.int32``, on the requested device.
    """

    def __init__(
        self,
        emission_encoder: TorchSequenceEncoder,
        len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder(),
    ) -> None:
        """Initialize an HMM batch encoder.

        Args:
            emission_encoder: Encoder for flattened emission values.
            len_encoder: Encoder for sequence lengths. ``None`` selects a null
                encoder.
        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a string representation of the encoder."""
        s = (
            "HiddenMarkovDataEncoder(emission_encoder="
            + str(self.emission_encoder)
            + ","
        )
        s += "len_encoder=" + str(self.len_encoder) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same length encoder."""
        if isinstance(other, HiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

        return False

    def seq_encode(
        self, x: List[List[T]], device: Optional[tn.device] = None
    ) -> "HiddenMarkovTorchSequence":
        """Encode a batch of variable-length observation sequences.

        Observations are flattened in time-major order before delegation to the
        emission encoder. For ``N`` sequences, maximum length ``L``, and ``M``
        total observations, the payload contains ``len_vec`` with shape
        ``(N,)``, ``idx_mat`` with shape ``(N, L)``, ``idx_bands`` with shape
        ``(L, 2)``, and ``idx_vec`` with shape ``(M,)``. ``idx_mat`` stores
        packed indices and uses ``-1`` for padding. ``has_next[t]`` indexes the
        packed observations at time ``t`` whose sequences continue.

        Args:
            x: Batch of observation sequences.
            device: Device for index tensors and delegated encodings. ``None``
                uses the called encoders' default behavior while the returned
                container records CPU.

        Returns:
            The packed HMM batch, including delegated emission and length
            encodings.
        """
        cnt = len(x)
        len_values = [len(u) for u in x]
        len_enc = self.len_encoder.seq_encode(len_values, device=device)

        len_vec = vec.int_tensor(len_values, device=device)
        max_len = max(len_values) if len_values else 0
        # len_cnt = np.bincount(len_vec)

        seq_x: List[T] = []
        idx_loc = 0
        idx_mat = vec.int_vec((cnt, max_len), device=device) - 1
        idx_bands_list: List[List[int]] = []
        has_next: List[tn.Tensor] = []
        idx_vec_list: List[int] = []

        for i in range(max_len):
            i0 = idx_loc
            has_next_loc: List[int] = []
            for j in range(cnt):
                if i < len_values[j]:
                    if i < (len_values[j] - 1):
                        has_next_loc.append(idx_loc - i0)
                    idx_vec_list.append(j)
                    seq_x.append(x[j][i])
                    idx_mat[j, i] = idx_loc
                    idx_loc += 1

            has_next.append(vec.int_tensor(has_next_loc, device=device))
            idx_bands_list.append([i0, idx_loc])

        idx_bands = vec.int_tensor(idx_bands_list, device=device)
        tot_cnt = len(seq_x)
        enc_data = self.emission_encoder.seq_encode(seq_x, device=device)
        idx_vec = vec.int_tensor(idx_vec_list, device=device)

        return HiddenMarkovTorchSequence(
            data=(
                (
                    tot_cnt,
                    max_len,
                    idx_bands,
                    has_next,
                    len_vec,
                    idx_mat,
                    idx_vec,
                    enc_data,
                ),
                len_enc,
            ),
            device=device,
        )


class HiddenMarkovTorchSequence(TorchEncodedSequence):
    """Store a packed batch of variable-length HMM sequences.

    ``data`` is ``((M, L, idx_bands, has_next, len_vec, idx_mat, idx_vec,
    enc_data), len_enc)`` for total observation count ``M`` and maximum length
    ``L``. Index tensors reside on ``device`` and use ``torch.int32``; the
    delegated emission and length encodings control their own tensor dtypes.
    The container records a device but does not move its payload.
    """

    def __init__(
        self,
        data: Tuple[
            Tuple[
                int,
                int,
                tn.Tensor,
                List[tn.Tensor],
                tn.Tensor,
                tn.Tensor,
                tn.Tensor,
                TorchEncodedSequence,
            ],
            TorchEncodedSequence,
        ],
        device: Optional[tn.device] = None,
    ):
        """Initialize a packed HMM sequence container.

        Args:
            data: Packed index tensors plus emission and length encodings.
            device: Device recorded for the payload. ``None`` records CPU.
        """
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the recorded device."""
        return f"HiddenMarkovTorchSequence(device={repr(self.device)})"
