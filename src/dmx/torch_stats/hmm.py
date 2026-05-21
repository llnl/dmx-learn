"""
Create, estimate, and sample from a hidden markov model with K emission distributions.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from math import exp
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union

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
        tn.tensor,
        List[tn.tensor],
        tn.tensor,
        tn.tensor,
        tn.tensor,
        TorchEncodedSequence,
    ],
    TorchEncodedSequence,
]


class HiddenMarkovModelDistribution(TorchProbabilityDistribution):

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
        """
        HiddenMarkovModelDistribution object defining HMM compatible with data type T.
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
        self.transitions = vec.tensor(self.transitions, device=self._device)
        self.log_transitions = tn.log(self.transitions)
        self.terminal_values = terminal_values

        self.len_dist = (
            len_dist if len_dist is not None else NullDistribution(device=self._device)
        )

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
        self.transitions = self.transitions.to(device)

        self.log_w = tn.log(self.w)
        self.log_transitions = tn.log(self.transitions)

        if self.taus is not None:
            self.taus = self.taus.to(device)
            self.log_taus = tn.log(self.taus)

        self._device = device

    def __repr__(self) -> str:
        """Returns string representation of HiddenMarkovDistribution instance."""
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
        """Returns the density of HMM for an observed sequence x."""
        return exp(self.log_density(x))

    def log_density(self, x: List[T]) -> float:
        """Returns the log-density of HMM for observed sequence x."""
        if x is None or len(x) == 0:
            return self.len_dist.log_density(
                0
            )  # this will return 0.0 if NullDistribution()

        enc_data = self.dist_to_encoder().seq_encode([x], device=self._device)

        return float(self.seq_log_density(enc_data)[0])

    def seq_log_density(self, x: "HiddenMarkovTorchSequence") -> tn.Tensor:

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
        """Create a HiddenMarkovSampler object with seed passed."""
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
        """
        Create HiddenMarkovEstimator for estimating HiddenMarkovDistribution objects
        from.
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
        """
        Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM
        observations.
        """
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder
        )


class HiddenMarkovSampler(DistributionSampler):

    def __init__(
        self, dist: "HiddenMarkovModelDistribution", seed: Optional[int] = None
    ) -> None:
        """HiddenMarkovSampler object for sampling from HMM."""
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.has_topics:
            taus = dist.taus.data.cpu().numpy()
            self.obs_samplers = [
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
            self.len_sampler = dist.len_dist.sampler(
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
        """Sample iid HMM sequences."""
        if size is None:
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)
            obs_seq = [self.obs_samplers[state_seq[i]].sample() for i in range(n)]

            return obs_seq

        n = self.len_sampler.sample(size=size)
        state_seq = [self.state_sampler.sample_seq(size=nn) for nn in n]
        obs_seq = [[self.obs_samplers[j].sample() for j in nn] for nn in state_seq]

        return obs_seq

    def sample_terminal(self, terminal_set: Set[T]) -> List[T]:
        """
        Sample an HMM sequence, until a terminal value is samples from the emission.
        """
        z = self.state_sampler.sample_seq()
        rv = [self.obs_samplers[z].sample()]

        while rv[-1] not in terminal_set:
            z = self.state_sampler.sample_seq(v0=z)
            rv.append(self.obs_samplers[z].sample())

        return rv

    def sample(self, size: Optional[int] = None):
        """Draw iid samples from HMM."""
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

    def __init__(
        self,
        accumulators: Sequence[TorchStatisticAccumulator],
        len_accumulator: Optional[TorchStatisticAccumulator] = None,
        keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        device: Optional[tn.device] = None,
    ) -> None:
        super().__init__(device)
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64
        )
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator(device)
        )

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

    def seq_initialize(
        self, x: "HiddenMarkovTorchSequence", weights: tn.Tensor, tng: tn.Generator
    ) -> None:
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
        """
        Combine the sufficient statistics of HiddenMarkovAccumulator with suff_stat arg.
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
        """Returns sufficient statistics of HiddenMarkovAccumulator object instance."""
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
        """
        Set the sufficient statistics of HiddenMarkovAccumulator object instance to
        value x.
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
        """
        Merge the sufficient statistics of object instance with sufficient statistics
        in.
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
        """
        Replace the sufficient statistics of HiddenMarkovAccumulator object with
        matching.
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
        """
        Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM
        observations.
        """
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder
        )


class HiddenMarkovAccumulatorFactory(TorchStatisticAccumulatorFactory):

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
        """
        HiddenMarkovAccumulatorFactory object for creating
        HiddenMarkovEstimatorAccumulator.
        """
        self.factories = factories
        self.keys = keys if keys is None else (None, None, None)
        self.len_factory = len_factory

    def make(self, device: Optional[tn.device] = None) -> "HiddenMarkovAccumulator":
        """Returns a HiddenMarkovAccumulator object."""
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return HiddenMarkovAccumulator(
            [factory.make() for factory in self.factories],
            len_accumulator=len_acc,
            keys=self.keys,
            device=device,
        )


class HiddenMarkovEstimator(TorchParameterEstimator):

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
        """
        HiddenMarkovEstimator object for estimating HiddenMarkovDistribution for
        aggregated.
        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )

    def accumulator_factory(self):
        """Returns an HiddenMarkovAccumulatorFactory object."""
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
        """
        Estimate HiddenMarkovModel from aggregated sufficient statistics contained in
        arg.
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

    def __init__(
        self,
        emission_encoder: TorchSequenceEncoder,
        len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder(),
    ) -> None:
        """
        HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations.
        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDataEncoder object instance."""
        s = (
            "HiddenMarkovDataEncoder(emission_encoder="
            + str(self.emission_encoder)
            + ","
        )
        s += "len_encoder=" + str(self.len_encoder) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Check if other is equivalent to HiddenMarkovDataEncoder object instance."""
        if isinstance(other, HiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

        return False

    def seq_encode(
        self, x: List[List[T]], device: Optional[tn.device] = None
    ) -> "HiddenMarkovTorchSequence":
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

    def __init__(
        self,
        data: Tuple[
            Tuple[
                int,
                int,
                tn.tensor,
                List[tn.tensor],
                tn.tensor,
                tn.tensor,
                tn.tensor,
                TorchEncodedSequence,
            ],
            TorchEncodedSequence,
        ],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"HiddenMarkovTorchSequence(device={repr(self.device)})"
