r"""Provide hidden Markov models over rooted, ordered node arrays.

One observed tree is a sequence ``[(d_0, y_0), ..., (d_{n-1}, y_{n-1})]``.
Each descriptor :math:`d_v=(v,p_v)` gives a node index and its parent index;
the root is ``(0, -1)`` and every non-root parent precedes its child.  Each
node :math:`v` has an emission :math:`y_v` and a latent state
:math:`Z_v\in\{0,\ldots,K-1\}`.  With
:math:`\pi_i=P(Z_0=i)`, :math:`A_{ij}=P(Z_c=j\mid Z_p=i)`, and
:math:`b_i(y)=p(y\mid Z_v=i)`, the state-and-emission model is

.. math::

   p(y,z\mid d) = \pi_{z_0}b_{z_0}(y_0)
      \prod_{v=1}^{n-1} A_{z_{p_v},z_v}b_{z_v}(y_v).

``w`` stores :math:`\boldsymbol{\pi}` with shape ``(K,)``;
``transitions`` stores :math:`A` with shape ``(K, K)``; and ``topics[i]``
provides :math:`b_i`.  A posterior for one tree has shape ``(n, K)`` and a
Viterbi result has shape ``(n,)``, both ordered exactly as the input nodes.

The encoder converts a batch of trees into flattened node emissions plus
integer parent/child and tree-boundary arrays.  It supports equivalent NumPy
and Numba layouts; those arrays are implementation details, while callers
continue to pass the tree sequence described above.  Inference marginalizes
or maximizes over node states with tree upward/downward recursions.  Estimation
collects root counts, parent-to-child transition counts, state-weighted
emission statistics, and child-count statistics.  ``len_dist`` governs child
counts while sampling and is updated from those statistics, but it is not an
additional factor in ``log_density`` or ``seq_log_density``.

Please give focused review to this tree notation, the parent-index convention,
and the rendered math before treating it as canonical public documentation.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numba
import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import exp, maxrandint
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

D = Tuple[int, Optional[int]]
T = TypeVar("T")  # Type for emissions
SS0 = TypeVar("SS0")  # Type for suff stat of emissions
SS1 = TypeVar("SS1")  # Type for suff-stat of length dist

E = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
]
N3 = Tuple[np.ndarray, np.ndarray, np.ndarray]
N4 = Tuple[int, np.ndarray, np.ndarray, np.ndarray]
N7 = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
E0 = Tuple[
    bool,
    Tuple[
        np.ndarray, N4, N7, EncodedDataSequence, Tuple[np.ndarray, EncodedDataSequence]
    ],
]  # numba
E1 = Tuple[
    bool,
    Tuple[
        int,
        np.ndarray,
        N3,
        E,
        EncodedDataSequence,
        Optional[Tuple[np.ndarray, EncodedDataSequence]],
    ],
]

# def get_combos(u: Union[List[int], np.ndarray]) -> Tuple[List[int], List[int]]:
#     """Get
#
#     Args:
#         u:
#
#     Returns:
#
#     """
#     v = np.asarray(u, dtype=np.int32)
#     nv = len(v) - 1
#     combs = itertools.combinations(v, nv)
#     singles = [v[i] for i in range(nv, -1, -1) for j in range(nv)]
#     return [v for x in list(combs) for v in x], singles


def find_level(parents: np.ndarray) -> List[int]:
    """Find the level in the tree for nodes, given an array of parents.

    Args:
        parents (np.ndarray): Numpy array of integers with first entry -1.

    Returns:
        Level of each node in the free excluding the first entry which is the root
        (level = 0).

    """
    n = len(parents)
    if n == 1:
        return []
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        out[i] = out[parents[i]] + 1
    return list(out[1:])


class TreeHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a finite-state HMM whose observations are rooted trees.

    Attributes:
        topics: State-indexed emission distributions.
        num_states: Number of latent states, ``K``.
        w: Root-state probabilities with shape ``(K,)``.
        transitions: Parent-to-child probabilities with shape ``(K, K)``.
        len_dist: Child-count distribution used by sampling and estimation.
        terminal_level: Maximum depth used only while sampling.
        use_numba: Whether to use the Numba-oriented encoded layout.
    """

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        w: Union[Sequence[float], np.ndarray],
        transitions: Union[List[List[float]], np.ndarray],
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        terminal_level: int = 10,
        name: Optional[str] = None,
        use_numba: bool = False,
    ) -> None:
        """Initialize a tree hidden Markov model.

        Args:
            topics: One emission distribution for each latent state.
            w: Root-state probability vector with shape ``(K,)``.
            transitions: Parent-to-child transition matrix with shape ``(K, K)``.
            len_dist: Child-count distribution for sampling and estimation.
            terminal_level: Maximum sampling depth.
            name: Optional distribution name.
            use_numba: Select the Numba-oriented encoded representation.
        """
        super().__init__()
        with np.errstate(divide="ignore"):
            self.topics = topics
            self.num_states = len(w)
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(
                transitions, (self.num_states, self.num_states)
            )
            self.log_transitions = np.log(self.transitions)
            self.name = name
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.terminal_level = terminal_level
            self.use_numba = use_numba

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr(list(self.transitions.tolist()))
        s4 = str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.use_numba)

        return (
            f"TreeHiddenMarkovModelDistribution(topics=[{s1}], w={s2}, "
            f"transitions={s3}, len_dist={s4}, name={s5}, use_numba={s6})"
        )

    def density(self, x: Sequence[Tuple[D, T]]) -> float:
        """Return the marginal density of one observed tree."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Sequence[Tuple[D, T]]) -> float:
        """Return the marginal log density of one observed tree."""
        enc_x = self.dist_to_encoder().seq_encode([x])
        return float(self.seq_log_density(enc_x)[0])

    def seq_log_density(self, x: "TreeHiddenMarkovEncodedDataSequence") -> np.ndarray:
        """Return one marginal log density per encoded tree."""
        if not isinstance(x, TreeHiddenMarkovEncodedDataSequence):
            raise TypeError(
                "Requires TreeHiddenMarkovEncodedDataSequence for `seq_` calls."
            )

        if x.data[0]:
            (
                tz,
                (max_level, xln, xlnl, tlnz),
                (xbi, xp, xc, xl, txz, tp, tpz),
                enc_x,
                _len_enc,
            ) = x.data[1]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]
            num_trees = len(tz) - 1

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            level_state_prob(max_level + 1, num_states, a_mat, w, p_level)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_trees, dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.ones_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            numba_seq_log_density(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                xlnl,
                pr_obs,
                p_level,
                a_mat,
                pr_max0,
                betas,
                etas,
                ll_ret,
            )

            # if len_enc is not None:
            #     ret_len = np.zeros(num_trees, dtype=np.float64)
            # ll_ret += vec_bincount(len_enc[0],
            # self.len_dist.seq_log_density(len_enc[1]), ret_len)

            return ll_ret

        (
            cnt,
            tz,
            (xln, xlnl, xlni),
            (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _),
            enc_x,
            _len_enc,
        ) = x.data[1]

        num_states = self.num_states
        max_level = len(level_idx)
        a_mat = self.transitions
        w = self.w
        num_trees = len(tz) - 1

        betas = np.ones((cnt, num_states), dtype=np.float64)
        etas = np.zeros((len(xbi), num_states), dtype=np.float64)

        p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
        p_level[0, :] += w

        for level in range(1, max_level + 1):
            p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

        pr_obs = np.zeros((cnt, num_states), dtype=np.float64)
        ll_ret = np.zeros(num_trees, dtype=np.float64)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

        pr_max0 = pr_obs.max(axis=1)
        pr_obs -= pr_max0[:, None]
        np.exp(pr_obs, out=pr_obs)

        #  set the leaf nodes
        betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
        betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
        betas[xln, :] /= betas_sum

        ll_ret += np.bincount(
            xlni,
            weights=np.log(betas_sum.flatten()) + pr_max0[xln],
            minlength=num_trees,
        )

        #  upward pass on betas
        for level in range(len(level_idx) - 1, -1, -1):

            lidx = level_idx[level]
            _idxs, xbis, _xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

            #  Get etas
            temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
            temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
            temp = np.sum(a_mat.T * temp, axis=1)
            etas[xbis, :] += temp

            temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
            temp[1:, :] += np.log(etas[xbis, :])
            log_etas = np.cumsum(temp, axis=0)
            log_etas = log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]

            betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
            betas[p_nxt[level], :] *= p_level[level, :]
            betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

            betas[p_nxt[level], :] /= betas_sum

            ll_ret += np.bincount(
                i_nxt[level],
                weights=np.log(betas_sum.flatten()) + pr_max0[p_nxt[level]],
                minlength=num_trees,
            )

        # if len_enc is not None:
        #     ret_len = np.zeros(num_trees, dtype=np.float64)
        # ll_ret += vec_bincount(len_enc[0],
        # self.len_dist.seq_log_density(len_enc[1]), ret_len)

        return ll_ret

    def seq_posterior(
        self, x: "TreeHiddenMarkovEncodedDataSequence"
    ) -> List[np.ndarray]:
        """Return node-state posterior arrays of shape ``(n, K)`` per tree."""
        if not isinstance(x, TreeHiddenMarkovEncodedDataSequence):
            raise TypeError(
                "Requires TreeHiddenMarkovEncodedDataSequence for `seq_` calls."
            )

        if x.data[0]:
            (
                tz,
                (max_level, xln, xlnl, tlnz),
                (xbi, xp, xc, xl, txz, tp, tpz),
                enc_x,
                _,
            ) = x.data[1]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            level_state_prob(max_level + 1, num_states, a_mat, w, p_level)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.zeros_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            ### Need to do upward and downward, then read back the gammas
            numba_posteriors(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                xlnl,
                pr_obs,
                p_level,
                a_mat,
                betas,
                etas,
            )

            return [betas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

        (
            cnt,
            tz,
            (xln, xlnl, _xlni),
            (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, _i_nxt, _),
            enc_x,
            _len_enc,
        ) = x.data[1]

        num_states = self.num_states
        max_level = len(level_idx)
        a_mat = self.transitions
        w = self.w
        betas = np.ones((cnt, num_states), dtype=np.float64)
        etas = np.zeros((len(xbi), num_states), dtype=np.float64)

        p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
        p_level[0, :] += w

        for level in range(1, max_level + 1):
            p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

        pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

        pr_max0 = pr_obs.max(axis=1)
        pr_obs -= pr_max0[:, None]
        np.exp(pr_obs, out=pr_obs)

        #  set the leaf nodes
        betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
        betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
        betas[xln, :] /= betas_sum

        #  upward pass on betas
        for level in range(len(level_idx) - 1, -1, -1):
            lidx = level_idx[level]
            _idxs, xbis, _xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

            #  Get etas
            temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
            temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
            temp = np.sum(a_mat.T * temp, axis=1)
            etas[xbis, :] += temp

            temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
            temp[1:, :] += np.log(etas[xbis, :])
            log_etas = np.cumsum(temp, axis=0)
            log_etas = log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]

            betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
            betas[p_nxt[level], :] *= p_level[level, :]
            betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

            betas[p_nxt[level], :] /= betas_sum

        #  Return betas by observed sequence need tz
        return [betas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: Sequence[Tuple[D, T]]) -> np.ndarray:
        """Return the most likely state vector, with shape ``(n,)``."""
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_viterbi(enc_x)[0]

    def seq_viterbi(self, x: "TreeHiddenMarkovEncodedDataSequence") -> List[np.ndarray]:
        """Return one most-likely state vector of shape ``(n,)`` per tree."""
        if not isinstance(x, TreeHiddenMarkovEncodedDataSequence):
            raise TypeError(
                "Requires TreeHiddenMarkovEncodedDataSequence for `seq_` calls."
            )

        if x.data[0]:
            (
                tz,
                (max_level, xln, _xlnl, tlnz),
                (xbi, xp, xc, xl, txz, tp, tpz),
                enc_x,
                _,
            ) = x.data[1:]

            num_states = self.num_states
            log_w = self.log_w
            log_a_mat = self.log_transitions
            tot_cnt = tz[-1]

            log_pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            for i in range(num_states):
                log_pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            betas = np.ones_like(log_pr_obs, dtype=np.float64)
            etas = np.ones((len(xbi), num_states), dtype=np.float64)
            out = np.zeros(tot_cnt, dtype=np.int32)

            numba_viterbi(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                log_pr_obs,
                log_w,
                log_a_mat,
                betas,
                etas,
                out,
            )

            return [out[tz[i] : tz[i + 1]] for i in range(len(tz) - 1)]

        (
            cnt,
            tz,
            (xln, _xlnl, _xlni),
            (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, _i_nxt, rns),
            enc_x,
            _,
        ) = x.data[1:]

        num_states = self.num_states
        max_level = len(level_idx)
        log_a_mat = self.log_transitions
        log_w = self.log_w

        log_delta = np.ones((cnt, num_states), dtype=np.float64)
        log_eta = np.zeros((len(xbi), num_states), dtype=np.float64)
        state_tracker = np.zeros(cnt, dtype=np.int32)

        # Compute state likelihood vectors, and initialize the deltas for each state
        for i in range(num_states):
            log_delta[:, i] += self.topics[i].seq_log_density(enc_x)

        state_tracker[xln] += np.argmax(log_delta[xln, :], axis=1).flatten()

        #  upward pass on deltas
        for level in range(max_level - 1, -1, -1):
            lidx = level_idx[level]
            _idxs, xbis, _xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

            #  Get log_etas
            log_eta[xbis, :] += np.max(
                np.reshape(log_delta[xcs, :], (-1, 1, num_states)) + log_a_mat,
                axis=2,
            )
            temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
            temp[1:, :] += np.cumsum(log_eta[xbis, :], axis=0)
            temp = temp[eta_p[level][1:], :] - temp[eta_p[level][:-1], :]
            log_delta[p_nxt[level], :] += temp
            state_tracker[p_nxt[level]] += np.argmax(
                log_delta[p_nxt[level], :], axis=1, keepdims=False
            )

        #  Set the init for leaf nodes
        log_delta[rns, :] += log_w
        state_tracker[rns] += np.argmax(log_delta[rns, :], axis=1).flatten()

        return [state_tracker[tz[i] : tz[i + 1]] for i in range(len(tz) - 1)]

    def sampler(self, seed: Optional[int] = None) -> "TreeHiddenMarkovSampler":
        """Create a tree sampler, requiring a non-null child-count distribution."""
        if isinstance(self.len_dist, NullDistribution):
            raise RuntimeError(
                "TreeHiddenMarkovSampler requires len_dist with support on "
                "non-negative integers"
            )
        return TreeHiddenMarkovSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "TreeHiddenMarkovEstimator":
        """Create an estimator for emissions, root weights, and transitions."""
        len_est = (
            None
            if self.len_dist is None
            else self.len_dist.estimator(pseudo_count=pseudo_count)
        )
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return TreeHiddenMarkovEstimator(
            comp_ests,
            pseudo_count=(pseudo_count, pseudo_count),
            len_estimator=len_est,
            name=self.name,
        )

    def dist_to_encoder(self) -> "TreeHiddenMarkovDataEncoder":
        """Return the encoder matching this distribution's tree representation."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return TreeHiddenMarkovDataEncoder(
            emission_encoder=emission_encoder,
            len_encoder=len_encoder,
            use_numba=self.use_numba,
        )


class TreeHiddenMarkovSampler(DistributionSampler):
    """Sample rooted trees, latent states, and node emissions."""

    def __init__(
        self, dist: "TreeHiddenMarkovModelDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize state, emission, and child-count samplers."""
        super().__init__(dist, seed)
        self.num_states = dist.num_states
        self.obs_samplers = [
            topic.sampler(seed=self.rng.randint(maxrandint)) for topic in dist.topics
        ]
        self.init_w = dist.w
        self.transitions = dist.transitions

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(
                seed=self.rng.randint(0, maxrandint)
            )
        else:
            self.len_sampler = None

    def sample_state(
        self, given_state: int, size: Optional[int] = None
    ) -> Union[int, np.ndarray]:
        """Sample one or more child states conditioned on a parent state."""
        return self.rng.choice(
            self.num_states, p=self.transitions[given_state, :], replace=True, size=size
        )

    def sample_tree(self, size: Optional[int] = None) -> Union[
        List[Tuple[D, Any]],
        List[List[Tuple[D, Any]]],
    ]:
        """Sample one tree, or a list of independently sampled trees."""
        if size is None:
            if self.len_sampler is None:
                raise RuntimeError("Length sampler is required for sample_tree().")

            seq: List[Tuple[D, Any]] = []
            xi = 0
            zi = int(self.rng.choice(self.num_states, p=self.init_w))
            ni = int(self.len_sampler.sample())
            nodes = [(xi, zi, ni)]
            y0 = self.obs_samplers[zi].sample()

            seq.append(((0, -1), y0))
            iter_cond = ni > 0

            cnt = 1
            lvl_cnt = 0

            while iter_cond and lvl_cnt < self.dist.terminal_level:
                nodes_next = []
                for node in nodes:
                    xi, zi, ni = node

                    zj = np.asarray(
                        self.sample_state(given_state=zi, size=ni), dtype=np.int32
                    )
                    nj = np.asarray(self.len_sampler.sample(size=ni), dtype=np.int32)

                    for j in range(ni):
                        if nj[j] > 0:
                            nodes_next.append((cnt + j, int(zj[j]), int(nj[j])))
                        seq.append(
                            ((cnt + j, xi), self.obs_samplers[int(zj[j])].sample())
                        )
                    cnt += ni
                if len(nodes_next) == 0:
                    iter_cond = False
                else:
                    nodes = list(nodes_next)

                lvl_cnt += 1

            return seq

        return cast(
            List[List[Tuple[D, Any]]],
            [self.sample_tree() for xx in range(size)],
        )

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[D, Any]], List[List[Tuple[D, Any]]]]:
        """Sample trees using the configured child-count distribution."""
        if self.len_sampler is not None:
            return self.sample_tree(size=size)
        raise RuntimeError(
            "TreeHiddenMarkovSampler requires either a length distribution for "
            "number of children."
        )


class TreeHiddenMarkovAccumulator(  # pylint: disable=too-many-instance-attributes
    SequenceEncodableStatisticAccumulator
):
    """Accumulate root, transition, emission, and child-count statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: Optional[
            SequenceEncodableStatisticAccumulator
        ] = NullAccumulator(),
        keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        name: Optional[str] = None,
        use_numba: bool = True,
    ) -> None:
        """Initialize sufficient-statistic accumulators for a tree HMM."""
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64
        )
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator = (
            len_accumulator if len_accumulator is not None else NullAccumulator()
        )

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.name = name
        self.use_numba = use_numba

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: Optional[RandomState] = None
        self._acc_rng: Optional[List[RandomState]] = None
        self._idx_rng: Optional[RandomState] = None
        self._w_rng: Optional[RandomState] = None

    def update(
        self,
        x: Sequence[Tuple[D, T]],
        weight: float,
        estimate: TreeHiddenMarkovModelDistribution,
    ) -> None:
        """Accumulate weighted sufficient statistics for one raw tree."""
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Create independent random streams for statistic initialization."""
        rng_seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._acc_rng = [
            RandomState(seed=rng_seeds[2 + i]) for i in range(self.num_states)
        ]
        self._w_rng = RandomState(seed=rng.randint(2**30))
        self._init_rng = True

    def initialize(
        self, x: Sequence[Tuple[D, T]], weight: float, rng: RandomState
    ) -> None:
        """Initialize statistics for one tree with random state assignments."""
        if not self._init_rng:
            self._rng_initialize(rng)

        enc_x = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc_x, weights=np.asarray([weight]), rng=rng)

    def seq_initialize(
        self,
        x: "TreeHiddenMarkovEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize statistics for a batch of encoded trees."""
        if not self._init_rng:
            self._rng_initialize(rng)
        assert self._idx_rng is not None
        assert self._acc_rng is not None
        assert self._len_rng is not None

        if x.data[0]:

            tz, _, (_xbi, xp, xc, _xl, txz, tp, tpz), enc_x, len_enc = x.data[1]

            states = self._idx_rng.choice(self.num_states, replace=True, size=tz[-1])

            numba_initialize(
                tz,
                txz,
                tp,
                tpz,
                xp,
                xc,
                states,
                weights,
                self.init_counts,
                self.state_counts,
                self.trans_counts,
            )

            idx = len_enc[0]
            nz_idx = np.unique(idx)
            weights_nz = weights[nz_idx]

            for i in range(self.num_states):
                w = weights_nz[idx].copy()
                w[states == i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(
                    len_enc[1], weights[len_enc[0]], self._len_rng
                )

        else:
            (
                cnt,
                tz,
                _,
                (idx, _xbi, xp, xc, level_idx, _p_nxt, _eta_p, i_nxt, rns),
                enc_x,
                len_enc,
            ) = x.data[1]

            num_states = self.num_states
            states = self._idx_rng.choice(self.num_states, replace=True, size=cnt)

            #  Get root node states
            root_states = np.bincount(
                states[rns], weights=weights[i_nxt[0]], minlength=num_states
            )
            self.init_counts += root_states
            self.state_counts += root_states

            # count state transitions by the levels
            ns2 = num_states**2
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xps, xcs = idx[lidx], xp[lidx], xc[lidx]

                bin_weights = []
                bin_weights.extend([weights[kk] for kk in idxs])

                arr = np.asarray([states[xps], states[xcs]], dtype=np.int32)
                multi_idx = np.ravel_multi_index(arr, (num_states, num_states))

                trans_cnts = np.bincount(multi_idx, weights=bin_weights, minlength=ns2)
                self.trans_counts += np.reshape(trans_cnts, (num_states, num_states))

            obs_idx = len_enc[0]
            nz_idx = np.unique(obs_idx)
            weights_nz = weights[nz_idx]

            for i in range(self.num_states):
                w = weights_nz[obs_idx].copy()
                w[states == i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(
                    len_enc[1], weights[len_enc[0]], self._len_rng
                )

    def seq_update(
        self,
        x: "TreeHiddenMarkovEncodedDataSequence",
        weights: np.ndarray,
        estimate: TreeHiddenMarkovModelDistribution,
    ) -> None:
        """Accumulate weighted tree forward-backward sufficient statistics."""
        if x.data[0]:
            (
                tz,
                (max_level, xln, xlnl, tlnz),
                (xbi, xp, xc, xl, txz, tp, tpz),
                enc_x,
                len_enc,
            ) = x.data[1]

            tot_cnt = tz[-1]
            num_states = estimate.num_states
            w = estimate.w
            a_mat = estimate.transitions
            num_trees = len(tz) - 1

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)

            level_state_prob(max_level + 1, num_states, a_mat, w, p_level)
            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            xi_acc = np.zeros((num_trees, num_states, num_states), dtype=np.float64)
            pi_acc = np.zeros((num_trees, num_states), dtype=np.float64)

            numba_baum_welch(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                xlnl,
                pr_obs,
                p_level,
                a_mat,
                weights,
                betas,
                etas,
                alphas,
                xi_acc,
                pi_acc,
            )

            self.init_counts += pi_acc.sum(axis=0)
            self.trans_counts += xi_acc.sum(axis=0)

            for i in range(num_states):
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            if len_enc is not None:
                self.len_accumulator.seq_update(
                    len_enc[1], weights[len_enc[0]], estimate.len_dist
                )

        else:
            ## numpy calculation from encoding
            (
                cnt,
                tz,
                (xln, xlnl, _xlni),
                (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, _i_nxt, rns),
                enc_x,
                len_enc,
            ) = x.data[1]

            num_states = estimate.num_states
            max_level = len(level_idx)
            a_mat = estimate.transitions
            w = estimate.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((cnt, num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
                temp[1:, :] += np.log(etas[xbis, :])
                log_etas = np.cumsum(temp, axis=0)
                log_etas = (
                    log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]
                )

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

            ## alpha (upward pass) set the root nodes
            alphas[rns, :] += betas[rns, :]

            for level, level_idx_level in enumerate(level_idx):
                lidx = level_idx_level
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]
                weights_loc = np.reshape(weights[idxs], (-1, 1, 1))

                xi0 = (
                    np.reshape(alphas[xps, :] / etas[xbis, :], (-1, num_states, 1))
                    * a_mat
                )
                xi1 = np.reshape(
                    betas[xcs, :] / p_level[level + 1, :], (-1, 1, num_states)
                )
                xi_loc = xi0 * xi1

                xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(
                    axis=2, keepdims=True
                )
                xi_loc_sum[xi_loc_sum == 0] = 1.0

                temp = xi_loc.sum(axis=1)
                temp_sum = temp.sum(axis=1, keepdims=True)
                temp_sum[temp_sum == 0] = 1.0
                temp /= temp_sum

                xi_loc *= weights_loc / xi_loc_sum

                self.trans_counts += xi_loc.sum(axis=0)
                alphas[xcs, :] += temp

            self.init_counts += np.sum(alphas[rns, :], axis=0)
            self.state_counts += alphas.sum(axis=0)

            for i in range(num_states):
                alphas[:, i] *= weights[len_enc[0]]
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            if len_enc is not None:
                self.len_accumulator.seq_update(
                    len_enc[1], weights[len_enc[0]], estimate.len_dist
                )

    def combine(
        self,
        suff_stat: Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]
        ],
    ) -> "TreeHiddenMarkovAccumulator":
        """Combine a sufficient-statistic tuple into this accumulator."""
        (
            _num_states,
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
        """Return state counts and component statistics for estimation."""
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
        x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]],
    ) -> "TreeHiddenMarkovAccumulator":
        """Restore this accumulator from a sufficient-statistic tuple."""
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
        """Merge keyed sufficient statistics into ``stats_dict``."""
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
                for i, acc_i in enumerate(acc):
                    acc_i = acc_i.combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed sufficient statistics from ``stats_dict``."""
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

    def acc_to_encoder(self) -> "TreeHiddenMarkovDataEncoder":
        """Return the tree encoder corresponding to these accumulators."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return TreeHiddenMarkovDataEncoder(
            emission_encoder=emission_encoder,
            len_encoder=len_encoder,
            use_numba=self.use_numba,
        )


class TreeHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    """Build statistic accumulators for tree HMM estimation."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
        use_numba: bool = True,
    ) -> None:
        """Initialize factories for emissions and child-count statistics."""
        self.factories = factories
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory
        self.name = name
        self.use_numba = use_numba

    def make(self) -> "TreeHiddenMarkovAccumulator":
        """Create a fresh tree HMM statistic accumulator."""
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return TreeHiddenMarkovAccumulator(
            [self.factories[i].make() for i in range(len(self.factories))],
            len_accumulator=len_acc,
            keys=self.keys,
            name=self.name,
            use_numba=self.use_numba,
        )


class TreeHiddenMarkovEstimator(ParameterEstimator):
    """Estimate tree HMM parameters from accumulated sufficient statistics."""

    def __init__(
        self,
        estimators: List[ParameterEstimator],
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        use_numba: bool = True,
    ) -> None:
        """Initialize component estimators and optional pseudo-counts."""
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.name = name
        self.use_numba = use_numba

    def accumulator_factory(self) -> TreeHiddenMarkovAccumulatorFactory:
        """Return a factory for this estimator's sufficient statistics."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return TreeHiddenMarkovAccumulatorFactory(
            factories=est_factories,
            len_factory=len_factory,
            keys=self.keys,
            name=self.name,
            use_numba=self.use_numba,
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]
        ],
    ) -> "TreeHiddenMarkovModelDistribution":
        """Estimate emissions, root weights, transitions, and child counts."""
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = (
            suff_stat
        )

        len_dist = self.len_estimator.estimate(nobs, len_ss)
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

        return TreeHiddenMarkovModelDistribution(
            topics=topics,
            w=w,
            transitions=transitions,
            len_dist=len_dist,
            name=self.name,
            use_numba=self.use_numba,
        )


class TreeHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Encode tree batches as flattened nodes and parent-child index arrays.

    The raw tree contract remains ``(node_index, parent_index), emission``.
    The encoded data flattens all node emissions, retains cumulative tree
    boundaries, and stores parent, child, level, and leaf indices.  In either
    encoder layout, node-level arrays have length equal to the batch's total
    node count and transition-pair arrays have length equal to its total edge
    count.
    """

    def __init__(
        self,
        emission_encoder: DataSequenceEncoder,
        len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder(),
        use_numba: bool = True,
    ) -> None:
        """Initialize emission and child-count encoders for tree batches."""
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.use_numba = use_numba

    def __str__(self) -> str:
        """Return an evaluable representation of the encoder."""
        s1 = repr(self.emission_encoder)
        s2 = repr(self.len_encoder)
        s3 = repr(self.use_numba)
        return (
            f"TreeHiddenMarkovDataEncoder(emission_encoder={s1}, len_encoder={s2}, "
            f"use_numba={s3})"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` has the same child-count encoder."""
        return (
            isinstance(other, TreeHiddenMarkovDataEncoder)
            and self.len_encoder == other.len_encoder
        )

    def _seq_encode(self, x: Sequence[Sequence[Tuple[D, T]]]) -> E1:
        """Encode a batch with the NumPy-oriented tree index layout."""
        xs: List[T] = []  # flattened values of nodes in order encoded
        obs_idx: List[int] = []  #  tree seq idx for observed flattened nodes
        idx: List[int] = []  # idx for node observation by tree in seq used in betas
        tz: List[int] = [0]  #  Track entries in beta by observation.
        #  Encodings for the beta pass
        xln: List[int] = []  # leaf nodes
        xlnl: List[int] = []  # levels for the leaf nodes
        xlni: List[int] = []
        root_id: List[int] = []

        xbi: List[int] = []  # Use this to track beta_j(p(u), u)
        xp: List[int] = []  # parents, repeated for each child
        xl: List[int] = []  # level of xc below
        xc: List[int] = []  # children of xp

        nc: List[int] = []  # number of children for a given node.

        cnt = 0
        eta_cnt = 0
        for i, xx in enumerate(x):

            n = len(xx)
            tz.append(n)
            if n > 0:
                root_id.append(i)

            xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
            xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

            p_sort = np.argsort(xp0)

            xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
            ## relabel entries to be 0,1,2,3,....,n-1
            xi0 = xi0[p_sort] + cnt
            xp0 = xp0[p_sort]

            xs.extend([xx[i][1] for i in p_sort])

            u0, u1 = np.unique(xp0[1:], return_counts=True)

            #  beta parent/child combos
            if len(u1) > 0:
                for j, u1_j in enumerate(u1):
                    xp.extend([u0[j] + cnt] * u1_j)
                    xc.extend(cnt + xc0[np.flatnonzero(xp0[1:] == u0[j])])

            if len(xp0) > 1:
                xbi.extend([kk + eta_cnt for kk in range(len(xp0) - 1)])
                eta_cnt += len(xp0) - 1

                xl_temp = find_level(xp0)
                xl.extend(xl_temp)
                xln_temp = np.delete(np.arange(n), u0)
                xlnl.extend([xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp])
                xlni.extend([i] * len(xln_temp))
                xln.extend(xln_temp + cnt)
                idx.extend([i] * len(xl_temp))

            #  Length distribution
            nc_temp = np.zeros(n, dtype=np.int32)
            nc_temp[u0] = u1
            nc.extend(nc_temp)
            obs_idx.extend([i] * n)

            cnt += n

        idx_arr = np.asarray(idx, dtype=np.int32)
        xbi_arr = np.asarray(xbi, dtype=np.int32)
        xp_arr = np.asarray(xp, dtype=np.int32)
        xc_arr = np.asarray(xc, dtype=np.int32)
        xl_arr = np.asarray(xl, dtype=np.int32)
        xln_arr = np.asarray(xln, dtype=np.int32)
        xlnl_arr = np.asarray(xlnl, dtype=np.int32)
        xlni_arr = np.asarray(xlni, dtype=np.int32)
        root_idx = np.asarray(root_id, dtype=np.int32)

        level_idx: List[np.ndarray] = []
        eta_p: List[np.ndarray] = []
        p_nxt: List[np.ndarray] = []
        i_nxt = [root_idx]

        for level in range(1, int(np.max(xl_arr)) + 1):

            level_idx.append(np.flatnonzero(xl_arr == level))
            unique_counts = np.unique(xp_arr[level_idx[-1]], return_counts=True)
            u0 = unique_counts[0]
            u1 = unique_counts[1]
            eta_p.append(np.cumsum(np.append([0], u1)))
            p_nxt.append(u0)
            i_nxt.append(idx_arr[level_idx[-1]])

        rns = np.unique(xp_arr[level_idx[0]])  # root nodes

        enc_x = self.emission_encoder.seq_encode(xs)
        len_enc = self.len_encoder.seq_encode(nc)

        tz_arr = np.cumsum(tz).astype(np.int32)
        obs_idx_arr = np.asarray(obs_idx, dtype=np.int32)

        if len_enc is not None:
            return False, (
                cnt,
                tz_arr,
                (xln_arr, xlnl_arr, xlni_arr),
                (idx_arr, xbi_arr, xp_arr, xc_arr, level_idx, p_nxt, eta_p, i_nxt, rns),
                enc_x,
                (obs_idx_arr, len_enc),
            )
        return False, (
            cnt,
            tz_arr,
            (xln_arr, xlnl_arr, xlni_arr),
            (idx_arr, xbi_arr, xp_arr, xc_arr, level_idx, p_nxt, eta_p, i_nxt, rns),
            enc_x,
            None,
        )

    def seq_encode(
        self, x: Sequence[Sequence[Tuple[D, T]]]
    ) -> "TreeHiddenMarkovEncodedDataSequence":
        """Encode a batch of rooted node sequences for tree inference."""
        if self.use_numba:
            xs: List[T] = []  # flattened values of nodes in order encoded
            idx: List[int] = []  # idx corresponding to weight
            tz = [0]  # slice entries for a given observed tree

            #  Encodings for the beta pass
            xln: List[int] = []  # leaf nodes
            xlnl: List[int] = []  # levels for the leaf nodes
            tlnz = [0]  # slice leaf nodes for given tree observation
            xbi: List[int] = []  # Use this to track beta_j(p(u), u)
            xp: List[int] = []  # parents, repeated for each child
            xl: List[int] = []  # level of xc below
            xc: List[int] = []  # children of xp
            txz = [0]  # slice xp, xc, and xl for observed tree
            tp: List[int] = (
                []
            )  # partition couples of (p, c) for all of a parents children.
            tpz = [0]  # slice tp for an observed tree

            nc: List[int] = []  # number of children for a given node.

            for i, xx in enumerate(x):

                n = len(xx)

                xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
                xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

                p_sort = np.argsort(xp0)

                xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
                #  relabel entries to be 0,1,2,3,....,n-1
                xi0 = xi0[p_sort]
                xp0 = xp0[p_sort]
                xs.extend([xx[i][1] for i in p_sort])

                u0, u1 = np.unique(xp0[1:], return_counts=True)

                #  beta parent/child combos
                if len(u1) > 0:
                    for j, u1_j in enumerate(u1):
                        xp.extend([u0[j]] * u1_j)
                        xc.extend(xc0[np.flatnonzero(xp0[1:] == u0[j])])

                    txz.append(int(np.sum(u1)))
                    tp.extend([int(v) for v in np.cumsum([0] + list(u1))])
                    tpz.append(len(u1) + 1)

                else:
                    txz.append(0)
                    tp.append(0)
                    tpz.append(1)

                if len(xp0) > 1:
                    xbi.extend(list(range(len(xp0) - 1)))

                    xl_temp = find_level(xp0)
                    xl.extend(xl_temp)
                    xln_temp = list(np.delete(np.arange(n), u0))
                    xlnl.extend(
                        [xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp]
                    )
                    xln.extend(xln_temp)

                    tlnz.append(len(xln_temp))
                else:
                    tlnz.append(0)

                tz.append(n)

                #  Length distribution
                idx.extend([i] * n)

                nc_temp = np.zeros(n, dtype=np.int32)
                nc_temp[u0] = u1
                nc.extend(nc_temp)

            tz_arr = np.cumsum(tz).astype(np.int32)

            xln_arr = np.asarray(xln, dtype=np.int32)
            xlnl_arr = np.asarray(xlnl, dtype=np.int32)
            tlnz_arr = np.cumsum(tlnz).astype(np.int32)

            xbi_arr = np.asarray(xbi, dtype=np.int32)
            xp_arr = np.asarray(xp, dtype=np.int32)
            xc_arr = np.asarray(xc, dtype=np.int32)
            xl_arr = np.asarray(xl, dtype=np.int32)
            txz_arr = np.cumsum(txz).astype(np.int32)
            tp_arr = np.asarray(tp, dtype=np.int32)
            tpz_arr = np.cumsum(tpz).astype(np.int32)

            enc_x = self.emission_encoder.seq_encode(xs)
            len_enc = self.len_encoder.seq_encode(nc)

            # if len_enc is not None:
            len_enc_tuple = (np.asarray(idx, np.int32), len_enc)

            rv_enc = (
                tz_arr,
                (int(np.max(xln_arr)), xln_arr, xlnl_arr, tlnz_arr),
                (xbi_arr, xp_arr, xc_arr, xl_arr, txz_arr, tp_arr, tpz_arr),
                enc_x,
                len_enc_tuple,
            )

            return TreeHiddenMarkovEncodedDataSequence(data=(True, rv_enc))

        return TreeHiddenMarkovEncodedDataSequence(data=self._seq_encode(x))


class TreeHiddenMarkovEncodedDataSequence(EncodedDataSequence):
    """Hold a NumPy- or Numba-oriented encoded batch of rooted trees."""

    def __init__(self, data: Union[E0, E1]) -> None:
        """Initialize the encoded tree batch data container."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded tree data."""
        return f"TreeHiddenMarkovEncodedDataSequence(data=f{self.data})"


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], "
    "float64[:,:], float64[:])",
    fastmath=True,
    parallel=True,
)
def numba_seq_log_density(  # pylint: disable=too-many-positional-arguments
    num_states: int,
    tz: np.ndarray,
    txz: np.ndarray,
    tp: np.ndarray,
    tpz: np.ndarray,
    tlnz: np.ndarray,
    xp: np.ndarray,
    xc: np.ndarray,
    xl: np.ndarray,
    xbi: np.ndarray,
    xln: np.ndarray,
    xlnl: np.ndarray,
    pr_obs: np.ndarray,
    p_level: np.ndarray,
    tr_mat: np.ndarray,
    pr_max0: np.ndarray,
    betas: np.ndarray,
    etas: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute scaled marginal log densities for encoded tree batches."""
    for n in numba.prange(len(tz) - 1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            out[n] = 0
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        if i0 == i1:
            #  Only root node in tree
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]
                beta_sum += temp
            out[n] = math.log(beta_sum) + pr_max0[s0]

        ll_sum = 0.0
        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]
        b_max = pr_max0[s0:s1]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k, xlns_k in enumerate(xlns):
            leaf_node = xlns_k
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] *= temp
                beta_sum += temp

            ll_sum += math.log(beta_sum) + b_max[leaf_node]

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] *= b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] += eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            ll_sum += math.log(beta_sum) + b_max[p]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        out[n] = ll_sum


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], "
    "float64[:,:], float64[:,:], "
    "float64[:,:, :], float64[:,:])",
    parallel=True,
)
def numba_baum_welch(  # pylint: disable=too-many-positional-arguments
    num_states: int,
    tz: np.ndarray,
    txz: np.ndarray,
    tp: np.ndarray,
    tpz: np.ndarray,
    tlnz: np.ndarray,
    xp: np.ndarray,
    xc: np.ndarray,
    xl: np.ndarray,
    xbi: np.ndarray,
    xln: np.ndarray,
    xlnl: np.ndarray,
    pr_obs: np.ndarray,
    p_level: np.ndarray,
    tr_mat: np.ndarray,
    weights: np.ndarray,
    betas: np.ndarray,
    etas: np.ndarray,
    alphas: np.ndarray,
    xi_acc: np.ndarray,
    pi_acc: np.ndarray,
) -> None:
    """Accumulate tree forward-backward state and transition statistics."""
    for n in numba.prange(len(tz) - 1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]
        weight_loc = weights[n]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            # Only one node with no children, need to handle this. No transition updates
            # just pi_acc
            alpha_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]

                alphas[s0, i] = temp * weight_loc
                alpha_sum += temp

            for i in range(num_states):
                alphas[s0, i] /= alpha_sum
                pi_acc[n, i] += alphas[s0, i]

            continue

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k, xlns_k in enumerate(xlns):
            leaf_node = xlns_k
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        ### do the alpha pass
        alpha_mat = alphas[s0:s1, :]
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        #  set the root
        for i in range(num_states):
            alpha_mat[0, i] += beta_mat[0, i] * weight_loc

        for nn in range(0, len(tps) - 1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, _level = xps[t0], xls[t0]

            for k in range(t0, t1):
                c, eta_idx = xcs[k], xbis[k]
                xi_buff_sum = 0

                gamma_sum = 0
                for i in range(num_states):
                    alpha_sum = 0
                    for j in range(num_states):
                        temp = tr_mat[j, i] * alpha_mat[p, j] / eta_mat[eta_idx, j]
                        alpha_sum += temp

                        temp *= beta_mat[c, i]
                        temp /= p_level[level, i]

                        xi_buff_sum += temp
                        xi_buff[j, i] = temp

                    alpha_sum *= beta_mat[c, i]
                    alpha_sum /= p_level[level, i]

                    alpha_mat[c, i] += alpha_sum
                    gamma_sum += alpha_sum

                if gamma_sum > 0:
                    gamma_sum = weight_loc / gamma_sum
                if xi_buff_sum > 0:
                    xi_buff_sum = weight_loc / xi_buff_sum
                for i in range(num_states):
                    alpha_mat[c, i] *= gamma_sum
                    for j in range(num_states):
                        xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_mat[0, i]


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:,:], float64[:,:], float64[:,:])",
    fastmath=True,
    parallel=True,
)
def numba_posteriors(  # pylint: disable=too-many-positional-arguments
    num_states: int,
    tz: np.ndarray,
    txz: np.ndarray,
    tp: np.ndarray,
    tpz: np.ndarray,
    tlnz: np.ndarray,
    xp: np.ndarray,
    xc: np.ndarray,
    xl: np.ndarray,
    xbi: np.ndarray,
    xln: np.ndarray,
    xlnl: np.ndarray,
    pr_obs: np.ndarray,
    p_level: np.ndarray,
    tr_mat: np.ndarray,
    betas: np.ndarray,
    etas: np.ndarray,
) -> None:
    """Compute posterior node-state probabilities for encoded trees."""
    for n in numba.prange(len(tz) - 1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            # Only one node with no children, need to handle this. No transition updates
            # just pi_acc
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]

                betas[s0, i] += temp
                beta_sum += temp

            for i in range(num_states):
                betas[s0, i] /= beta_sum

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k, xlns_k in enumerate(xlns):
            leaf_node = xlns_k
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum


@numba.jit(
    "void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int64[:], "
    "float64[:], float64[:], "
    "float64[:], float64[:,:])",
    parallel=True,
    nopython=True,
)
def numba_initialize(  # pylint: disable=too-many-positional-arguments
    tz: np.ndarray,
    txz: np.ndarray,
    tp: np.ndarray,
    tpz: np.ndarray,
    xp: np.ndarray,
    xc: np.ndarray,
    states: np.ndarray,
    weights: np.ndarray,
    init_counts: np.ndarray,
    state_counts: np.ndarray,
    trans_counts: np.ndarray,
) -> None:
    """Accumulate statistics from randomly initialized tree states."""
    for n in numba.prange(len(tz) - 1):
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        ss = states[s0:s1]
        init_counts[ss[0]] += weight_loc
        state_counts[ss[0]] += weight_loc

        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            continue

        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 1):
            j0, j1 = tps[nn], tps[nn + 1]
            p = ss[xps[j0]]
            for k in range(j0, j1):
                c = ss[xcs[k]]
                trans_counts[p, c] += weight_loc
                state_counts[c] += weight_loc


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], "
    "int32[:])",
    parallel=True,
)
# pylint: disable-next=too-many-positional-arguments
def numba_viterbi(
    num_states: int,
    tz: np.ndarray,
    txz: np.ndarray,
    tp: np.ndarray,
    tpz: np.ndarray,
    tlnz: np.ndarray,
    xp: np.ndarray,
    xc: np.ndarray,
    xl: np.ndarray,
    xbi: np.ndarray,
    xln: np.ndarray,
    log_pr_obs: np.ndarray,
    log_init_p: np.ndarray,
    log_tr_mat: np.ndarray,
    betas: np.ndarray,
    etas: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute most-likely node states for encoded tree batches."""
    # Keep the numba kernel in a single function to preserve compilation behavior.
    # pylint: disable=too-many-nested-blocks
    for n in numba.prange(len(tz) - 1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        outs = out[s0:s1]

        if i0 == i1:
            # Only one node with no children, need to handle this. No transition updates
            # just pi_acc
            beta_max = None
            beta_max_i = 0
            for i in range(num_states):
                temp = log_pr_obs[s0, i] + log_init_p[i]
                if beta_max is None:
                    beta_max = temp
                    beta_max_i = i
                else:
                    if beta_max < temp:
                        beta_max = temp
                        beta_max_i = i

            outs[0] = beta_max_i

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        log_b = log_pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]

        for k, xlns_k in enumerate(xlns):
            leaf_node = xlns_k
            temp = log_b[leaf_node, 0]
            beta_mat[leaf_node, 0] += temp
            max_leaf_v = temp
            max_leaf_i = 0
            for i in range(1, num_states):
                temp = log_b[leaf_node, i]
                beta_mat[leaf_node, i] += temp

                if max_leaf_v < temp:
                    max_leaf_v = temp
                    max_leaf_i = i

            outs[leaf_node] = max_leaf_i

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, _level = xps[t0], xls[t0]
            beta_max_v = -np.inf
            beta_max_i = 0
            #  Get eta(p, u)_i and sum then get beta_i(p)
            for i in range(0, num_states):

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_max = beta_mat[c, 0] + log_tr_mat[i, 0]

                    for j in range(1, num_states):
                        temp = beta_mat[c, j] + log_tr_mat[i, j]
                        eta_max = max(eta_max, temp)

                    eta_mat[eta_idx, i] += eta_max
                    beta_mat[p, i] += log_b[p, i]
                    if beta_max_v < beta_mat[p, i]:
                        beta_max_v = beta_mat[p, i]
                        beta_max_i = i

            outs[p] = beta_max_i


@numba.njit("float64[:](int32[:], float64[:], float64[:])", parallel=True)
def vec_bincount(idx: np.ndarray, ll: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Add values into ``out`` at parallel integer-indexed positions."""
    for i in numba.prange(len(idx)):
        out[idx[i]] += ll[i]
    return out


@numba.njit("void(int32, int32, float64[:, :], float64[:], float64[:, :])")
def level_state_prob(
    levels: int,
    num_states: int,
    tr_mat: np.ndarray,
    init_prob: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill per-level marginal state probabilities with shape ``(L, K)``."""
    for i in range(num_states):
        out[0, i] = init_prob[i]

    for k in range(1, levels):
        for i in range(num_states):
            for j in range(num_states):
                out[k, i] += out[k - 1, i] * tr_mat[i, j]
