"""Tests for num_params() on Markov-chain and HMM distributions without a harness."""

import numpy as np
from scipy.sparse import csr_matrix

from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    IntegerHiddenMarkovModelDistribution,
    IntegerMarkovChainDistribution,
    LookbackHiddenMarkovDistribution,
    MarkovChainDistribution,
    SparseMarkovAssociationDistribution,
)
from dmx.stats.tree_hmm import TreeHiddenMarkovModelDistribution


def _hmm(keys: tuple) -> HiddenMarkovModelDistribution:
    """Return a 2-state HMM with Gaussian emissions and the given keys."""
    return HiddenMarkovModelDistribution(
        topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
        w=[0.5, 0.5],
        transitions=[[0.9, 0.1], [0.2, 0.8]],
        keys=keys,
    )


def _int_hmm(keys: tuple) -> IntegerHiddenMarkovModelDistribution:
    """Return a 2-state HMM over 4 integer words with the given keys."""
    pmat = np.full((4, 2), 0.25)
    return IntegerHiddenMarkovModelDistribution(
        pmat=pmat, w=[0.5, 0.5], transitions=[[0.9, 0.1], [0.2, 0.8]], keys=keys
    )


def test_hmm_with_topic_weights() -> None:
    # 2 states over 3 categorical topics (3 values each), no length:
    # init 1 + transitions 2*1 + topics 3*2 + taus 2*(3-1) = 13.
    topics = [CategoricalDistribution({"a": 0.2, "b": 0.3, "c": 0.5}) for _ in range(3)]
    dist = HiddenMarkovModelDistribution(
        topics=topics,
        w=[0.5, 0.5],
        transitions=[[0.5, 0.5], [0.5, 0.5]],
        taus=[[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
    )
    assert dist.num_params() == 13


def test_hmm_keyed_sharing() -> None:
    # One HMM: init 1 + transitions 2 + emissions 2*2 = 7.
    assert _hmm((None, None, None)).num_params() == 7
    # Two HMMs sharing emissions count them once: 7 + (1 + 2) = 10.
    shared_states = CompositeDistribution(
        [_hmm((None, None, "states")), _hmm((None, None, "states"))]
    )
    assert shared_states.num_params() == 10
    # Sharing initial and transition blocks: 7 + 4 = 11.
    shared_dynamics = CompositeDistribution(
        [_hmm(("init", "trans", None)), _hmm(("init", "trans", None))]
    )
    assert shared_dynamics.num_params() == 11
    # Everything shared: 7.
    all_shared = CompositeDistribution(
        [_hmm(("init", "trans", "states")), _hmm(("init", "trans", "states"))]
    )
    assert all_shared.num_params() == 7


def test_int_hidden_markov() -> None:
    # 2 states, 4 words: init 1 + transitions 2*1 + emissions 2*3 = 9.
    assert _int_hmm((None, None, None)).num_params() == 9
    # With a 3-value integer length distribution: 9 + 2 = 11.
    dist = IntegerHiddenMarkovModelDistribution(
        pmat=np.full((4, 2), 0.25),
        w=[0.5, 0.5],
        transitions=[[0.9, 0.1], [0.2, 0.8]],
        len_dist=IntegerCategoricalDistribution(1, [0.2, 0.3, 0.5]),
    )
    assert dist.num_params() == 11
    # Sharing transitions and emissions across two models: 9 + 1 = 10.
    shared = CompositeDistribution(
        [_int_hmm((None, "trans", "emit")), _int_hmm((None, "trans", "emit"))]
    )
    assert shared.num_params() == 10


def test_lookback_hmm() -> None:
    # 2 states, Gaussian window emissions, one Gaussian initial block per state,
    # 2-value length: init 1 + transitions 2 + topics 2*2 + init blocks 2*2 + 1 = 12.
    dist = LookbackHiddenMarkovDistribution(
        topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)],
        w=np.asarray([0.5, 0.5]),
        transitions=[[0.9, 0.1], [0.2, 0.8]],
        lag=1,
        init_dist=[GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)],
        len_dist=CategoricalDistribution({2: 0.5, 3: 0.5}),
    )
    assert dist.num_params() == 12


def test_tree_hmm() -> None:
    # 3 states, Gaussian emissions, 2-value child counts:
    # root 2 + transitions 3*2 + emissions 3*2 + child counts 1 = 15.
    dist = TreeHiddenMarkovModelDistribution(
        topics=[GaussianDistribution(float(i), 1.0) for i in range(3)],
        w=[0.2, 0.3, 0.5],
        transitions=np.full((3, 3), 1.0 / 3.0),
        len_dist=CategoricalDistribution({0: 0.5, 2: 0.5}),
    )
    assert dist.num_params() == 15


def test_int_markov_chain() -> None:
    # 3 values, lag 2: 3**2 rows * (3-1) = 18; 4-value length adds 3 = 21.
    cond = np.full((9, 3), 1.0 / 3.0)
    dist = IntegerMarkovChainDistribution(
        3, cond, lag=2, len_dist=IntegerCategoricalDistribution(2, [0.25] * 4)
    )
    assert dist.num_params() == 21
    # 4 values, lag 1, no length: 4 * 3 = 12. A shared key counts the chain once.
    keyed = IntegerMarkovChainDistribution(4, np.full((4, 4), 0.25), keys="mc")
    assert keyed.num_params() == 12
    assert CompositeDistribution([keyed, keyed]).num_params() == 12


def test_markov_chain_sparse_maps_and_keys() -> None:
    # init {a, b}: 1; a -> {a, b, c}: 2; b -> {c}: 0; c -> {}: 0. Total 3.
    dist = MarkovChainDistribution(
        {"a": 0.4, "b": 0.6},
        {"a": {"a": 0.2, "b": 0.3, "c": 0.5}, "b": {"c": 1.0}, "c": {}},
        keys="chain",
    )
    assert dist.num_params() == 3
    assert CompositeDistribution([dist, dist]).num_params() == 3


def test_sparse_markov_association() -> None:
    # 3 source values: init 2. Rows have 2, 1, and 0 nonzeros: (2-1) + (1-1) = 1.
    # 2-value length distribution adds 1. Total 4.
    cond = csr_matrix(np.asarray([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    dist = SparseMarkovAssociationDistribution(
        [0.2, 0.3, 0.5],
        cond,
        len_dist=CategoricalDistribution({(1, 1): 0.5, (2, 1): 0.5}),
    )
    assert dist.num_params() == 4
