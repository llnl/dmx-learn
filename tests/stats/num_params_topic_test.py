"""Tests for num_params() on topic, association, and tree distributions.

These distributions have no shared-harness test file, so their hand-computed
counts and keyed-sharing cases live here.
"""

from typing import Optional

import numpy as np

from dmx.stats.categorical import CategoricalDistribution
from dmx.stats.composite import CompositeDistribution
from dmx.stats.icltree import ICLTreeDistribution
from dmx.stats.int_hidden_association import IntegerHiddenAssociationDistribution
from dmx.stats.int_plsi import IntegerPLSIDistribution
from dmx.stats.lda import LDADistribution
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution


def _lda(keys: tuple = (None, None)) -> LDADistribution:
    topics = [
        CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
        CategoricalDistribution({"a": 0.2, "b": 0.3, "c": 0.5}),
    ]
    return LDADistribution(topics, alpha=[1.0, 1.0], keys=keys)


def _int_hidden_association(
    keys: tuple = (None, None),
    len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
) -> IntegerHiddenAssociationDistribution:
    state_prob_mat = np.full((2, 3), 1.0 / 3.0)  # K=2 states, V2=3 targets
    cond_weights = np.full((4, 2), 0.5)  # V1=4 sources, K=2 states
    return IntegerHiddenAssociationDistribution(
        state_prob_mat, cond_weights, len_dist=len_dist, keys=keys
    )


def test_lda_num_params() -> None:
    # alpha: 2, two Categorical(3) topics: 2 * 2 = 4.
    assert _lda().num_params() == 6


def test_lda_keyed_sharing() -> None:
    # Both blocks shared: the second LDA adds nothing.
    shared = CompositeDistribution([_lda(("a", "t")), _lda(("a", "t"))])
    assert shared.num_params() == 6
    # Only alpha shared: the second LDA adds its topics (4).
    alpha_only = CompositeDistribution([_lda(("a", None)), _lda(("a", None))])
    assert alpha_only.num_params() == 10
    # Only topics shared: the second LDA adds its alpha (2).
    topics_only = CompositeDistribution([_lda((None, "t")), _lda((None, "t"))])
    assert topics_only.num_params() == 8


def test_int_hidden_association_num_params() -> None:
    # V1*(K-1) + K*(V2-1) = 4*1 + 2*2 = 8; Null prev/len add 0.
    assert _int_hidden_association().num_params() == 8
    len_dist = CategoricalDistribution({2: 0.5, 3: 0.5})
    assert _int_hidden_association(len_dist=len_dist).num_params() == 9


def test_int_hidden_association_keyed_sharing() -> None:
    shared = CompositeDistribution(
        [_int_hidden_association(("w", "s")), _int_hidden_association(("w", "s"))]
    )
    assert shared.num_params() == 8
    # Only cond_weights shared: the second adds state_prob_mat (4).
    weights_only = CompositeDistribution(
        [_int_hidden_association(("w", None)), _int_hidden_association(("w", None))]
    )
    assert weights_only.num_params() == 12


def test_int_plsi_keyed_sharing() -> None:
    # S=2, V=3, D=2: S*(V-1) + D*(S-1) + (D-1) = 4 + 2 + 1 = 7.
    def plsi(keys: tuple) -> IntegerPLSIDistribution:
        return IntegerPLSIDistribution(
            np.full((3, 2), 1.0 / 3.0), np.full((2, 2), 0.5), [0.5, 0.5], keys=keys
        )

    assert plsi((None, None, None)).num_params() == 7
    shared = CompositeDistribution([plsi(("w", "s", "d")), plsi(("w", "s", "d"))])
    assert shared.num_params() == 7
    # Only the word block shared: the second adds 2 + 1.
    words_only = CompositeDistribution(
        [plsi(("w", None, None)), plsi(("w", None, None))]
    )
    assert words_only.num_params() == 10


def test_icltree_num_params() -> None:
    # Root over 3 values: 2. Feature 1 (2 values) given feature 0: 3 * 1 = 3.
    # Feature 2 (4 values) given feature 0: 3 * 3 = 9. Total 14.
    dist = ICLTreeDistribution(
        [(0, None), (1, 0), (2, 0)],
        [
            np.log(np.full(3, 1.0 / 3.0)),
            np.log(np.full((3, 2), 0.5)),
            np.log(np.full((3, 4), 0.25)),
        ],
    )
    assert dist.num_params() == 14
