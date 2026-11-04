"""Fisher scores of dmx.stats Markov chains.

A fully observed chain scores its first state, the transitions out of each
source state, and its length. These tests pin the values on a 2-state chain,
both sequence scaling modes, ``default_value`` handling, and the sparse form.
"""

import math
from typing import Any

import numpy as np
import pytest
import scipy.sparse

from dmx.stats import (
    BinomialDistribution,
    CategoricalDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    MarkovChainDistribution,
    PoissonDistribution,
)
from dmx.stats._fisher_lengths import length_mean, length_pmf
from dmx.utils.fisher import FisherScoreOptions

MARGINAL = FisherScoreOptions(sequence_scaling="marginal")
INIT = {0: 0.6, 1: 0.4}
TRANS = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.2, 1: 0.8}}


def _encode(dist: Any, data: list) -> Any:
    """Encode ``data`` for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def _scores(dist: Any, data: list, options: Any = None) -> np.ndarray:
    """Return dense Fisher scores of ``data`` under ``dist``."""
    scores = dist.seq_fisher_score(_encode(dist, data), options)
    if scipy.sparse.issparse(scores):
        return np.asarray(scores.toarray())
    return np.asarray(scores)


def _onehot(level: int, probs: list) -> np.ndarray:
    """Return the categorical score ``e_level / sqrt(q) - sqrt(q)``."""
    q = np.asarray(probs)
    score = -np.sqrt(q)
    score[level] += 1.0 / math.sqrt(q[level])
    return np.asarray(score)


def test_conditional_values() -> None:
    dist = MarkovChainDistribution(dict(INIT), TRANS)
    # Sequences in an order that encodes state 1 before state 0.
    scores = _scores(dist, [[1], [0, 0, 1, 1, 1], []])
    assert scores.shape == (3, 6)
    row0 = (_onehot(0, [0.7, 0.3]) + _onehot(1, [0.7, 0.3])) / math.sqrt(2.0)
    row1 = 2.0 * _onehot(1, [0.2, 0.8]) / math.sqrt(2.0)
    expected = [
        # One state: no transitions, so zero transition blocks.
        np.concatenate((_onehot(1, [0.6, 0.4]), np.zeros(4))),
        np.concatenate((_onehot(0, [0.6, 0.4]), row0, row1)),
        np.zeros(6),  # empty chain
    ]
    np.testing.assert_allclose(scores, expected, atol=1e-12)


def test_marginal_divides_by_expected_visits() -> None:
    lengths = IntegerCategoricalDistribution(2, [0.5, 0.5])
    dist = MarkovChainDistribution(dict(INIT), TRANS, len_dist=lengths)
    scores = _scores(dist, [[0, 1, 1], [1, 0]], MARGINAL)
    # Lengths 2 and 3: E[c] = pi_0 + P(n = 3) pi_0 P, with pi_0 P = (0.5, 0.5).
    visits = (0.6 + 0.5 * 0.5, 0.4 + 0.5 * 0.5)
    expected = [
        np.concatenate(
            (
                _onehot(0, [0.6, 0.4]),
                _onehot(1, [0.7, 0.3]) / math.sqrt(visits[0]),
                _onehot(1, [0.2, 0.8]) / math.sqrt(visits[1]),
            )
        ),
        np.concatenate(
            (
                _onehot(1, [0.6, 0.4]),
                np.zeros(2),
                _onehot(0, [0.2, 0.8]) / math.sqrt(visits[1]),
            )
        ),
    ]
    np.testing.assert_allclose(scores[:, :6], expected, atol=1e-12)
    np.testing.assert_allclose(scores[:, 6:], _scores(lengths, [3, 2]), atol=1e-15)


def test_marginal_needs_a_known_length_distribution() -> None:
    no_length = MarkovChainDistribution(dict(INIT), TRANS)
    with pytest.raises(ValueError, match="conditional"):
        no_length.seq_fisher_score(_encode(no_length, [[0, 1]]), MARGINAL)
    unknown = MarkovChainDistribution(
        dict(INIT), TRANS, len_dist=ExponentialDistribution(2.0)
    )
    with pytest.raises(ValueError, match="ExponentialDistribution"):
        unknown.seq_fisher_score(_encode(unknown, [[0, 1]]), MARGINAL)


def test_default_value_adds_other_columns() -> None:
    dist = MarkovChainDistribution(dict(INIT), TRANS, default_value=0.1)
    assert dist.fisher_dim() == 9
    scores = _scores(dist, [[0, 2], [2, 0]])
    weights = [[0.6, 0.4, 0.1], [0.7, 0.3, 0.1], [0.2, 0.8, 0.1]]
    q = [list(np.asarray(w) / sum(w)) for w in weights]
    expected = [
        # The unseen state 2 is the "other" target of row 0.
        np.concatenate((_onehot(0, q[0]), _onehot(2, q[1]), np.zeros(3))),
        # State 2 is the "other" initial state and has no transition row.
        np.concatenate((_onehot(2, q[0]), np.zeros(6))),
    ]
    np.testing.assert_allclose(scores, expected, atol=1e-12)


def test_zero_default_value_gives_no_level_rows() -> None:
    dist = MarkovChainDistribution(dict(INIT), TRANS)
    scores = _scores(dist, [[0, 2], [2, 0]])
    expected = [
        np.concatenate((_onehot(0, [0.6, 0.4]), -np.sqrt([0.7, 0.3]), np.zeros(2))),
        np.concatenate((-np.sqrt([0.6, 0.4]), np.zeros(4))),
    ]
    np.testing.assert_allclose(scores, expected, atol=1e-12)


def test_sparse_form_drops_offsets() -> None:
    dist = MarkovChainDistribution(dict(INIT), TRANS)
    data = [[0, 0, 1, 1, 1]]
    one_dense = FisherScoreOptions(dense_category_limit=1)
    sparse: Any = dist.seq_fisher_score(_encode(dist, data), one_dense)
    assert scipy.sparse.isspmatrix_csr(sparse)
    # Each block drops sqrt(q) once per summed row, then is divided by sqrt(c).
    offset = np.concatenate(
        (
            np.sqrt([0.6, 0.4]),
            math.sqrt(2.0) * np.sqrt([0.7, 0.3]),
            math.sqrt(2.0) * np.sqrt([0.2, 0.8]),
        )
    )
    np.testing.assert_allclose(sparse.toarray() - _scores(dist, data), [offset])


def test_field_slices() -> None:
    plain = MarkovChainDistribution(dict(INIT), TRANS)
    assert plain.fisher_field_slices() == {
        "initial": slice(0, 2),
        "transitions": slice(2, 6),
    }
    with_length = MarkovChainDistribution(
        dict(INIT), TRANS, len_dist=PoissonDistribution(3.0)
    )
    assert with_length.fisher_dim() == 7
    assert with_length.fisher_field_slices() == {
        "initial": slice(0, 2),
        "transitions": slice(2, 6),
        "length": slice(6, 7),
    }


def test_wrong_encoding_raises() -> None:
    dist: Any = MarkovChainDistribution(dict(INIT), TRANS)
    encoded = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([1.0])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        dist.seq_fisher_score(encoded)


@pytest.mark.parametrize(
    "len_dist",
    [
        PoissonDistribution(3.5),
        GeometricDistribution(0.25),
        BinomialDistribution(0.3, 10, min_val=2),
        IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]),
        CategoricalDistribution({2: 0.5, 4: 0.5}),
    ],
    ids=lambda v: type(v).__name__,
)
def test_length_pmf_matches_mean(len_dist: Any) -> None:
    pmf = length_pmf(len_dist)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.dot(np.arange(len(pmf)), pmf) == pytest.approx(length_mean(len_dist))
