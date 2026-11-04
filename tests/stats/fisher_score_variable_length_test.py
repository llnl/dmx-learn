"""Fisher scores of dmx.stats variable-length families.

Sequence, IntegerMultinomial, Multinomial, and BernoulliSet sum or collect
per-element scores into one row per record. These tests pin the values, both
sequence scaling modes, the length block, and the sparse forms.
"""

import math
from typing import Any

import numpy as np
import pytest
import scipy.sparse

from dmx.stats import (
    BernoulliSetDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    IntegerMultinomialDistribution,
    MultinomialDistribution,
    PoissonDistribution,
    SequenceDistribution,
)
from dmx.stats._fisher_lengths import length_mean
from dmx.utils.fisher import FisherScoreOptions
from tests.fisher_contract import check_moments

MARGINAL = FisherScoreOptions(sequence_scaling="marginal")
SPARSE = FisherScoreOptions(dense_category_limit=2)
SQRT_HALF = math.sqrt(0.5)


def _encode(dist: Any, data: list) -> Any:
    """Encode ``data`` for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def _scores(dist: Any, data: list, options: Any = None) -> np.ndarray:
    """Return dense Fisher scores of ``data`` under ``dist``."""
    scores = dist.seq_fisher_score(_encode(dist, data), options)
    if scipy.sparse.issparse(scores):
        return np.asarray(scores.toarray())
    return np.asarray(scores)


def _gaussian_sequence() -> SequenceDistribution:
    """Return standard-normal elements with lengths 1, 2, 3 (mean length 2)."""
    return SequenceDistribution(
        GaussianDistribution(0.0, 1.0),
        len_dist=IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]),
    )


SEQUENCES = [[0.5], [1.0, -1.0], [2.0, 0.0, 1.0], []]


def test_sequence_conditional_values() -> None:
    dist = _gaussian_sequence()
    scores = _scores(dist, SEQUENCES)
    # Element block: sum of [u, (u^2 - 1)/sqrt(2)] over the sequence, / sqrt(n).
    expected_elements = [
        [0.5, (0.25 - 1.0) * SQRT_HALF],
        [0.0, 0.0],
        [3.0 / math.sqrt(3.0), 2.0 * SQRT_HALF / math.sqrt(3.0)],
        [0.0, 0.0],  # empty sequence: zero element block
    ]
    np.testing.assert_allclose(scores[:, :2], expected_elements, atol=1e-12)
    lengths = dist.len_dist
    np.testing.assert_allclose(
        scores[:, 2:], _scores(lengths, [1, 2, 3, 0]), atol=1e-15
    )
    assert np.all(np.isfinite(scores))


def test_sequence_marginal_divides_by_mean_length() -> None:
    dist = _gaussian_sequence()
    scores = _scores(dist, SEQUENCES, MARGINAL)
    # E[n] = 0.3 * 1 + 0.4 * 2 + 0.3 * 3 = 2.
    expected_elements = [
        [0.5 / math.sqrt(2.0), (0.25 - 1.0) * SQRT_HALF / math.sqrt(2.0)],
        [0.0, 0.0],
        [3.0 / math.sqrt(2.0), 2.0 * SQRT_HALF / math.sqrt(2.0)],
        [0.0, 0.0],
    ]
    np.testing.assert_allclose(scores[:, :2], expected_elements, atol=1e-12)


def test_sequence_marginal_needs_a_known_mean_length() -> None:
    no_length = SequenceDistribution(GaussianDistribution(0.0, 1.0))
    with pytest.raises(ValueError, match="conditional"):
        no_length.seq_fisher_score(_encode(no_length, [[1.0]]), MARGINAL)
    unknown = SequenceDistribution(
        GaussianDistribution(0.0, 1.0), len_dist=ExponentialDistribution(2.0)
    )
    with pytest.raises(ValueError, match="ExponentialDistribution"):
        unknown.seq_fisher_score(_encode(unknown, [[1.0]]), MARGINAL)


def test_conditional_scaling_whitens_within_each_length() -> None:
    # Lengths 1 and 5 only; the element block must be whitened for each.
    dist = SequenceDistribution(
        GaussianDistribution(1.0, 2.0),
        len_dist=IntegerCategoricalDistribution(1, [0.5, 0.0, 0.0, 0.0, 0.5]),
    )
    data = dist.sampler(seed=7).sample(2000)
    scores = _scores(dist, data)
    lengths = np.asarray([len(row) for row in data])
    for n in (1, 5):
        check_moments(scores[lengths == n, :2], rank_deficit=0, centered=True)


def test_sequence_without_length_distribution_has_one_field() -> None:
    dist = SequenceDistribution(GaussianDistribution(0.0, 1.0), name="values")
    assert dist.fisher_dim() == 2
    assert dist.fisher_field_slices() == {"values": slice(0, 2)}
    np.testing.assert_allclose(
        _scores(dist, [[1.0, 1.0]]), [[2.0 / math.sqrt(2.0), 0.0]], atol=1e-12
    )


def test_sequence_field_slices_inside_composite() -> None:
    dist = CompositeDistribution(
        (
            PoissonDistribution(2.0, name="count"),
            SequenceDistribution(
                GaussianDistribution(0.0, 1.0),
                len_dist=PoissonDistribution(3.0),
                name="visits",
            ),
        )
    )
    assert dist.fisher_field_slices() == {
        "count": slice(0, 1),
        "visits.elements": slice(1, 3),
        "visits.length": slice(3, 4),
    }


@pytest.mark.parametrize(
    "len_dist, mean",
    [
        (PoissonDistribution(3.5), 3.5),
        (GeometricDistribution(0.25), 4.0),
        (BinomialDistribution(0.3, 10, min_val=2), 5.0),
        (IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]), 2.0),
        (CategoricalDistribution({2: 0.5, 4: 0.5}), 3.0),
    ],
    ids=lambda v: type(v).__name__,
)
def test_length_mean(len_dist: Any, mean: Any) -> None:
    assert length_mean(len_dist) == pytest.approx(mean)


def test_integer_multinomial_values() -> None:
    p = [0.2, 0.3, 0.5]
    dist = IntegerMultinomialDistribution(
        0, p, len_dist=IntegerCategoricalDistribution(4, [0.5, 0.5])
    )
    # Counts [1, 0, 3] with total N = 4: (n_c - N p_c) / sqrt(N p_c).
    scores = _scores(dist, [[(0, 1), (2, 3)]])
    n, total = np.asarray([1.0, 0.0, 3.0]), 4.0
    expected = (n - total * np.asarray(p)) / np.sqrt(total * np.asarray(p))
    np.testing.assert_allclose(scores[0, :3], expected, atol=1e-12)
    np.testing.assert_allclose(scores[:, 3:], _scores(dist.len_dist, [4]), atol=1e-15)
    assert dist.fisher_field_slices() == {"counts": slice(0, 3), "length": slice(3, 5)}


def test_integer_multinomial_sparse_drops_offset() -> None:
    dist = IntegerMultinomialDistribution(0, [0.2, 0.3, 0.5])
    data = [[(0, 1), (2, 3)], [(1, 2)]]
    sparse: Any = dist.seq_fisher_score(_encode(dist, data), SPARSE)
    assert scipy.sparse.isspmatrix_csr(sparse)
    counts = np.asarray([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    totals = counts.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        sparse.toarray(), counts / np.sqrt(totals * np.asarray([0.2, 0.3, 0.5]))
    )


def test_multinomial_matches_integer_multinomial() -> None:
    p = [0.2, 0.3, 0.5]
    lengths = IntegerCategoricalDistribution(2, [0.5, 0.5])
    integer = IntegerMultinomialDistribution(0, p, len_dist=lengths)
    labelled = MultinomialDistribution(
        CategoricalDistribution({0: 0.2, 1: 0.3, 2: 0.5}), len_dist=lengths
    )
    data = [[(0, 1.0), (2, 2.0)], [(1, 2.0)], [(2, 1.0), (0, 1.0), (1, 1.0)]]
    for options in (None, MARGINAL):
        np.testing.assert_allclose(
            _scores(labelled, data, options),
            _scores(integer, data, options),
            atol=1e-12,
        )


def test_bernoulli_set_values() -> None:
    dist = BernoulliSetDistribution({"a": 0.5, "b": 0.2})
    scores = _scores(dist, [["a"], ["a", "b"], []])
    expected = [[1.0, -0.5], [1.0, 2.0], [-1.0, -0.5]]
    np.testing.assert_allclose(scores, expected, atol=1e-12)


def test_bernoulli_set_required_element_and_sparse_form() -> None:
    dist = BernoulliSetDistribution({"a": 1.0, "b": 0.2}, min_prob=0.0)
    assert np.all(np.isfinite(_scores(dist, [["a"], ["a", "b"]])))
    plain = BernoulliSetDistribution({"a": 0.5, "b": 0.2})
    # Two elements: a limit of 1 forces the sparse form.
    one_dense = FisherScoreOptions(dense_category_limit=1)
    sparse: Any = plain.seq_fisher_score(_encode(plain, [["a"], ["b"]]), one_dense)
    assert scipy.sparse.isspmatrix_csr(sparse)
    # The offset-free form is the dense one plus p / sqrt(p (1 - p)).
    offset = np.asarray([0.5 / 0.5, 0.2 / 0.4])
    np.testing.assert_allclose(
        sparse.toarray() - _scores(plain, [["a"], ["b"]]), np.tile(offset, (2, 1))
    )


@pytest.mark.parametrize(
    "dist",
    [
        _gaussian_sequence(),
        IntegerMultinomialDistribution(0, [0.5, 0.5]),
        MultinomialDistribution(CategoricalDistribution({"a": 1.0})),
        BernoulliSetDistribution({"a": 0.5}),
    ],
    ids=lambda v: type(v).__name__,
)
def test_wrong_encoding_raises(dist: Any) -> None:
    encoded = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([1.0])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        dist.seq_fisher_score(encoded)
