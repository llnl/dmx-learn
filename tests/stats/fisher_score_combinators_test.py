"""Fisher scores of dmx.stats observed combinators.

Composite, Optional, Ignored, and NullDistribution build their scores from
their children's. These tests pin how: concatenation, field names, field
balancing, the missing-value indicator, and empty scores.
"""

import math
from typing import Any

import numpy as np
import pytest
import scipy.sparse

from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    IgnoredDistribution,
    NullDistribution,
    OptionalDistribution,
    PoissonDistribution,
    VonMisesFisherDistribution,
)
from dmx.utils.fisher import FisherScoreOptions, has_fisher_score

BALANCED = FisherScoreOptions(balance_fields=True)
ABC = {"a": 0.5, "b": 0.25, "c": 0.25}


def _encode(dist: Any, data: list) -> Any:
    """Encode ``data`` for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def _dense(scores: Any) -> np.ndarray:
    """Return scores as a dense array."""
    if scipy.sparse.issparse(scores):
        return np.asarray(scores.toarray())
    return np.asarray(scores)


def _mixed() -> CompositeDistribution:
    """Return the mixed composite used by several tests."""
    return CompositeDistribution(
        (
            GaussianDistribution(1.0, 4.0, name="height"),
            CategoricalDistribution(dict(ABC), name="color"),
            OptionalDistribution(PoissonDistribution(2.0), p=0.25, name="visits"),
            IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
        )
    )


# The fourth (ignored) field still needs a value in every record.
MIXED_DATA = [(3.0, "a", 1, 0.0), (-1.0, "c", None, 0.5), (1.0, "b", 4, -0.5)]


def test_composite_is_concatenation_of_fields() -> None:
    dist = _mixed()
    scores = _dense(dist.seq_fisher_score(_encode(dist, MIXED_DATA)))
    expected = np.hstack(
        [
            field.seq_fisher_score(_encode(field, [row[i] for row in MIXED_DATA]))
            for i, field in enumerate(dist.dists[:3])
        ]
    )
    np.testing.assert_array_equal(scores, expected)
    assert dist.fisher_dim() == 2 + 3 + 2


def test_composite_field_slices_name_each_field() -> None:
    # The ignored field has no columns and no entry.
    assert _mixed().fisher_field_slices() == {
        "height": slice(0, 2),
        "color": slice(2, 5),
        "visits": slice(5, 7),
    }


def test_composite_field_slices_for_unnamed_and_repeated_names() -> None:
    dist = CompositeDistribution(
        (
            GaussianDistribution(0.0, 1.0),
            PoissonDistribution(2.0, name="count"),
            PoissonDistribution(3.0, name="count"),
        )
    )
    assert dist.fisher_field_slices() == {
        "field_0": slice(0, 2),
        "count": slice(2, 3),
        "count_2": slice(3, 4),
    }


def test_nested_composite_field_slices_use_dotted_names() -> None:
    inner = CompositeDistribution(
        (
            PoissonDistribution(2.0, name="b"),
            CategoricalDistribution(dict(ABC), name="c"),
        ),
        name="inner",
    )
    dist = CompositeDistribution((GaussianDistribution(0.0, 1.0, name="a"), inner))
    assert dist.fisher_field_slices() == {
        "a": slice(0, 2),
        "inner.b": slice(2, 3),
        "inner.c": slice(3, 6),
    }


def test_balance_fields_divides_each_field_once() -> None:
    inner = CompositeDistribution(
        (
            PoissonDistribution(2.0, name="b"),
            CategoricalDistribution(dict(ABC), name="c"),
        ),
        name="inner",
    )
    dist = CompositeDistribution((GaussianDistribution(0.0, 1.0, name="a"), inner))
    encoded = _encode(dist, [(0.5, (1, "a")), (-2.0, (3, "c"))])
    plain = _dense(dist.seq_fisher_score(encoded))
    balanced = _dense(dist.seq_fisher_score(encoded, BALANCED))
    # Fields a, inner.b, and inner.c have widths 2, 1, and 3; nesting does not
    # divide inner.b or inner.c a second time.
    weights = np.concatenate(
        [np.full(2, 1.0 / math.sqrt(2.0)), [1.0], np.full(3, 1.0 / math.sqrt(3.0))]
    )
    np.testing.assert_allclose(balanced, plain * weights, rtol=1e-12)
    assert dist.fisher_dim(BALANCED) == dist.fisher_dim()


def test_composite_with_sparse_field_returns_csr() -> None:
    dist = CompositeDistribution(
        (
            GaussianDistribution(0.0, 1.0, name="x"),
            CategoricalDistribution(dict(ABC), name="color"),
        )
    )
    encoded = _encode(dist, [(0.0, "a"), (1.0, "c")])
    sparse_options = FisherScoreOptions(dense_category_limit=2)
    scores = dist.seq_fisher_score(encoded, sparse_options)
    assert scipy.sparse.isspmatrix_csr(scores)
    assert scores.shape == (2, 5)
    dense = _dense(dist.seq_fisher_score(encoded))
    np.testing.assert_allclose(_dense(scores)[:, :2], dense[:, :2])
    # The sparse categorical block is the dense one without the -sqrt(q) offset.
    np.testing.assert_allclose(
        _dense(scores)[:, 2:] - dense[:, 2:],
        np.tile(np.sqrt([0.5, 0.25, 0.25]), (2, 1)),
    )


def test_optional_values_with_missing_probability() -> None:
    dist = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25)
    scores = _dense(dist.seq_fisher_score(_encode(dist, [1.0, None, 3.0])))
    present = -0.25 / math.sqrt(0.1875)
    missing = 0.75 / math.sqrt(0.1875)
    scale = 1.0 / math.sqrt(0.75)
    expected = [
        [present, scale * 1.0, scale * 0.0],
        [missing, 0.0, 0.0],
        [present, scale * 3.0, scale * 8.0 / math.sqrt(2.0)],
    ]
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-15)
    assert dist.fisher_dim() == 3


@pytest.mark.parametrize("p", [None, 0.0], ids=["unmodeled", "zero"])
def test_optional_without_missingness_drops_indicator(p: Any) -> None:
    dist = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=p)
    scores = _dense(dist.seq_fisher_score(_encode(dist, [1.0, None])))
    np.testing.assert_allclose(scores, [[1.0, 0.0], [0.0, 0.0]], atol=1e-15)
    assert dist.fisher_dim() == 2


def test_ignored_and_null_have_empty_scores() -> None:
    ignored = IgnoredDistribution(GaussianDistribution(0.0, 1.0))
    assert ignored.fisher_dim() == 0
    assert ignored.seq_fisher_score(_encode(ignored, [1.0, 2.0, 3.0])).shape == (3, 0)
    null = NullDistribution()
    assert null.fisher_dim() == 0
    # A null encoding stores no observations, so the row count is unknown.
    assert null.seq_fisher_score(_encode(null, [None, None])).shape == (0, 0)


def test_composite_of_only_empty_fields() -> None:
    dist = CompositeDistribution(
        (
            IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
            IgnoredDistribution(PoissonDistribution(1.0)),
        ),
        name="nothing",
    )
    scores = dist.seq_fisher_score(_encode(dist, [(0.0, 1), (1.0, 2)]))
    assert scores.shape == (2, 0)
    assert dist.fisher_field_slices() == {"nothing": slice(0, 0)}


def test_support_check_recurses_into_children() -> None:
    unscored = VonMisesFisherDistribution([1.0, 0.0], 2.0)
    assert has_fisher_score(_mixed())
    assert not has_fisher_score(
        CompositeDistribution((GaussianDistribution(0.0, 1.0), unscored))
    )
    assert not has_fisher_score(OptionalDistribution(unscored, p=0.1))


@pytest.mark.parametrize(
    "dist",
    [_mixed(), OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.1)],
    ids=lambda v: type(v).__name__,
)
def test_wrong_encoding_raises(dist: Any) -> None:
    encoded = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([1.0])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        dist.seq_fisher_score(encoded)
