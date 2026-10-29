"""Tests for the Fisher score API: options, support checks, and base defaults."""

import dataclasses
from typing import Any, Optional

import numpy as np
import pytest

from dmx.bstats import GaussianDistribution as BstatsGaussianDistribution
from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MarkovChainDistribution,
    MixtureDistribution,
)
from dmx.utils.fisher import (
    FisherScoreOptions,
    as_score_array,
    has_fisher_score,
    resolve_fisher_options,
)


class _ScoredGaussian(GaussianDistribution):
    """Gaussian that reports a Fisher dimension, standing in for a real score."""

    def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int:
        """Return a fixed dimension, or one less with ``location_only``."""
        return 1 if resolve_fisher_options(options).location_only else 2


# The bstats base class marks entropy() abstract; the test only needs fisher_dim.
# pylint: disable-next=abstract-method
class _ScoredBstatsGaussian(BstatsGaussianDistribution):
    """Bayesian Gaussian that reports a Fisher dimension."""

    def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int:
        """Return a fixed dimension."""
        del options
        return 2


def _unsupported_stats() -> list[Any]:
    """Return dmx.stats distributions that have no Fisher score yet."""
    return [
        GaussianDistribution(0.0, 1.0),
        CategoricalDistribution({"a": 0.5, "b": 0.5}),
        CompositeDistribution(
            (GaussianDistribution(0.0, 1.0), CategoricalDistribution({"a": 1.0}))
        ),
        MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        ),
        MarkovChainDistribution({"a": 1.0}, {"a": {"a": 1.0}}),
    ]


def test_options_defaults() -> None:
    options = FisherScoreOptions()
    assert options.prob_floor == 1.0e-6
    assert options.location_only is False
    assert options.sequence_scaling == "conditional"
    assert options.balance_fields is False
    assert options.on_missing == "raise"
    assert options.dense_category_limit == 256


def test_options_are_frozen() -> None:
    options = FisherScoreOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.prob_floor = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prob_floor": 0.0},
        {"prob_floor": 1.0},
        {"prob_floor": -1.0e-3},
        {"sequence_scaling": "per_element"},
        {"on_missing": "ignore"},
        {"dense_category_limit": 0},
        {"dense_category_limit": 2.5},
        {"dense_category_limit": True},
    ],
)
def test_options_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        FisherScoreOptions(**kwargs)


def test_resolve_fisher_options() -> None:
    assert resolve_fisher_options(None) == FisherScoreOptions()
    options = FisherScoreOptions(location_only=True)
    assert resolve_fisher_options(options) is options


@pytest.mark.parametrize("dist", _unsupported_stats(), ids=lambda d: type(d).__name__)
def test_stats_base_methods_raise_naming_the_class(dist: Any) -> None:
    name = type(dist).__name__
    with pytest.raises(NotImplementedError, match=name):
        dist.seq_fisher_score(None)
    with pytest.raises(NotImplementedError, match=name):
        dist.fisher_dim()
    with pytest.raises(NotImplementedError, match=name):
        dist.fisher_field_slices()
    assert not has_fisher_score(dist)


def test_bstats_base_methods_raise_naming_the_class() -> None:
    dist = BstatsGaussianDistribution(0.0, 1.0)
    with pytest.raises(NotImplementedError, match="GaussianDistribution"):
        dist.seq_fisher_score(np.zeros(1))
    with pytest.raises(NotImplementedError, match="GaussianDistribution"):
        dist.fisher_dim()
    with pytest.raises(NotImplementedError, match="GaussianDistribution"):
        dist.fisher_field_slices()
    assert not has_fisher_score(dist)


def test_has_fisher_score_reports_supported_distributions() -> None:
    assert has_fisher_score(_ScoredGaussian(0.0, 1.0))
    assert has_fisher_score(_ScoredBstatsGaussian(0.0, 1.0))


def test_has_fisher_score_rejects_non_distributions() -> None:
    assert not has_fisher_score(object())
    assert not has_fisher_score(None)


def test_stats_field_slices_default() -> None:
    assert _ScoredGaussian(0.0, 1.0).fisher_field_slices() == {
        "_ScoredGaussian": slice(0, 2)
    }
    named = _ScoredGaussian(0.0, 1.0, name="height")
    assert named.fisher_field_slices() == {"height": slice(0, 2)}
    location = FisherScoreOptions(location_only=True)
    assert named.fisher_field_slices(location) == {"height": slice(0, 1)}


def test_bstats_field_slices_default() -> None:
    assert _ScoredBstatsGaussian(0.0, 1.0).fisher_field_slices() == {
        "_ScoredBstatsGaussian": slice(0, 2)
    }
    named = _ScoredBstatsGaussian(0.0, 1.0, name="height")
    assert named.fisher_field_slices() == {"height": slice(0, 2)}


def test_as_score_array_shapes() -> None:
    column = as_score_array([1, 2, 3], 1)
    assert column.shape == (3, 1)
    assert column.dtype == np.float64
    matrix = as_score_array(np.arange(6, dtype=np.int64).reshape(3, 2), 2)
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float64
    assert matrix.flags.c_contiguous
    empty = as_score_array(np.zeros((0, 4)), 4)
    assert empty.shape == (0, 4)


@pytest.mark.parametrize(
    "values, dim",
    [([1.0, 2.0], 2), (np.zeros((3, 2)), 3), (np.zeros((2, 2, 2)), 2)],
)
def test_as_score_array_rejects_wrong_shapes(values: Any, dim: int) -> None:
    with pytest.raises(ValueError, match="Fisher scores"):
        as_score_array(values, dim)
