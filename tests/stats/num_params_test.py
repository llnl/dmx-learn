"""Tests for num_params() across dmx.stats combinators and keyed sharing.

Per-distribution counts live in each distribution's test case through the
``num_params_cases`` hook of ``tests.stats.stats_tests.StatsTestClass``. This
module covers distributions without a harness test case, parameter tying through
estimator keys across distribution types, and the base-class default.
"""

from typing import Any, Optional

import pytest

from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    MixtureDistribution,
    OptionalDistribution,
    PoissonDistribution,
    SemiSupervisedMixtureDistribution,
    SequenceDistribution,
)
from dmx.stats.pdist import ProbabilityDistribution
from dmx.stats.select import SelectDistribution


def gaussian(keys: Optional[str] = None) -> GaussianDistribution:
    """Return a standard Gaussian with optional keys."""
    return GaussianDistribution(0.0, 1.0, keys=keys)


def test_ss_mixture_num_params() -> None:
    # 3 components: 2 weights + 3 * 2 Gaussian parameters.
    dist = SemiSupervisedMixtureDistribution(
        [gaussian(), gaussian(), gaussian()], [0.2, 0.3, 0.5]
    )
    assert dist.num_params() == 8


def test_select_num_params() -> None:
    # Gaussian (2) + exponential (1); the choice function has no parameters.
    dist = SelectDistribution(
        [gaussian(), ExponentialDistribution(1.0)], choice_function=lambda x: 0
    )
    assert dist.num_params() == 3


def test_keyed_leaves_counted_once() -> None:
    # Three Gaussians share key "g" (2), plus one unkeyed Gaussian (2).
    dist = CompositeDistribution(
        [gaussian("g"), gaussian("g"), gaussian(), gaussian("g")]
    )
    assert dist.num_params() == 4


def test_keyed_components_within_mixture() -> None:
    # Components tied to one key: 2 weights + one Gaussian (2).
    dist = MixtureDistribution(
        [gaussian("g"), gaussian("g"), gaussian("g")], [0.2, 0.3, 0.5]
    )
    assert dist.num_params() == 4


def test_keyed_sharing_across_nesting_levels() -> None:
    # The keyed Gaussian appears inside an optional, a sequence, and a mixture.
    # Optional: 2 + p (1). Sequence: tied Gaussian (0) + Poisson (1).
    # Mixture: 1 weight + tied Gaussian (0) + exponential (1).
    dist = CompositeDistribution(
        [
            OptionalDistribution(gaussian("g"), p=0.1),
            SequenceDistribution(gaussian("g"), len_dist=PoissonDistribution(2.0)),
            MixtureDistribution(
                [gaussian("g"), ExponentialDistribution(1.0)], [0.5, 0.5]
            ),
        ]
    )
    assert dist.num_params() == 6


def test_keyed_mixture_weights_across_mixtures() -> None:
    # Weights shared by key "w" (2) and each mixture's own components (3 * 2).
    mix1 = MixtureDistribution(
        [gaussian(), gaussian(), gaussian()], [0.2, 0.3, 0.5], keys=("w", None)
    )
    mix2 = MixtureDistribution(
        [
            CategoricalDistribution({"a": 0.5, "b": 0.5}),
            CategoricalDistribution({"a": 0.1, "b": 0.9}),
            CategoricalDistribution({"a": 0.3, "b": 0.7}),
        ],
        [0.1, 0.2, 0.7],
        keys=("w", None),
    )
    assert CompositeDistribution([mix1, mix2]).num_params() == 2 + 6 + 3


def test_num_params_is_repeatable_and_does_not_mutate() -> None:
    dist = CompositeDistribution([gaussian("g"), gaussian("g")])
    before = str(dist)
    assert dist.num_params() == 2
    assert dist.num_params() == 2
    assert str(dist) == before


class _NoCountDistribution(ProbabilityDistribution):
    """Subclass that does not implement the num_params hook."""

    def density(self, x: Any) -> float:
        return 1.0

    def log_density(self, x: Any) -> float:
        return 0.0

    def sampler(self, seed: Optional[int] = None) -> Any:
        raise NotImplementedError

    def estimator(self, pseudo_count: Optional[float] = None) -> Any:
        raise NotImplementedError


def test_base_default_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="_NoCountDistribution"):
        _NoCountDistribution().num_params()
