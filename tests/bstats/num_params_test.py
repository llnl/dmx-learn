"""Tests for num_params() and DPM effective component counts in dmx.bstats.

Distributions exercised by the shared Bayesian harness check their count
through ``BayesianDistributionTestCase.num_params``. This module covers the
remaining distributions, parameter tying through keys, and the Dirichlet-process
mixture's effective component count.
"""

import contextlib
import io
from typing import Any, Optional

import numpy as np
import pytest

from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.bestimation import optimize
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.categorical import CategoricalDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.conditional import ConditionalDistribution
from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
)
from dmx.bstats.exponential import ExponentialDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.bstats.geometric import GeometricDistribution
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.bstats.optional import OptionalDistribution
from dmx.bstats.pdist import ProbabilityDistribution
from dmx.bstats.poisson import PoissonDistribution
from dmx.bstats.sequence import SequenceDistribution
from dmx.bstats.setdist import BernoulliSetDistribution
from dmx.bstats.symdirichlet import SymmetricDirichletDistribution
from dmx.utils.automatic import get_dpm_mixture


def gaussian() -> GaussianDistribution:
    """Return a Gaussian likelihood with two free parameters."""
    return GaussianDistribution(0.0, 1.0)


@pytest.mark.parametrize(
    ("dist", "expected"),
    [
        # Two shape parameters.
        (BetaDistribution(2.0, 3.0), 2),
        # Shape and scale.
        (GammaDistribution(2.0, 3.0), 2),
        # mu, lam, a, b.
        (NormalGammaDistribution(0.0, 1.0, 2.0, 3.0), 4),
        # Four length-2 arrays: 4 * 2.
        (
            MultivariateNormalGammaDistribution(
                np.zeros(2), np.ones(2), np.ones(2) * 2.0, np.ones(2)
            ),
            8,
        ),
        # One concentration per dimension, or one shared scalar.
        (DirichletDistribution([1.0, 2.0, 3.0]), 3),
        (DirichletDistribution(2.0), 1),
        (SymmetricDirichletDistribution(2.0, ndim=4), 1),
        # One concentration per key, or one shared scalar.
        (DictDirichletDistribution({"a": 1.0, "b": 2.0, "c": 3.0}), 3),
        (DictDirichletDistribution(2.0), 1),
        # One Bernoulli probability per label.
        (BernoulliSetDistribution({"x": 0.2, "y": 0.5, "z": 0.7}), 3),
    ],
)
def test_prior_and_set_distributions(dist: Any, expected: int) -> None:
    assert dist.num_params() == expected


def test_composite_num_params() -> None:
    # Gaussian (2) + categorical over 3 values (2) + Poisson (1).
    dist = CompositeDistribution(
        (
            gaussian(),
            CategoricalDistribution({"a": 0.2, "b": 0.3, "c": 0.5}),
            PoissonDistribution(2.0),
        )
    )
    assert dist.num_params() == 5


def test_optional_num_params() -> None:
    # Missing-value probability (1) + Gaussian (2).
    assert OptionalDistribution(gaussian(), p=0.1).num_params() == 3


def test_sequence_num_params() -> None:
    # Gaussian (2) + Poisson length (1); a null length adds 0.
    assert SequenceDistribution(gaussian(), PoissonDistribution(2.0)).num_params() == 3
    assert SequenceDistribution(gaussian()).num_params() == 2


def test_conditional_num_params() -> None:
    # Two conditionals (2 + 2) + exponential default (1); cond_dist is fixed.
    dist = ConditionalDistribution(
        {"a": gaussian(), "b": gaussian()},
        CategoricalDistribution({"a": 0.5, "b": 0.5}),
        default_dist=ExponentialDistribution(1.0),
    )
    assert dist.num_params() == 5


def test_mixture_num_params() -> None:
    # 3 components: 2 weights + 3 * 2 Gaussian parameters.
    dist = MixtureDistribution([gaussian(), gaussian(), gaussian()], [0.2, 0.3, 0.5])
    assert dist.num_params() == 8


def test_keyed_leaves_counted_once() -> None:
    # Each keyed leaf appears twice but counts once: 1 + 1 + 1, plus an unkeyed
    # Poisson (1).
    dist = CompositeDistribution(
        (
            BernoulliDistribution(0.3, keys="b"),
            BernoulliDistribution(0.6, keys="b"),
            PoissonDistribution(2.0, keys="p"),
            PoissonDistribution(3.0, keys="p"),
            GeometricDistribution(0.4, keys="g"),
            GeometricDistribution(0.5, keys="g"),
            PoissonDistribution(4.0),
        )
    )
    assert dist.num_params() == 4


def test_keyed_composite_and_optional_tie_subtrees() -> None:
    # Keyed composites (2 + 1) and keyed optionals (1 + 2) each count once.
    inner = (gaussian(), PoissonDistribution(1.0))
    dist = CompositeDistribution(
        (
            CompositeDistribution(inner, keys="c"),
            CompositeDistribution(inner, keys="c"),
            OptionalDistribution(gaussian(), keys="o"),
            OptionalDistribution(gaussian(), keys="o"),
        )
    )
    assert dist.num_params() == 6


def test_num_params_is_repeatable_and_does_not_mutate() -> None:
    dist = CompositeDistribution(
        (PoissonDistribution(1.0, keys="p"), PoissonDistribution(2.0, keys="p"))
    )
    before = str(dist)
    assert dist.num_params() == 1
    assert dist.num_params() == 1
    assert str(dist) == before


# Only num_params is exercised, so the other optional methods stay unimplemented.
class _NoCountDistribution(  # pylint: disable=abstract-method
    ProbabilityDistribution[Any, Any, Any]
):
    """Subclass that does not implement the num_params hook."""


def test_base_default_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="_NoCountDistribution"):
        _NoCountDistribution().num_params()


def _dpm(weights: list[float]) -> DirichletProcessMixtureDistribution:
    """Return a Gaussian DPM with the given predictive weights."""
    k = len(weights)
    return DirichletProcessMixtureDistribution(
        [GaussianDistribution(float(i), 1.0) for i in range(k)],
        weights,
        1.0,
        np.ones((k, 2)),
        [NormalGammaDistribution(0.0, 1.0, 2.0, 1.0) for _ in range(k)],
    )


def test_dpm_num_params_counts_full_truncation() -> None:
    # K=4: 3 weights + 4 * 2 Gaussian parameters; the concentration is excluded.
    assert _dpm([0.5, 0.3, 0.2, 0.0]).num_params() == 11


def test_dpm_effective_counts_use_weight_threshold() -> None:
    dist = _dpm([0.5, 0.3, 0.15, 0.05])
    assert dist.effective_num_components(0.0) == 4
    # Components with w >= 0.15 are kept, matching get_dpm_mixture's ">=".
    assert dist.effective_num_components(0.15) == 3
    # 3 retained: 2 weights + 3 * 2 Gaussian parameters.
    assert dist.effective_num_params(0.15) == 8
    assert dist.effective_num_components(0.9) == 0
    assert dist.effective_num_params(0.9) == 0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_dpm_effective_rejects_invalid_threshold(bad: float) -> None:
    with pytest.raises(ValueError, match="min_weight"):
        _dpm([0.5, 0.5]).effective_num_components(bad)


def _planted_data(seed: int) -> list[float]:
    """Draw 100 points from each of three well-separated Gaussians."""
    rng = np.random.RandomState(seed)
    clusters = [rng.normal(mean, 0.5, 100) for mean in (-10.0, 0.0, 10.0)]
    return [float(x) for x in np.concatenate(clusters)]


def test_dpm_effective_count_recovers_planted_k() -> None:
    data = _planted_data(3)
    estimator = DirichletProcessMixtureEstimator(
        tuple(GaussianEstimator() for _ in range(10))
    )
    with contextlib.redirect_stdout(io.StringIO()):
        dpm: Optional[Any] = optimize(
            data,
            estimator,
            max_its=200,
            rng=np.random.RandomState(3),
            out=io.StringIO(),
            print_iter=1000,
        )
    assert isinstance(dpm, DirichletProcessMixtureDistribution)
    threshold = 0.5 / len(data)

    # Full truncation: 9 weights + 10 * 2 Gaussian parameters.
    assert dpm.num_params() == 29
    assert dpm.effective_num_components(threshold) == 3
    # 2 weights + 3 * 2 Gaussian parameters.
    assert dpm.effective_num_params(threshold) == 8


def test_dpm_effective_count_matches_get_dpm_mixture_pruning() -> None:
    data = _planted_data(5)
    with contextlib.redirect_stdout(io.StringIO()):
        dpm = optimize(
            data,
            DirichletProcessMixtureEstimator(
                tuple(GaussianEstimator() for _ in range(10))
            ),
            max_its=200,
            rng=np.random.RandomState(5),
            out=io.StringIO(),
            print_iter=1000,
        )
        pruned = get_dpm_mixture(
            data,
            estimator=GaussianEstimator(),
            max_comp=10,
            rng=np.random.RandomState(5),
            max_its=200,
            print_iter=1000,
        )
    assert isinstance(dpm, DirichletProcessMixtureDistribution)
    threshold = 0.5 / len(data)

    assert dpm.effective_num_components(threshold) == len(pruned.components)
    assert dpm.effective_num_params(threshold) == pruned.num_params()
