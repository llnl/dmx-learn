"""Regression tests for truncated Bayesian DPM estimation."""

from __future__ import annotations

from io import StringIO
from typing import cast

import numpy as np

from dmx.bstats import initialize, seq_encode, seq_estimate
from dmx.bstats.bestimation import optimize
from dmx.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
)
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi
from dmx.utils.automatic import get_dpm_mixture

DATA = (-2.0, -1.8, -1.6, 1.7, 1.9, 2.1)


def _estimator(components: int = 3) -> DirichletProcessMixtureEstimator:
    """Create a small homogeneous Gaussian DPM estimator."""
    return DirichletProcessMixtureEstimator(
        tuple(GaussianEstimator() for _ in range(components))
    )


def test_initialization_is_deterministic_and_normalized() -> None:
    """Check fixed-seed initialization and finite predictive weights."""
    first = cast(
        DirichletProcessMixtureDistribution,
        initialize(DATA, _estimator(), np.random.RandomState(7), 1.0),
    )
    second = cast(
        DirichletProcessMixtureDistribution,
        initialize(DATA, _estimator(), np.random.RandomState(7), 1.0),
    )

    assert isinstance(first, DirichletProcessMixtureDistribution)
    np.testing.assert_allclose(first.w, second.w)
    np.testing.assert_allclose(
        [cast(GaussianDistribution, component).mu for component in first.components],
        [cast(GaussianDistribution, component).mu for component in second.components],
    )
    np.testing.assert_allclose(first.w.sum(), 1.0)
    assert np.all(first.w >= 0.0)
    assert np.all(np.isfinite(first.w))


def test_one_step_update_changes_variational_parameters() -> None:
    """Check one coordinate update preserves shape and changes beta parameters."""
    estimator = _estimator()
    initial = cast(
        DirichletProcessMixtureDistribution,
        initialize(DATA, estimator, np.random.RandomState(11), 1.0),
    )
    encoded = seq_encode(DATA, initial)
    updated = seq_estimate(encoded, estimator, initial)

    assert isinstance(updated, DirichletProcessMixtureDistribution)
    assert updated.g.shape == (3, 2)
    assert not np.allclose(updated.g, initial.g)
    np.testing.assert_allclose(updated.w.sum(), 1.0)
    assert np.all(updated.w >= 0.0)


def test_estimation_sorts_components_and_priors_by_count() -> None:
    """Check decreasing counts reorder each component and its original prior."""
    first_prior = NormalGammaDistribution(-5.0, 1.0, 2.0, 2.0)
    second_prior = NormalGammaDistribution(5.0, 1.0, 2.0, 2.0)
    estimator = DirichletProcessMixtureEstimator(
        (
            GaussianEstimator(prior=first_prior),
            GaussianEstimator(prior=second_prior),
        )
    )
    sufficient_statistics = (
        np.asarray([1.0, 3.0]),
        np.zeros((2, 2)),
        1.0,
        0.0,
        (
            (-5.0, 25.0, -5.0, 1.0, 1.0),
            (12.0, 50.0, 12.0, 3.0, 3.0),
        ),
    )

    model = estimator.estimate(sufficient_statistics)

    first = cast(GaussianDistribution, model.components[0])
    second = cast(GaussianDistribution, model.components[1])
    assert first.mu > second.mu
    assert model.component_priors == (second_prior, first_prior)
    assert model.w[0] > model.w[1]


def test_expected_scores_and_optimizer_output_are_finite() -> None:
    """Check expected scoring and documented convergence values are finite."""
    output = StringIO()
    model = optimize(
        DATA,
        _estimator(),
        max_its=1,
        delta=None,
        init_p=1.0,
        rng=np.random.RandomState(13),
        out=output,
        print_iter=1,
    )
    encoded = model.seq_encode(DATA)

    assert np.isfinite(model.expected_log_density(0.0))
    assert np.all(np.isfinite(model.seq_expected_log_density(encoded)))
    assert np.all(np.isfinite(model.seq_log_density(encoded)))
    assert all(label in output.getvalue() for label in ("LL=", "MLL=", "VLL="))


def test_automatic_conversion_and_mpi_import_remain_compatible() -> None:
    """Check local conversion returns a finite mixture and MPI helper imports."""
    mixture = get_dpm_mixture(
        DATA,
        estimator=GaussianEstimator(),
        max_comp=3,
        rng=np.random.RandomState(17),
        max_its=1,
        print_iter=2,
        mix_threshold_count=0.0,
    )

    assert isinstance(mixture, MixtureDistribution)
    assert callable(get_dpm_mixture_mpi)
    np.testing.assert_allclose(mixture.w.sum(), 1.0)
    assert np.all(mixture.w >= 0.0)
    assert np.all(np.isfinite(mixture.seq_log_density(mixture.seq_encode(DATA))))
