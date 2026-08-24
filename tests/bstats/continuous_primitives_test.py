"""Regression tests for Bayesian continuous primitive likelihoods."""

from __future__ import annotations

from typing import Any

import numpy as np

from dmx.bstats.dirac import DiracDistribution, DiracEstimator
from dmx.bstats.dmvn import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from dmx.bstats.exponential import ExponentialDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.bstats.nulldist import null_dist
from dmx.utils.automatic import get_gaussian_estimator
from tests.bstats.bstats_tests import (
    BayesianDistributionTestCase,
    BayesianDistributionTests,
)

_NO_INFORMATION_MEASURES = {
    "entropy": "likelihood information measures are not implemented",
    "cross_entropy": "likelihood information measures are not implemented",
}


class TestExponentialDistribution(BayesianDistributionTests):
    """Exercise exponential scoring, sampling, estimators, and priors."""

    case = BayesianDistributionTestCase(
        distribution_factory=lambda: ExponentialDistribution(2.0, name="waiting"),
        observations=(0.0, 0.25, 1.5),
        alternate_prior_factory=lambda: GammaDistribution(3.0, 0.5),
        unsupported_methods=_NO_INFORMATION_MEASURES,
    )


class TestGaussianDistribution(BayesianDistributionTests):
    """Exercise scalar Gaussian scoring, sampling, estimators, and priors."""

    case = BayesianDistributionTestCase(
        distribution_factory=lambda: GaussianDistribution(1.0, 2.5, name="measurement"),
        observations=(-1.0, 0.5, 3.0),
        alternate_prior_factory=lambda: NormalGammaDistribution(0.5, 2.0, 3.0, 4.0),
        unsupported_methods=_NO_INFORMATION_MEASURES,
    )


class TestDiagonalGaussianDistribution(BayesianDistributionTests):
    """Exercise diagonal Gaussian vector paths and conjugate priors."""

    case = BayesianDistributionTestCase(
        distribution_factory=lambda: DiagonalGaussianDistribution(
            [1.0, -1.0], [2.0, 0.5], name="vector"
        ),
        observations=(
            np.array([0.0, -1.0]),
            np.array([1.5, 0.0]),
            np.array([3.0, -2.0]),
        ),
        alternate_prior_factory=lambda: MultivariateNormalGammaDistribution(
            np.array([0.5, -0.5]),
            np.array([2.0, 3.0]),
            np.array([3.0, 4.0]),
            np.array([4.0, 5.0]),
        ),
        unsupported_methods=_NO_INFORMATION_MEASURES,
    )


class TestDiracDistribution(BayesianDistributionTests):
    """Exercise fixed point-mass behavior and empty statistics."""

    case = BayesianDistributionTestCase(
        distribution_factory=lambda: DiracDistribution(2.5),
        observations=(2.5, 2.5, 2.5),
        alternate_prior_factory=lambda: GammaDistribution(2.0, 3.0),
        unsupported_methods={
            **_NO_INFORMATION_MEASURES,
            "string_round_trip": (
                "DiracDistribution is intentionally not a package-level export"
            ),
        },
    )


def test_density_formulas_and_support() -> None:
    """Check finite valid scores and explicit behavior outside support."""
    exponential = ExponentialDistribution(2.0)
    np.testing.assert_allclose(exponential.log_density(0.5), np.log(2.0) - 1.0)
    assert exponential.log_density(-0.5) == -np.inf

    gaussian = GaussianDistribution(1.0, 2.0)
    expected = -0.5 * np.log(4.0 * np.pi) - 0.25
    np.testing.assert_allclose(gaussian.log_density(2.0), expected)

    diagonal = DiagonalGaussianDistribution([1.0, -1.0], [2.0, 0.5])
    assert np.isfinite(diagonal.log_density(np.array([0.0, 0.0])))
    assert diagonal.log_density(np.array([0.0])) == -np.inf

    point_mass = DiracDistribution("fixed")
    assert point_mass.density("fixed") == 1.0
    assert point_mass.log_density("other") == -np.inf


def test_estimators_update_conjugate_posteriors() -> None:
    """Check sufficient statistics update each documented prior family."""
    exponential = ExponentialDistribution(2.0, prior=GammaDistribution(2.0, 0.5))
    exponential_fit = exponential.estimator().estimate((3.0, 4.0))
    assert isinstance(exponential_fit.get_prior(), GammaDistribution)
    assert exponential_fit.get_prior().get_parameters() == (5.0, 1.0 / 6.0)
    np.testing.assert_allclose(exponential_fit.get_parameters(), 4.0 / 6.0)

    gaussian_prior = NormalGammaDistribution(0.0, 2.0, 3.0, 4.0)
    gaussian_fit = GaussianEstimator(prior=gaussian_prior).estimate(
        (4.0, 10.0, 4.0, 2.0, 2.0)
    )
    posterior = gaussian_fit.get_prior()
    assert isinstance(posterior, NormalGammaDistribution)
    np.testing.assert_allclose(posterior.get_parameters(), (1.0, 4.0, 4.0, 7.0))
    np.testing.assert_allclose(gaussian_fit.get_parameters(), (1.0, 2.0))

    vector_prior = MultivariateNormalGammaDistribution(
        np.zeros(2), np.full(2, 2.0), np.full(2, 3.0), np.full(2, 4.0)
    )
    diagonal_fit = DiagonalGaussianEstimator(dim=2, prior=vector_prior).estimate(
        (np.array([4.0, 2.0]), np.array([10.0, 4.0]), 2.0)
    )
    vector_posterior = diagonal_fit.get_prior()
    assert isinstance(vector_posterior, MultivariateNormalGammaDistribution)
    expected_parameters = (
        np.array([1.0, 0.5]),
        np.array([4.0, 4.0]),
        np.array([4.0, 4.0]),
        np.array([7.0, 5.5]),
    )
    for actual, expected_value in zip(
        vector_posterior.get_parameters(), expected_parameters
    ):
        np.testing.assert_allclose(actual, expected_value)


def test_nonconjugate_expected_density_uses_plugin_parameters() -> None:
    """Check alternate priors do not break expected-density scoring."""
    distributions_and_observations: list[tuple[Any, Any]] = [
        (ExponentialDistribution(2.0, prior=null_dist), 0.5),
        (GaussianDistribution(1.0, 2.0, prior=null_dist), 0.5),
        (
            DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0], prior=null_dist),
            [1.0, 0.0],
        ),
    ]
    for distribution, observation in distributions_and_observations:
        np.testing.assert_allclose(
            distribution.expected_log_density(observation),
            distribution.log_density(observation),
        )


def test_estimator_wiring_and_automatic_compatibility() -> None:
    """Check estimator construction used by distributions and automatic routing."""
    gaussian = GaussianDistribution(0.0, 1.0, name="x")
    gaussian_estimator = gaussian.estimator()
    assert isinstance(gaussian_estimator, GaussianEstimator)
    assert gaussian_estimator.name == "x"
    assert gaussian_estimator.get_prior() is gaussian.get_prior()

    diagonal = DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0], name="xy")
    diagonal_estimator = diagonal.estimator()
    assert isinstance(diagonal_estimator, DiagonalGaussianEstimator)
    assert diagonal_estimator.dim == 2
    assert diagonal_estimator.name == "xy"
    assert diagonal_estimator.get_prior() is diagonal.get_prior()

    automatic = get_gaussian_estimator({-1.0: 1.0, 1.0: 1.0}, use_bstats=True)
    assert isinstance(automatic, GaussianEstimator)
    assert isinstance(DiagonalGaussianEstimator(dim=2), DiagonalGaussianEstimator)
    assert isinstance(DiracEstimator("fixed").estimate(None), DiracDistribution)
