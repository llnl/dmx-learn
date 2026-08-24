"""Deterministic tests for scalar and continuous Bayesian priors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dmx.bstats import dump_models, load_models
from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.dmvn import DiagonalGaussianDistribution
from dmx.bstats.exponential import ExponentialDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.gaussian import GaussianDistribution
from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.bstats.poisson import PoissonDistribution


def _distributions_and_valid_inputs() -> list[tuple[Any, Any]]:
    """Return fresh prior distributions and representative support points."""
    return [
        (BetaDistribution(2.0, 3.0, name="beta prior"), 0.25),
        (GammaDistribution(2.0, 3.0, name="gamma prior"), 4.0),
        (
            NormalGammaDistribution(0.5, 2.0, 3.0, 4.0, name="normal prior"),
            (0.25, 1.5),
        ),
        (
            MultivariateNormalGammaDistribution(
                np.array([0.5, -1.0]),
                np.array([2.0, 3.0]),
                np.array([3.0, 4.0]),
                np.array([4.0, 5.0]),
                name="vector prior",
            ),
            (np.array([0.25, -0.5]), np.array([1.5, 2.0])),
        ),
    ]


def test_density_matches_documented_parameterizations() -> None:
    """Check densities against direct formulas in the documented conventions."""
    beta_dist = BetaDistribution(2.0, 3.0)
    np.testing.assert_allclose(beta_dist.density(0.25), 1.6875)
    np.testing.assert_allclose(
        beta_dist.log_density(0.25), np.log(beta_dist.density(0.25))
    )

    gamma_dist = GammaDistribution(2.0, 3.0)
    expected_gamma = 4.0 * np.exp(-4.0 / 3.0) / 9.0
    np.testing.assert_allclose(gamma_dist.density(4.0), expected_gamma)

    normal_gamma = NormalGammaDistribution(0.5, 2.0, 3.0, 4.0)
    location, precision = 0.25, 1.5
    gamma_density = 4.0**3 * precision ** (3.0 - 1.0) * np.exp(-4.0 * precision) / 2.0
    normal_density = np.sqrt(2.0 * precision / (2.0 * np.pi)) * np.exp(
        -2.0 * precision * (location - 0.5) ** 2 / 2.0
    )
    np.testing.assert_allclose(
        normal_gamma.density((location, precision)),
        gamma_density * normal_density,
    )

    multivariate = MultivariateNormalGammaDistribution(
        np.array([0.5, -1.0]),
        np.array([2.0, 3.0]),
        np.array([3.0, 4.0]),
        np.array([4.0, 5.0]),
    )
    observation = (np.array([0.25, -0.5]), np.array([1.5, 2.0]))
    coordinate_product = NormalGammaDistribution(0.5, 2.0, 3.0, 4.0).density(
        (0.25, 1.5)
    )
    coordinate_product *= NormalGammaDistribution(-1.0, 3.0, 4.0, 5.0).density(
        (-0.5, 2.0)
    )
    np.testing.assert_allclose(multivariate.density(observation), coordinate_product)


@pytest.mark.parametrize(
    ("distribution", "invalid"),
    [
        (BetaDistribution(2.0, 3.0), -0.1),
        (BetaDistribution(2.0, 3.0), 1.0),
        (GammaDistribution(2.0, 3.0), 0.0),
        (GammaDistribution(2.0, 3.0), -1.0),
        (NormalGammaDistribution(0.0, 1.0, 2.0, 3.0), (0.0, 0.0)),
        (
            MultivariateNormalGammaDistribution(
                np.zeros(2), np.ones(2), np.ones(2), np.ones(2)
            ),
            (np.zeros(2), np.array([1.0, -1.0])),
        ),
    ],
)
def test_invalid_support_has_zero_density(distribution: Any, invalid: Any) -> None:
    """Check the explicit zero-density and negative-infinity support policy."""
    assert distribution.density(invalid) == 0.0
    assert distribution.log_density(invalid) == -np.inf


def test_gamma_sequence_scoring_matches_scalar_support_behavior() -> None:
    """Check encoded gamma scoring for valid and invalid support points."""
    distribution = GammaDistribution(2.0, 3.0)
    observations = [1.0, 4.0, 0.0, -1.0]
    encoded = distribution.seq_encode(observations)
    expected = [distribution.log_density(value) for value in observations]
    np.testing.assert_equal(distribution.seq_log_density(encoded), expected)
    np.testing.assert_equal(distribution.seq_expected_log_density(encoded), expected)


def test_finite_expected_log_density_entropy_and_cross_entropy() -> None:
    """Check inherited plug-in scoring and analytic information measures."""
    for distribution, observation in _distributions_and_valid_inputs():
        assert np.isfinite(distribution.expected_log_density(observation))
        assert np.isfinite(distribution.entropy())
        np.testing.assert_allclose(
            distribution.cross_entropy(distribution),
            distribution.entropy(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_sampling_is_repeatable_and_on_support() -> None:
    """Check fixed-seed draws and the scalar versus batch return contracts."""
    for distribution, _ in _distributions_and_valid_inputs():
        first = distribution.sampler(seed=7).sample(size=5)
        second = distribution.sampler(seed=7).sample(size=5)
        np.testing.assert_equal(first, second)

    assert 0.0 < BetaDistribution(2.0, 3.0).sampler(3).sample() < 1.0
    assert GammaDistribution(2.0, 3.0).sampler(3).sample() > 0.0
    _, precision = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0).sampler(3).sample()
    assert precision > 0.0
    _, precisions = (
        MultivariateNormalGammaDistribution(
            np.zeros(2), np.ones(2), np.ones(2), np.ones(2)
        )
        .sampler(3)
        .sample()
    )
    assert np.all(precisions > 0.0)


def test_string_round_trip_preserves_parameters_and_names() -> None:
    """Check package serialization for scalar and vector parameter values."""
    for distribution, _ in _distributions_and_valid_inputs():
        restored = load_models(dump_models(distribution))
        assert type(restored) is type(distribution)
        assert restored.name == distribution.name
        actual = restored.get_parameters()
        expected = distribution.get_parameters()
        if isinstance(actual, tuple):
            for actual_value, expected_value in zip(actual, expected):
                np.testing.assert_equal(actual_value, expected_value)


def test_parameter_access_and_gamma_cache_refresh() -> None:
    """Check parameter tuples and replacement of cached gamma state."""
    beta_dist = BetaDistribution(2.0, 3.0)
    beta_dist.set_parameters((4.0, 5.0))
    assert beta_dist.get_parameters() == (4.0, 5.0)

    gamma_dist = GammaDistribution(2.0, 3.0)
    gamma_dist.set_parameters((4.0, 5.0))
    assert gamma_dist.get_parameters() == (4.0, 5.0)
    np.testing.assert_allclose(
        gamma_dist.log_density(2.0),
        np.log(gamma_dist.density(2.0)),
    )

    normal_gamma = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0)
    normal_gamma.set_parameters((1.0, 2.0, 3.0, 4.0))
    assert normal_gamma.get_parameters() == (1.0, 2.0, 3.0, 4.0)

    multivariate = MultivariateNormalGammaDistribution(
        np.zeros(2), np.ones(2), np.ones(2), np.ones(2)
    )
    replacement = (
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
        np.array([3.0, 4.0]),
        np.array([4.0, 5.0]),
    )
    multivariate.set_parameters(replacement)
    for actual, expected in zip(multivariate.get_parameters(), replacement):
        np.testing.assert_equal(actual, expected)


def test_downstream_likelihood_default_priors_and_expectations() -> None:
    """Check primitive likelihoods still consume these conjugate priors."""
    bernoulli = BernoulliDistribution(0.4)
    gaussian = GaussianDistribution(0.0, 1.0)
    diagonal = DiagonalGaussianDistribution(np.zeros(2), np.ones(2))
    exponential = ExponentialDistribution(2.0)
    poisson = PoissonDistribution(2.0)

    assert isinstance(bernoulli.get_prior(), BetaDistribution)
    assert isinstance(gaussian.get_prior(), NormalGammaDistribution)
    assert isinstance(diagonal.get_prior(), MultivariateNormalGammaDistribution)
    assert isinstance(exponential.get_prior(), GammaDistribution)
    assert isinstance(poisson.get_prior(), GammaDistribution)

    assert np.isfinite(bernoulli.expected_log_density(True))
    assert np.isfinite(gaussian.expected_log_density(0.5))
    assert np.isfinite(diagonal.expected_log_density(np.array([0.5, -0.5])))
    assert np.isfinite(exponential.expected_log_density(0.5))
    assert np.isfinite(poisson.expected_log_density(2))
