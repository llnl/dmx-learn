"""Regression tests for finite Bayesian mixture distributions."""

from __future__ import annotations

import numpy as np

from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.gaussian import GaussianDistribution
from dmx.bstats.mixture import (
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEstimator,
    MixtureEstimatorAccumulator,
)
from dmx.bstats.normgamma import NormalGammaDistribution


def _primitive_mixture() -> MixtureDistribution:
    """Create a two-component Gaussian mixture."""
    return MixtureDistribution(
        (GaussianDistribution(-2.0, 0.5), GaussianDistribution(3.0, 2.0)),
        (0.25, 0.75),
        name="values",
        prior=DirichletDistribution([2.0, 4.0]),
    )


def _composite_mixture() -> MixtureDistribution:
    """Create a mixture over ordered discrete-continuous products."""
    return MixtureDistribution(
        (
            CompositeDistribution(
                (BernoulliDistribution(0.8), GaussianDistribution(-1.0, 1.0))
            ),
            CompositeDistribution(
                (BernoulliDistribution(0.2), GaussianDistribution(2.0, 1.5))
            ),
        ),
        (0.4, 0.6),
    )


def test_log_sum_exp_scoring_is_stable() -> None:
    """Check scalar scoring against a stable component-wise calculation."""
    distribution = MixtureDistribution(
        (GaussianDistribution(-1000.0, 1.0), GaussianDistribution(1000.0, 1.0)),
        (0.25, 0.75),
    )
    observation = 0.0
    terms = np.asarray(
        [
            component.log_density(observation) + np.log(weight)
            for component, weight in zip(distribution.components, distribution.w)
        ]
    )

    assert np.isfinite(distribution.log_density(observation))
    np.testing.assert_allclose(
        distribution.log_density(observation), np.logaddexp.reduce(terms)
    )


def test_constructor_normalizes_filtered_component_weights() -> None:
    """Check helper-produced retained weight mass is normalized on input."""
    distribution = MixtureDistribution(
        (GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)),
        (0.2, 0.6),
    )

    np.testing.assert_allclose(distribution.w, (0.25, 0.75))
    np.testing.assert_allclose(distribution.w.sum(), 1.0)


def test_sequence_scoring_and_encoder_match_scalar_scores() -> None:
    """Check raw and wrapped encodings preserve scalar mixture scoring."""
    distribution = _primitive_mixture()
    observations = (-4.0, -1.0, 0.0, 3.0, 8.0)
    direct = distribution.seq_encode(observations)
    encoder = distribution.dist_to_encoder()
    wrapped = encoder.seq_encode(observations)
    expected = [distribution.log_density(value) for value in observations]

    assert isinstance(encoder, MixtureDataEncoder)
    np.testing.assert_allclose(distribution.seq_log_density(direct), expected)
    np.testing.assert_allclose(distribution.seq_log_density(wrapped), expected)
    assert distribution.seq_component_log_density(direct).shape == (5, 2)


def test_sampler_is_repeatable_for_primitive_and_composite_components() -> None:
    """Check fixed seeds repeat component choices and child samples."""
    for distribution in (_primitive_mixture(), _composite_mixture()):
        first = distribution.sampler(seed=19).sample(size=30)
        second = distribution.sampler(seed=19).sample(size=30)
        np.testing.assert_equal(first, second)


def test_posterior_responsibility_shapes_and_normalization() -> None:
    """Check scalar and sequence responsibilities retain component order."""
    distribution = _composite_mixture()
    observations = ((True, -1.0), (False, 2.0), (True, 0.5))
    scalar = distribution.posterior(observations[0])
    sequence = distribution.seq_posterior(distribution.seq_encode(observations))

    assert scalar.shape == (2,)
    assert sequence.shape == (3, 2)
    np.testing.assert_allclose(scalar.sum(), 1.0)
    np.testing.assert_allclose(sequence.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(sequence[0], scalar)


def test_estimator_factory_wiring_and_normalized_estimates() -> None:
    """Check estimator order, encoder wiring, and estimated weight normalization."""
    distribution = _primitive_mixture()
    estimator = distribution.estimator()
    accumulator = estimator.accumulator_factory().make()
    observations = (-2.5, -2.0, 2.0, 3.0, 4.0)
    encoded = distribution.seq_encode(observations)
    accumulator.seq_update(encoded, np.ones(len(observations)), distribution)
    fitted = estimator.estimate(accumulator.value())

    assert isinstance(estimator, MixtureEstimator)
    assert isinstance(accumulator, MixtureEstimatorAccumulator)
    assert accumulator.acc_to_encoder() == distribution.dist_to_encoder()
    assert fitted.name == distribution.name
    np.testing.assert_allclose(fitted.w.sum(), 1.0)
    assert np.all(fitted.w >= 0.0)

    fixed = MixtureEstimator(estimator.estimators, fixed_w=(0.1, 0.9))
    fixed_fitted = fixed.estimate(accumulator.value())
    np.testing.assert_allclose(fixed_fitted.w, (0.1, 0.9))
    np.testing.assert_allclose(fixed_fitted.w.sum(), 1.0)


def test_component_priors_propagate_positionally() -> None:
    """Check complete priors reach matching distribution and estimator children."""
    distribution = _primitive_mixture()
    first_prior = NormalGammaDistribution(-4.0, 2.0, 3.0, 5.0)
    second_prior = NormalGammaDistribution(4.0, 3.0, 2.0, 6.0)
    complete_prior = CompositeDistribution(
        (
            DirichletDistribution([5.0, 7.0]),
            CompositeDistribution((first_prior, second_prior)),
        )
    )
    distribution.set_prior(complete_prior)

    assert distribution.components[0].get_prior() is first_prior
    assert distribution.components[1].get_prior() is second_prior
    estimator = distribution.estimator()
    replacement_first = NormalGammaDistribution(-1.0, 4.0, 2.0, 3.0)
    replacement_second = NormalGammaDistribution(1.0, 5.0, 3.0, 4.0)
    estimator.set_prior(
        CompositeDistribution(
            (
                DirichletDistribution([3.0, 6.0]),
                CompositeDistribution((replacement_first, replacement_second)),
            )
        )
    )

    prior = estimator.get_prior()
    assert isinstance(prior, CompositeDistribution)
    assert isinstance(prior.dists[1], CompositeDistribution)
    assert prior.dists[1].dists[0] is replacement_first
    assert prior.dists[1].dists[1] is replacement_second
    assert isinstance(prior.dists[1].dists[0], NormalGammaDistribution)
    assert not isinstance(prior.dists[1].dists[0], BetaDistribution)
