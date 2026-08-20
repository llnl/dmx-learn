"""Regression tests for Bayesian composite product distributions."""

from __future__ import annotations

import numpy as np

from dmx.bstats.bernoulli import BernoulliDistribution, BernoulliEstimator
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.composite import (
    CompositeDataEncoder,
    CompositeDistribution,
    CompositeEncodedData,
    CompositeEstimator,
)
from dmx.bstats.dpm import DirichletProcessMixtureDistribution
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats.normgamma import NormalGammaDistribution


def _distribution(
    name: str = "observation", keys: str | None = "whole-observation"
) -> CompositeDistribution:
    """Create a discrete-continuous product with conjugate priors."""
    return CompositeDistribution(
        (
            BernoulliDistribution(0.7, name="flag"),
            GaussianDistribution(1.0, 2.0, name="value"),
        ),
        name=name,
        keys=keys,
    )


def test_product_and_expected_log_density_preserve_component_order() -> None:
    """Check product scoring and prior expectations sum ordered child scores."""
    distribution = _distribution()
    observation = (True, -0.5)

    expected_log_density = sum(
        component.log_density(value)
        for component, value in zip(distribution.dists, observation)
    )
    expected_prior_log_density = sum(
        component.expected_log_density(value)
        for component, value in zip(distribution.dists, observation)
    )

    np.testing.assert_allclose(
        distribution.log_density(observation), expected_log_density
    )
    np.testing.assert_allclose(
        distribution.expected_log_density(observation), expected_prior_log_density
    )
    assert distribution.get_name() == "observation"


def test_sequence_encoding_matches_scalar_scoring() -> None:
    """Check direct and dedicated encoders retain positional child encodings."""
    distribution = _distribution()
    observations = ((True, -0.5), (False, 1.0), (True, 2.5))
    direct = distribution.seq_encode(observations)
    encoder = distribution.dist_to_encoder()
    wrapped = encoder.seq_encode(observations)

    assert isinstance(encoder, CompositeDataEncoder)
    assert isinstance(wrapped, CompositeEncodedData)
    np.testing.assert_equal(wrapped.data[0], direct[0])
    np.testing.assert_equal(wrapped.data[1], direct[1])

    expected = np.asarray(
        [distribution.log_density(observation) for observation in observations]
    )
    expected_prior = np.asarray(
        [distribution.expected_log_density(observation) for observation in observations]
    )
    np.testing.assert_allclose(distribution.seq_log_density(direct), expected)
    np.testing.assert_allclose(distribution.seq_log_density(wrapped), expected)
    np.testing.assert_allclose(
        distribution.seq_expected_log_density(direct), expected_prior
    )


def test_estimator_wiring_preserves_metadata_and_component_positions() -> None:
    """Check estimator, factory, encoder, and estimate structures stay aligned."""
    distribution = _distribution()
    estimator = distribution.estimator()
    accumulator = estimator.accumulator_factory().make()
    observations = ((True, 0.0), (False, 2.0), (True, 4.0))
    encoded = distribution.seq_encode(observations)
    accumulator.seq_update(encoded, np.ones(len(observations)), distribution)
    fitted = estimator.estimate(accumulator.value())

    assert isinstance(estimator, CompositeEstimator)
    assert estimator.name == distribution.name
    assert estimator.keys == distribution.keys
    assert accumulator.acc_to_encoder() == distribution.dist_to_encoder()
    assert isinstance(fitted.dists[0], BernoulliDistribution)
    assert isinstance(fitted.dists[1], GaussianDistribution)
    assert fitted.name == distribution.name
    assert fitted.keys == distribution.keys
    assert isinstance(fitted.dists[0].get_prior(), BetaDistribution)
    assert isinstance(fitted.dists[1].get_prior(), NormalGammaDistribution)


def test_prior_get_set_and_estimator_propagation() -> None:
    """Check ordered child priors propagate through distributions and estimators."""
    distribution = _distribution()
    original = distribution.get_prior()

    assert isinstance(original, CompositeDistribution)
    assert isinstance(original.dists[0], BetaDistribution)
    assert isinstance(original.dists[1], NormalGammaDistribution)
    assert original.name == distribution.name
    assert original.keys == distribution.keys

    beta_prior = BetaDistribution(3.0, 5.0)
    gaussian_prior = NormalGammaDistribution(0.5, 2.0, 3.0, 4.0)
    alternate = CompositeDistribution((beta_prior, gaussian_prior))
    distribution.set_prior(alternate)

    assert distribution.dists[0].get_prior() is beta_prior
    assert distribution.dists[1].get_prior() is gaussian_prior
    estimator = distribution.estimator()
    assert estimator.get_prior().dists[0] is beta_prior
    assert estimator.get_prior().dists[1] is gaussian_prior

    second_beta = BetaDistribution(4.0, 2.0)
    second_gaussian = NormalGammaDistribution(-1.0, 3.0, 2.0, 5.0)
    estimator.set_prior(CompositeDistribution((second_beta, second_gaussian)))
    assert estimator.get_prior().dists[0] is second_beta
    assert estimator.get_prior().dists[1] is second_gaussian


def test_whole_and_child_keys_merge_and_replace_without_aliasing() -> None:
    """Check whole-product and independent child keys do not double-count."""
    estimator = CompositeEstimator(
        (
            BernoulliEstimator(keys="binary-counts"),
            GaussianEstimator(),
        ),
        keys="whole-counts",
    )
    first = estimator.accumulator_factory().make()
    second = estimator.accumulator_factory().make()
    first.update((True, 1.0), 1.0, None)
    second.update((False, 3.0), 1.0, None)

    shared: dict[str, object] = {}
    first.key_merge(shared)
    second.key_merge(shared)
    first.key_replace(shared)
    second.key_replace(shared)

    expected = ((1.0, 1.0), (4.0, 10.0, 4.0, 2.0, 2.0))
    assert first.value() == expected
    assert second.value() == expected
    binary_accumulator = shared["binary-counts"]
    assert binary_accumulator.value() == (1.0, 1.0)  # type: ignore[attr-defined]


def test_composite_is_usable_inside_mixture_and_dpm_scoring() -> None:
    """Check finite-mixture and DPM containers accept composite base models."""
    first = _distribution(name="first", keys=None)
    second = CompositeDistribution(
        (BernoulliDistribution(0.3), GaussianDistribution(-1.0, 1.5))
    )
    observation = (True, 0.25)

    mixture = MixtureDistribution((first, second), (0.4, 0.6))
    assert np.isfinite(mixture.log_density(observation))

    dpm = DirichletProcessMixtureDistribution(
        (first, second),
        (0.4, 0.6),
        1.0,
        np.ones((2, 2)),
        (first.get_prior(), second.get_prior()),
    )
    assert np.isfinite(dpm.log_density(observation))
