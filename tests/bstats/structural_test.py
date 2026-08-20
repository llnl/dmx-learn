"""Tests for optional, sequence, and set-like Bayesian wrappers."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.optional import OptionalDistribution, OptionalEstimator
from dmx.bstats.poisson import PoissonDistribution
from dmx.bstats.sequence import SequenceDistribution, SequenceEstimator
from dmx.bstats.setdist import BernoulliSetDistribution
from dmx.utils.automatic import get_estimator


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    """Compare sufficient-statistic values containing mappings and arrays."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual == expected
    elif isinstance(actual, tuple) and isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    elif actual is None or expected is None:
        assert actual is expected
    else:
        np.testing.assert_allclose(actual, expected)


def _assert_accumulator_paths(distribution: Any, observations: list[Any]) -> None:
    """Check scalar, sequence, and restored accumulator values agree."""
    factory = distribution.estimator().accumulator_factory()
    scalar = factory.make()
    for observation in observations:
        scalar.update(observation, 1.0, distribution)

    sequence = factory.make()
    sequence.seq_update(
        distribution.seq_encode(observations),
        np.ones(len(observations)),
        distribution,
    )
    _assert_nested_equal(sequence.value(), scalar.value())

    restored = factory.make()
    restored.from_value(copy.deepcopy(scalar.value()))
    _assert_nested_equal(restored.value(), scalar.value())


def test_optional_missing_and_present_scoring_encoding_and_statistics() -> None:
    """Check ``None`` and NaN markers without changing identity semantics."""
    distribution = OptionalDistribution(BernoulliDistribution(0.75), p=0.25)
    observations = [None, True, False, None]
    encoded = distribution.seq_encode(observations)
    expected = [distribution.log_density(value) for value in observations]
    np.testing.assert_allclose(distribution.seq_log_density(encoded), expected)
    assert encoded[1].tolist() == [1, 2]
    assert encoded[2].tolist() == [0, 3]
    _assert_accumulator_paths(distribution, observations)

    nan_distribution = OptionalDistribution(
        BernoulliDistribution(0.5), p=0.4, missing_value=np.nan
    )
    assert nan_distribution.log_density(float("nan")) == nan_distribution.log_p0
    assert np.isfinite(nan_distribution.log_density(True))


def test_optional_estimator_and_child_prior_propagation() -> None:
    """Check estimator wiring and both parts of the composite prior."""
    distribution = OptionalDistribution(BernoulliDistribution(0.6))
    estimator = distribution.estimator()
    assert isinstance(estimator, OptionalEstimator)

    prior = CompositeDistribution((BetaDistribution(2, 3), BetaDistribution(4, 5)))
    distribution.set_prior(prior)
    assert distribution.prior.get_parameters() == (2, 3)
    assert distribution.dist.get_prior().get_parameters() == (4, 5)

    estimator.set_prior(prior)
    assert estimator.prior.get_parameters() == (2, 3)
    assert estimator.estimator.get_prior().get_parameters() == (4, 5)
    assert (
        distribution.dist_to_encoder()
        == estimator.accumulator_factory().make().acc_to_encoder()
    )


def test_variable_length_sequence_scoring_encoding_and_statistics() -> None:
    """Check empty and unequal-length lists through scalar and encoded paths."""
    distribution = SequenceDistribution(
        BernoulliDistribution(0.7), PoissonDistribution(1.5)
    )
    observations = [[], [True], [False, True, True]]
    encoded = distribution.seq_encode(observations)
    expected = [distribution.log_density(value) for value in observations]
    np.testing.assert_allclose(distribution.seq_log_density(encoded), expected)
    assert encoded[0].tolist() == [1, 2, 2, 2]
    np.testing.assert_allclose(encoded[1], [0.0, 1.0, 1.0 / 3.0])
    _assert_accumulator_paths(distribution, observations)
    assert (
        distribution.dist_to_encoder()
        == distribution.estimator().accumulator_factory().make().acc_to_encoder()
    )


def test_sequence_estimator_and_child_prior_propagation() -> None:
    """Check sequence estimator wiring and ordered composite priors."""
    distribution = SequenceDistribution(
        BernoulliDistribution(0.5), PoissonDistribution(2.0)
    )
    estimator = distribution.estimator()
    assert isinstance(estimator, SequenceEstimator)

    prior = CompositeDistribution((BetaDistribution(3, 4), GammaDistribution(5, 2)))
    distribution.set_prior(prior)
    estimator.set_prior(prior)
    assert distribution.dist.get_prior().get_parameters() == (3, 4)
    assert distribution.len_dist is not None
    assert distribution.len_dist.get_prior().get_parameters() == (5, 2)
    assert estimator.estimator.get_prior().get_parameters() == (3, 4)
    assert estimator.len_estimator is not None
    assert estimator.len_estimator.get_prior().get_parameters() == (5, 2)


def test_set_observations_are_order_independent_and_round_trip_statistics() -> None:
    """Check set-like scoring without relying on sampled label order."""
    distribution = BernoulliSetDistribution(
        {"a": 0.2, "b": 0.7, "c": 0.4}, prior=BetaDistribution(2, 2)
    )
    assert distribution.log_density(["a", "c"]) == distribution.log_density(["c", "a"])
    observations = [[], ["a"], ["c", "b"], ["a", "b"]]
    expected = [distribution.log_density(value) for value in observations]
    np.testing.assert_allclose(
        distribution.seq_log_density(distribution.seq_encode(observations)), expected
    )
    _assert_accumulator_paths(distribution, observations)
    assert (
        distribution.dist_to_encoder()
        == distribution.estimator().accumulator_factory().make().acc_to_encoder()
    )
    assert distribution.estimator().get_prior() is distribution.get_prior()


def test_automatic_optional_and_sequence_routing_remains_compatible() -> None:
    """Check bstats automatic inference still selects structural estimators."""
    optional = get_estimator([None, "a", "b"], use_bstats=True)
    sequence = get_estimator([["a"], ["a", "b"]], use_bstats=True)
    assert isinstance(optional, OptionalEstimator)
    assert isinstance(sequence, SequenceEstimator)
