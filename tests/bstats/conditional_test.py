"""Regression tests for keyed Bayesian conditional distributions."""

from __future__ import annotations

import numpy as np

from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.categorical import CategoricalDistribution
from dmx.bstats.conditional import (
    ConditionalDataEncoder,
    ConditionalDistribution,
    ConditionalDistributionAccumulatorFactory,
    ConditionalDistributionEstimator,
    ConditionalEncodedData,
)


def _condition_model(value: str = "known") -> CategoricalDistribution:
    """Create a deterministic distribution for sampled condition keys."""
    return CategoricalDistribution({value: 1.0})


def test_known_and_default_conditions_use_the_selected_child() -> None:
    """Score explicit keys directly and absent keys through the default."""
    distribution = ConditionalDistribution(
        {"known": BernoulliDistribution(0.8)},
        _condition_model(),
        default_dist=BernoulliDistribution(0.25),
    )

    assert distribution.log_density(("known", True)) == np.log(0.8)
    assert distribution.log_density(("missing", True)) == np.log(0.25)

    without_default = ConditionalDistribution(
        {"known": BernoulliDistribution(0.8)}, _condition_model()
    )
    assert without_default.log_density(("missing", True)) == -np.inf


def test_sequence_encoding_has_stable_aligned_five_part_shape() -> None:
    """Keep condition groups, child data, and original indices aligned."""
    distribution = ConditionalDistribution(
        {"known": BernoulliDistribution(0.8)},
        _condition_model(),
        default_dist=BernoulliDistribution(0.25),
    )
    observations = (("known", True), ("missing", False), ("known", False))
    direct = distribution.seq_encode(observations)
    wrapped = distribution.dist_to_encoder().seq_encode(observations)

    assert isinstance(wrapped, ConditionalEncodedData)
    assert len(wrapped.data) == 5
    size, conditions, encoded_values, indices, encoded_conditions = wrapped.data
    assert size == 3
    assert conditions == ("known", "missing")
    assert len(conditions) == len(encoded_values) == len(indices)
    np.testing.assert_array_equal(indices[0], [0, 2])
    np.testing.assert_array_equal(indices[1], [1])
    assert encoded_conditions is not None

    expected = np.asarray(
        [distribution.log_density(observation) for observation in observations]
    )
    np.testing.assert_allclose(distribution.seq_log_density(direct), expected)
    np.testing.assert_allclose(distribution.seq_log_density(wrapped), expected)

    no_default = ConditionalDistribution(
        {"known": BernoulliDistribution(0.8)}, _condition_model()
    )
    missing_encoding = no_default.dist_to_encoder().seq_encode(observations)
    assert missing_encoding.data[1] == ("known", "missing")
    assert missing_encoding.data[2][1] is None
    assert no_default.seq_log_density(missing_encoding)[1] == -np.inf


def test_sampler_routes_known_and_missing_conditions_repeatably() -> None:
    """Draw from explicit children and use the default for an absent key."""
    explicit = ConditionalDistribution(
        {"known": BernoulliDistribution(1.0)}, _condition_model("known")
    )
    assert explicit.sampler(seed=4).sample() == ("known", True)
    assert explicit.sampler(seed=4).sample(size=3) == [("known", True)] * 3

    fallback = ConditionalDistribution(
        {"known": BernoulliDistribution(1.0)},
        _condition_model("missing"),
        default_dist=BernoulliDistribution(0.0),
    )
    assert fallback.sampler(seed=4).sample() == ("missing", False)


def test_estimator_factory_and_default_accumulator_stay_wired() -> None:
    """Fit explicit and default children without replacing the condition model."""
    condition_model = _condition_model()
    distribution = ConditionalDistribution(
        {"known": BernoulliDistribution(0.6)},
        condition_model,
        default_dist=BernoulliDistribution(0.4),
    )
    estimator = distribution.estimator()
    factory = estimator.accumulator_factory()
    accumulator = factory.make()
    observations = (("known", True), ("known", False), ("missing", True))
    encoded = distribution.dist_to_encoder().seq_encode(observations)
    accumulator.seq_update(encoded, np.ones(len(observations)), distribution)
    fitted = estimator.estimate(accumulator.value())

    assert isinstance(estimator, ConditionalDistributionEstimator)
    assert isinstance(factory, ConditionalDistributionAccumulatorFactory)
    assert isinstance(accumulator.acc_to_encoder(), ConditionalDataEncoder)
    assert accumulator.acc_to_encoder() == distribution.dist_to_encoder()
    assert fitted.cond_dist is condition_model
    assert isinstance(fitted.dmap["known"], BernoulliDistribution)
    assert isinstance(fitted.default_dist, BernoulliDistribution)
    assert fitted.has_default
