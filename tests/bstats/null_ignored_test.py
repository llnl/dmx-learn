"""Exercise neutral and fixed Bayesian distributions with the shared harness."""

import numpy as np

from dmx.bstats import (
    IgnoredDistribution,
    IgnoredEstimator,
    IgnoredSampler,
    NullDistribution,
    NullEstimator,
    NullSampler,
)
from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.ignored import IgnoredAccumulator, IgnoredDataEncoder
from dmx.bstats.nulldist import NullDataEncoder, null_dist
from tests.bstats.bstats_tests import (
    BayesianDistributionTestCase,
    BayesianDistributionTests,
)


def _make_null_distribution() -> NullDistribution:
    """Create a neutral distribution for shared checks."""
    return NullDistribution()


def _make_ignored_distribution() -> IgnoredDistribution:
    """Create a fixed Bernoulli distribution for shared checks."""
    return IgnoredDistribution(
        BernoulliDistribution(
            0.7,
            prior=BetaDistribution(2.0, 3.0),  # type: ignore[abstract, arg-type]
        )
    )


def _make_alternate_prior() -> BetaDistribution:
    """Create a distinct prior for delegation checks."""
    return BetaDistribution(4.0, 5.0)  # type: ignore[abstract]


class TestNullDistribution(BayesianDistributionTests):
    """Run shared distribution checks for the neutral factor."""

    case = BayesianDistributionTestCase(
        distribution_factory=_make_null_distribution,
        observations=(None, 1, 2, 3),
        unsupported_methods={
            "set_prior": "a null prior is intentionally immutable and parameter-free"
        },
    )

    def test_public_wiring(self) -> None:
        """Keep exported constructors and neutral singleton behavior compatible."""
        distribution = NullDistribution()

        assert isinstance(distribution.sampler(), NullSampler)
        assert isinstance(distribution.estimator(), NullEstimator)
        assert isinstance(distribution.dist_to_encoder(), NullDataEncoder)
        assert distribution.estimator().estimate(None) is null_dist
        assert distribution.estimator().estimate(None, None) is null_dist

        encoded = distribution.dist_to_encoder().seq_encode(self.case.observations)
        np.testing.assert_equal(distribution.seq_log_density(encoded), np.zeros(4))

    def test_sampler_behavior(self) -> None:
        """Return placeholders for scalar and sized sampling calls."""
        sampler = NullSampler(seed=3)

        assert sampler.sample() is None
        assert sampler.sample(size=3) == [None, None, None]


class TestIgnoredDistribution(BayesianDistributionTests):
    """Run shared distribution checks for a fixed wrapped model."""

    case = BayesianDistributionTestCase(
        distribution_factory=_make_ignored_distribution,
        observations=(True, False, True, True),
        alternate_prior_factory=_make_alternate_prior,
    )

    def test_public_wiring(self) -> None:
        """Expose the established sampler, estimator, accumulator, and encoder roles."""
        distribution = _make_ignored_distribution()
        estimator = distribution.estimator()
        accumulator = estimator.accumulator_factory().make()

        assert isinstance(distribution.sampler(), IgnoredSampler)
        assert isinstance(estimator, IgnoredEstimator)
        assert isinstance(accumulator, IgnoredAccumulator)
        assert isinstance(distribution.dist_to_encoder(), IgnoredDataEncoder)

        encoded = distribution.dist_to_encoder().seq_encode(self.case.observations)
        np.testing.assert_allclose(
            distribution.seq_log_density(encoded),
            distribution.seq_log_density(
                distribution.seq_encode(self.case.observations)
            ),
        )

    def test_estimation_keeps_wrapped_distribution_fixed(self) -> None:
        """Discard observations and return the identical wrapped distribution."""
        distribution = _make_ignored_distribution()
        estimator = distribution.estimator()
        accumulator = estimator.accumulator_factory().make()

        accumulator.update(False, 100.0, distribution)
        encoded = distribution.seq_encode((False, False, False))
        accumulator.seq_update(encoded, np.ones(3), distribution)
        accumulator.from_value(None)
        estimate = estimator.estimate(None)

        assert estimate.dist is distribution.dist
        assert estimate.get_parameters() == distribution.get_parameters()


def test_ignored_default_remains_a_neutral_fixed_distribution() -> None:
    """Keep the historical null-wrapped ignored default fully operational."""
    distribution = IgnoredDistribution()

    assert distribution.log_density("unused") == 0.0
    assert distribution.sampler(seed=2).sample(size=2) == [None, None]
    assert distribution.estimator().estimate(None, None).dist is null_dist


def test_null_prior_remains_usable_as_a_default() -> None:
    """Recognize a null prior as absence of Bayesian regularization."""
    distribution = BernoulliDistribution(0.5, prior=NullDistribution())

    assert getattr(distribution, "has_prior") is False
