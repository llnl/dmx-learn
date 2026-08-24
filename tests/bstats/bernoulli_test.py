"""Exercise the shared Bayesian harness with a Bernoulli distribution."""

from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.beta import BetaDistribution
from tests.bstats.bstats_tests import (
    BayesianDistributionTestCase,
    BayesianDistributionTests,
)


def _make_distribution() -> BernoulliDistribution:
    """Create the representative distribution used by the shared checks."""
    return BernoulliDistribution(
        0.7,
        name="coin",
        prior=BetaDistribution(2.0, 3.0),
        keys="coin_key",
    )


def _make_alternate_prior() -> BetaDistribution:
    """Create a distinct conjugate prior for the prior replacement check."""
    return BetaDistribution(4.0, 5.0)


class TestBernoulliDistribution(BayesianDistributionTests):
    """Run every shared Bayesian distribution check without custom methods."""

    case = BayesianDistributionTestCase(
        distribution_factory=_make_distribution,
        observations=(True, False, True, True),
        alternate_prior_factory=_make_alternate_prior,
    )
