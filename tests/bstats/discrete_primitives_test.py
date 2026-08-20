"""Regression tests for Bayesian discrete primitive likelihoods."""

from __future__ import annotations

import numpy as np

from dmx.bstats.bernoulli import BernoulliDistribution, BernoulliEstimator
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.categorical import CategoricalDistribution, CategoricalEstimator
from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.geometric import GeometricDistribution
from dmx.bstats.intrange import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)
from dmx.bstats.poisson import PoissonDistribution, PoissonEstimator
from dmx.utils.automatic import (
    get_categorical_estimator,
    get_estimator,
    get_poisson_estimator,
)
from tests.bstats.bstats_tests import (
    BayesianDistributionTestCase,
    BayesianDistributionTests,
)


def _categorical() -> CategoricalDistribution:
    """Create a categorical likelihood with a fixed conjugate prior."""
    return CategoricalDistribution(
        {"red": 0.25, "blue": 0.75},
        name="color",
        prior=DictDirichletDistribution({"red": 2.0, "blue": 3.0}),
    )


def _categorical_prior() -> DictDirichletDistribution:
    """Create an alternate categorical prior."""
    return DictDirichletDistribution({"red": 4.0, "blue": 5.0})


class TestCategoricalDistribution(BayesianDistributionTests):
    """Exercise categorical scoring, sampling, estimation, and encoding."""

    case = BayesianDistributionTestCase(
        distribution_factory=_categorical,
        observations=("red", "blue", "blue", "red"),
        alternate_prior_factory=_categorical_prior,
    )


def _integer_categorical() -> IntegerCategoricalDistribution:
    """Create an integer categorical likelihood with a dense prior."""
    return IntegerCategoricalDistribution(
        [0.2, 0.5, 0.3],
        min_index=-1,
        name="rating",
        prior=DirichletDistribution([2.0, 3.0, 4.0]),
    )


def _integer_prior() -> DirichletDistribution:
    """Create an alternate integer categorical prior."""
    return DirichletDistribution([4.0, 3.0, 2.0])


class TestIntegerCategoricalDistribution(BayesianDistributionTests):
    """Exercise integer categorical scoring, sampling, and estimation."""

    case = BayesianDistributionTestCase(
        distribution_factory=_integer_categorical,
        observations=(-1, 0, 1, 0),
        alternate_prior_factory=_integer_prior,
    )


def _poisson() -> PoissonDistribution:
    """Create a Poisson likelihood with a gamma prior."""
    return PoissonDistribution(
        2.5,
        name="count",
        prior=GammaDistribution(3.0, 0.5),
        keys="count_key",
    )


def _gamma_prior() -> GammaDistribution:
    """Create an alternate Poisson rate prior."""
    return GammaDistribution(4.0, 0.25)


class TestPoissonDistribution(BayesianDistributionTests):
    """Exercise Poisson scoring, sampling, estimation, and encoding."""

    case = BayesianDistributionTestCase(
        distribution_factory=_poisson,
        observations=(0, 1, 3, 5),
        alternate_prior_factory=_gamma_prior,
    )


def _geometric() -> GeometricDistribution:
    """Create a geometric likelihood with a beta prior."""
    return GeometricDistribution(
        0.4,
        name="trials",
        prior=BetaDistribution(2.0, 3.0),
        keys="trials_key",
    )


def _beta_prior() -> BetaDistribution:
    """Create an alternate geometric probability prior."""
    return BetaDistribution(4.0, 5.0)


class TestGeometricDistribution(BayesianDistributionTests):
    """Exercise geometric scoring, sampling, estimation, and encoding."""

    case = BayesianDistributionTestCase(
        distribution_factory=_geometric,
        observations=(1, 2, 4, 3),
        alternate_prior_factory=_beta_prior,
    )


def test_explicit_out_of_support_log_densities() -> None:
    """Return negative infinity for observations outside defined support."""
    categorical = CategoricalDistribution({"a": 0.4, "b": 0.6})
    integer = IntegerCategoricalDistribution([0.4, 0.6], min_index=2)
    count = PoissonDistribution(2.0)
    trials = GeometricDistribution(0.5)

    assert np.isneginf(categorical.log_density("missing"))
    assert np.isneginf(integer.log_density(1))
    assert np.isneginf(integer.log_density(4))
    assert np.isneginf(count.log_density(-1))
    assert np.isneginf(count.log_density(1.5))  # type: ignore[arg-type]
    assert np.isneginf(trials.log_density(0))
    assert np.isneginf(trials.log_density(1.5))  # type: ignore[arg-type]


def test_nonzero_escape_mass_has_scalar_sequence_agreement() -> None:
    """Apply the same escape-mass normalization in both scoring paths."""
    categorical = CategoricalDistribution({"a": 0.4, "b": 0.6}, default_value=0.2)
    integer = IntegerCategoricalDistribution([0.4, 0.6], default_value=0.2, min_index=2)

    categorical_values = ("a", "missing", "b")
    integer_values = (1, 2, 4, 3)
    np.testing.assert_allclose(
        categorical.seq_log_density(categorical.seq_encode(categorical_values)),
        [categorical.log_density(value) for value in categorical_values],
    )
    np.testing.assert_allclose(
        integer.seq_log_density(integer.seq_encode(integer_values)),
        [integer.log_density(value) for value in integer_values],
    )


def test_conjugate_estimators_update_prior_to_posterior() -> None:
    """Update every discrete primitive's conjugate hyperparameters."""
    bernoulli = BernoulliEstimator(prior=BetaDistribution(2.0, 3.0)).estimate(
        (3.0, 1.0)
    )
    categorical = CategoricalEstimator(
        prior=DictDirichletDistribution({"a": 2.0, "b": 3.0})
    ).estimate(({"a": 3.0, "b": 1.0}, 4.0))
    integer = IntegerCategoricalEstimator(
        prior=DirichletDistribution([2.0, 3.0])
    ).estimate((4, np.asarray([3.0, 1.0])))
    count = PoissonEstimator(prior=GammaDistribution(2.0, 0.5)).estimate((4.0, 7.0))
    trials = (
        GeometricDistribution(0.5, prior=BetaDistribution(2.0, 3.0))
        .estimator()
        .estimate((4.0, 10.0))
    )

    assert bernoulli.get_prior().get_parameters() == (5.0, 4.0)
    assert categorical.get_prior().get_parameters() == {"a": 5.0, "b": 4.0}
    np.testing.assert_allclose(integer.get_prior().get_parameters(), [5.0, 4.0])
    assert count.get_prior().get_parameters() == (9.0, 1.0 / 6.0)
    assert trials.get_prior().get_parameters() == (6.0, 9.0)


def test_automatic_bstats_estimator_routes_remain_compatible() -> None:
    """Retain categorical, count, integer, and binary bstats routing."""
    assert isinstance(
        get_categorical_estimator({"a": 2.0}, use_bstats=True),
        CategoricalEstimator,
    )
    assert isinstance(
        get_poisson_estimator({0: 1.0, 1: 2.0}, use_bstats=True),
        PoissonEstimator,
    )
    assert isinstance(get_estimator(["a", "b"], use_bstats=True), CategoricalEstimator)
    assert isinstance(get_estimator([-1, 0, 1], use_bstats=True), CategoricalEstimator)
    assert isinstance(
        get_estimator([True, False], use_bstats=True), CategoricalEstimator
    )
