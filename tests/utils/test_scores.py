"""Tests for utils/scores.py: held-out likelihood and information criteria."""

import dataclasses
import io
import math
from typing import Any

import numpy as np
import pytest
from numpy.random import RandomState

from dmx.bstats.dpm import DirichletProcessMixtureDistribution
from dmx.bstats.gaussian import GaussianDistribution as BayesianGaussian
from dmx.bstats.mixture import MixtureDistribution as BayesianMixture
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)
from dmx.utils.estimation import k_fold_split_index, optimize
from dmx.utils.scores import (
    InformationCriterion,
    aic,
    bic,
    heldout_log_likelihood,
    icl,
    kfold_log_likelihood,
)

GAUSSIAN_DATA = [-1.0, 0.0, 2.5]


def _normal_pdf(x: float, mu: float, var: float) -> float:
    """Return the Gaussian density written out by hand."""
    return math.exp(-0.5 * (x - mu) ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def _gaussian_mixture() -> MixtureDistribution:
    """Return a two-component Gaussian mixture with 5 free parameters."""
    return MixtureDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 0.5)], [0.4, 0.6]
    )


def _entropy(rows: list[list[float]]) -> float:
    """Return -sum tau * ln(tau) over a responsibility table."""
    return -sum(t * math.log(t) for row in rows for t in row if t > 0.0)


def test_gaussian_mixture_scores_match_hand_calculation() -> None:
    model = _gaussian_mixture()
    joint = [
        [0.4 * _normal_pdf(x, -1.0, 1.0), 0.6 * _normal_pdf(x, 2.0, 0.5)]
        for x in GAUSSIAN_DATA
    ]
    ll = sum(math.log(a + b) for a, b in joint)
    tau = [[a / (a + b), b / (a + b)] for a, b in joint]
    # 1 weight + 2 * (mean, variance).
    p, n = 5, 3

    assert heldout_log_likelihood(model, GAUSSIAN_DATA).log_likelihood == (
        pytest.approx(ll)
    )
    assert bic(model, GAUSSIAN_DATA).value == pytest.approx(-2 * ll + p * math.log(n))
    assert aic(model, GAUSSIAN_DATA).value == pytest.approx(-2 * ll + 2 * p)
    score = icl(model, GAUSSIAN_DATA)
    assert score.entropy == pytest.approx(_entropy(tau))
    assert score.value == pytest.approx(-2 * ll + p * math.log(n) + 2 * _entropy(tau))


def test_categorical_mixture_scores_match_hand_calculation() -> None:
    model = MixtureDistribution(
        [
            CategoricalDistribution({"a": 0.9, "b": 0.1}),
            CategoricalDistribution({"a": 0.2, "b": 0.8}),
        ],
        [0.3, 0.7],
    )
    data = ["a", "b", "a", "a"]
    joint = {"a": [0.3 * 0.9, 0.7 * 0.2], "b": [0.3 * 0.1, 0.7 * 0.8]}
    ll = sum(math.log(sum(joint[x])) for x in data)
    tau = [[v / sum(joint[x]) for v in joint[x]] for x in data]
    # 1 weight + 1 probability per two-value categorical.
    p, n = 3, 4

    result = bic(model, data)
    assert (result.num_params, result.num_obs) == (p, n)
    assert result.log_likelihood == pytest.approx(ll)
    assert result.value == pytest.approx(-2 * ll + p * math.log(n))
    assert aic(model, data).value == pytest.approx(-2 * ll + 2 * p)
    assert icl(model, data).value == pytest.approx(
        -2 * ll + p * math.log(n) + 2 * _entropy(tau)
    )


def test_heldout_mean_log_likelihood() -> None:
    model = GaussianDistribution(0.0, 1.0)
    score = heldout_log_likelihood(model, [0.0, 1.0])
    expected = math.log(_normal_pdf(0.0, 0.0, 1.0)) + math.log(
        _normal_pdf(1.0, 0.0, 1.0)
    )

    assert score.num_obs == 2
    assert score.mean_log_likelihood == pytest.approx(expected / 2)


def test_kfold_matches_manual_fold_fits() -> None:
    # A Gaussian has a closed-form fit, so each fold can be reproduced by hand.
    data = list(RandomState(1).normal(2.0, 3.0, 23))
    result = kfold_log_likelihood(data, GaussianEstimator(), k=4, seed=7, init_p=1.0)

    folds = k_fold_split_index(len(data), 4, RandomState(7))
    expected = []
    for fold in range(4):
        train = np.asarray([data[i] for i in np.flatnonzero(folds != fold)])
        test = [data[i] for i in np.flatnonzero(folds == fold)]
        mu, var = float(train.mean()), float(train.var())
        expected.append(sum(math.log(_normal_pdf(x, mu, var)) for x in test))

    assert result.k == 4
    assert result.fold_sizes == (6, 6, 6, 5)
    assert result.num_obs == 23
    np.testing.assert_allclose(result.fold_log_likelihoods, expected)
    assert result.log_likelihood == pytest.approx(sum(expected))


def test_kfold_is_reproducible_and_seed_sensitive() -> None:
    data = list(np.concatenate([RandomState(2).normal(m, 1.0, 30) for m in (-4, 4)]))
    estimator = MixtureEstimator([GaussianEstimator()] * 2)

    first = kfold_log_likelihood(data, estimator, k=3, seed=5, max_its=30)
    second = kfold_log_likelihood(data, estimator, k=3, seed=5, max_its=30)
    other = kfold_log_likelihood(data, estimator, k=3, seed=6, max_its=30)

    assert first == second
    assert first.seed == 5
    assert first.fold_log_likelihoods != other.fold_log_likelihoods


@pytest.mark.parametrize("k", [1, 4])
def test_kfold_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be between 2"):
        kfold_log_likelihood([0.0, 1.0, 2.0], GaussianEstimator(), k=k)


def test_empty_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        bic(GaussianDistribution(0.0, 1.0), [])


def test_num_params_override() -> None:
    model = GaussianDistribution(0.0, 1.0)
    default = bic(model, GAUSSIAN_DATA)
    override = bic(model, GAUSSIAN_DATA, num_params=7)

    assert override.num_params == 7
    assert override.value - default.value == pytest.approx(5 * math.log(3))
    with pytest.raises(ValueError, match="nonnegative"):
        aic(model, GAUSSIAN_DATA, num_params=-1)


def test_icl_requires_mixture_posterior() -> None:
    with pytest.raises(TypeError, match="seq_posterior"):
        icl(GaussianDistribution(0.0, 1.0), GAUSSIAN_DATA)


def test_scores_are_frozen_dataclasses() -> None:
    score = bic(_gaussian_mixture(), GAUSSIAN_DATA)
    assert isinstance(score, InformationCriterion)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.value = 0.0  # type: ignore[misc]


def test_bstats_mixture_matches_stats_mixture() -> None:
    bayesian = BayesianMixture(
        [BayesianGaussian(-1.0, 1.0), BayesianGaussian(2.0, 0.5)], [0.4, 0.6]
    )
    for score in (bic, aic, icl):
        assert score(bayesian, GAUSSIAN_DATA).value == pytest.approx(
            score(_gaussian_mixture(), GAUSSIAN_DATA).value
        )


def test_dpm_scored_with_effective_num_params() -> None:
    k = 4
    dpm = DirichletProcessMixtureDistribution(
        [BayesianGaussian(float(i), 1.0) for i in range(k)],
        [0.5, 0.45, 0.04, 0.01],
        1.0,
        np.ones((k, 2)),
        [NormalGammaDistribution(0.0, 1.0, 2.0, 1.0) for _ in range(k)],
    )
    # Two components keep w >= 0.1: 1 weight + 2 * 2 Gaussian parameters.
    effective = dpm.effective_num_params(0.1)
    assert effective == 5
    assert bic(dpm, GAUSSIAN_DATA, num_params=effective).num_params == 5
    assert bic(dpm, GAUSSIAN_DATA).num_params == 11


def _planted_composite_mixture() -> MixtureDistribution:
    """Return a 3-component mixture of (Gaussian, categorical) composites."""
    settings: list[tuple[float, dict[Any, float]]] = [
        (-6.0, {"a": 0.8, "b": 0.1, "c": 0.1}),
        (0.0, {"a": 0.1, "b": 0.8, "c": 0.1}),
        (6.0, {"a": 0.1, "b": 0.1, "c": 0.8}),
    ]
    components = [
        CompositeDistribution(
            [GaussianDistribution(mu, 1.0), CategoricalDistribution(pmap)]
        )
        for mu, pmap in settings
    ]
    return MixtureDistribution(components, [0.3, 0.3, 0.4])


def test_bic_selects_planted_k_on_composite_mixture() -> None:
    data = _planted_composite_mixture().sampler(seed=0).sample(300)
    scores = {}
    for k in range(1, 6):
        component = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
        estimator = MixtureEstimator([component] * k)
        fits = [
            optimize(
                data, estimator, max_its=200, rng=RandomState(s), out=io.StringIO()
            )
            for s in range(3)
        ]
        best = max(fits, key=lambda m: heldout_log_likelihood(m, data).log_likelihood)
        scores[k] = bic(best, data).value

    assert min(scores, key=lambda k: scores[k]) == 3
