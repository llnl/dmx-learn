"""Fisher scores of mixtures used as components (latent-variable families).

A nested mixture's score is the observed-data score scaled by the
complete-data Fisher information and then whitened with draws from the model.
These tests pin the raw formula, the model-only dependence of the whitening,
field slices, nesting in a composite, and unsupported components.
"""

import pickle
import warnings
from typing import Any, Optional

import numpy as np
import pytest

from dmx.stats import (
    CompositeDistribution,
    GammaDistribution,
    GaussianDistribution,
    HeterogeneousMixtureDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from dmx.stats._fisher_latent import _raw_mixture_score
from dmx.stats.pdist import ProbabilityDistribution
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherScoreOptions,
    component_score_dims,
    has_fisher_score,
)
from tests.fisher_contract import check_moments

DEFAULT = FisherScoreOptions()


class _UnscoredGaussian(GaussianDistribution):
    """Gaussian component with the base-class Fisher methods restored."""

    seq_fisher_score = ProbabilityDistribution.seq_fisher_score
    fisher_dim = ProbabilityDistribution.fisher_dim


def _mixture(second: Optional[Any] = None) -> MixtureDistribution:
    """Return an overlapping two-component Gaussian mixture."""
    second = GaussianDistribution(2.0, 1.0) if second is None else second
    return MixtureDistribution([GaussianDistribution(0.0, 1.0), second], [0.4, 0.6])


def _encode(dist: Any, data: Any) -> Any:
    """Encode ``data`` for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def test_raw_score_matches_formula() -> None:
    dist = _mixture()
    data = [-1.0, 0.5, 2.0, 4.0]
    encoded = _encode(dist, data)
    z = dist.seq_posterior(encoded)
    w = np.asarray([0.4, 0.6])
    expected = [(z - w) / np.sqrt(w)]
    for j, component in enumerate(dist.components):
        scores = component.seq_fisher_score(_encode(component, data))
        expected.append(scores * (z[:, j] / np.sqrt(w[j]))[:, None])
    raw = _raw_mixture_score(
        dist,
        encoded,
        lambda e, _: e.data,
        component_score_dims(dist.components, DEFAULT)[0],
        DEFAULT,
    )
    np.testing.assert_allclose(raw, np.hstack(expected), rtol=1e-12)


def test_raw_score_has_zero_mean_and_covariance_below_identity() -> None:
    # The observed-data score has mean zero; complete-data scaling over-states
    # the information, so its covariance is at most the identity, and strictly
    # below it in directions where the components overlap.
    dist = _mixture()
    data = dist.sampler(seed=11).sample(20000)
    raw = _raw_mixture_score(
        dist,
        _encode(dist, data),
        lambda e, _: e.data,
        component_score_dims(dist.components, DEFAULT)[0],
        DEFAULT,
    )
    np.testing.assert_allclose(raw.mean(axis=0), 0.0, atol=0.03)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(raw.T)))
    assert eigenvalues[-1] < 1.03
    # One null direction (weights sum to one), then clear information loss.
    assert abs(eigenvalues[0]) < 1e-9
    assert eigenvalues[1] < 0.5


def test_whitened_score_depends_only_on_the_model() -> None:
    dist = _mixture()
    data = dist.sampler(seed=5).sample(300)
    full = dist.seq_fisher_score(_encode(dist, data))
    part = dist.seq_fisher_score(_encode(dist, data[:7]))
    # Whitening is fitted on model draws, not on the batch: a subset scores
    # the same up to rounding in the matrix product.
    np.testing.assert_allclose(full[:7], part, rtol=1e-12, atol=1e-14)
    other_seed = dist.seq_fisher_score(
        _encode(dist, data[:7]), FisherScoreOptions(whitening_seed=1)
    )
    assert not np.array_equal(other_seed, part)
    np.testing.assert_allclose(other_seed, part, atol=0.2)


def test_whitened_score_is_a_projector_of_rank_dim_minus_one() -> None:
    dist = _mixture()
    scores = dist.seq_fisher_score(_encode(dist, dist.sampler(seed=3).sample(5000)))
    assert scores.shape == (5000, 2 + 2 * 2)
    check_moments(scores, rank_deficit=1, centered=True)


def test_dim_and_field_slices() -> None:
    dist = _mixture()
    assert dist.fisher_dim() == 6
    assert dist.fisher_field_slices() == {
        "weights": slice(0, 2),
        "component_0": slice(2, 4),
        "component_1": slice(4, 6),
    }


def test_nested_mixture_inside_composite() -> None:
    dist = CompositeDistribution(
        (PoissonDistribution(2.0, name="count"), _mixture()), name="record"
    )
    assert dist.fisher_field_slices() == {
        "count": slice(0, 1),
        "field_1.weights": slice(1, 3),
        "field_1.component_0": slice(3, 5),
        "field_1.component_1": slice(5, 7),
    }
    encoded = _encode(dist, [(1, 0.5), (3, 2.5)])
    scores = dist.seq_fisher_score(encoded)
    np.testing.assert_array_equal(
        scores[:, 1:],
        dist.dists[1].seq_fisher_score(_encode(dist.dists[1], [0.5, 2.5])),
    )


def test_heterogeneous_mixture_with_different_encoders() -> None:
    dist = HeterogeneousMixtureDistribution(
        [GaussianDistribution(5.0, 1.0), GammaDistribution(4.0, 1.5)], [0.5, 0.5]
    )
    data = [3.0, 5.5, 9.0]
    encoded = _encode(dist, data)
    z = dist.seq_posterior(encoded)
    w = np.asarray([0.5, 0.5])
    expected = [(z - w) / np.sqrt(w)]
    for j, component in enumerate(dist.components):
        scores = component.seq_fisher_score(_encode(component, data))
        expected.append(scores * (z[:, j] / np.sqrt(w[j]))[:, None])

    def component_data(enc: Any, j: int) -> Any:
        tag_list, enc_data = enc.data
        return next(enc_data[t] for t, members in enumerate(tag_list) if j in members)

    raw = _raw_mixture_score(
        dist,
        encoded,
        component_data,
        component_score_dims(dist.components, DEFAULT)[0],
        DEFAULT,
    )
    np.testing.assert_allclose(raw, np.hstack(expected), rtol=1e-12)
    scores = dist.seq_fisher_score(encoded)
    assert scores.shape == (3, 2 + 2 + 2)
    assert np.all(np.isfinite(scores))


def test_unsupported_component_raises_by_default() -> None:
    dist = _mixture(_UnscoredGaussian(2.0, 1.0))
    assert not has_fisher_score(dist)
    with pytest.raises(NotImplementedError, match="_UnscoredGaussian"):
        dist.seq_fisher_score(_encode(dist, [1.0]))


def test_posterior_only_skips_unsupported_component_with_one_warning() -> None:
    dist = _mixture(_UnscoredGaussian(2.0, 1.0))
    options = FisherScoreOptions(on_missing="posterior_only")
    assert has_fisher_score(dist, options)
    assert dist.fisher_dim(options) == 2 + 2
    assert dist.fisher_field_slices(options) == {
        "weights": slice(0, 2),
        "component_0": slice(2, 4),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = dist.seq_fisher_score(_encode(dist, [-1.0, 1.0, 3.0]), options)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1 and "_UnscoredGaussian" in messages[0]
    assert scores.shape == (3, 4)
    assert np.all(np.isfinite(scores))


def test_mixture_wrong_encoding_raises() -> None:
    encoded: Any = PoissonDistribution(1.0).dist_to_encoder().seq_encode([1])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        _mixture().seq_fisher_score(encoded)


# --- EmpiricalFisherWhitener -------------------------------------------------


def _correlated(n: int = 5000) -> np.ndarray:
    """Return correlated Gaussian samples with a nonzero mean."""
    rng = np.random.default_rng(0)
    mixing = np.asarray([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, -0.5, 0.3]])
    return np.asarray(rng.standard_normal((n, 3)) @ mixing + [1.0, -2.0, 0.5])


@pytest.mark.parametrize("method", ["full", "diag", "pca"])
def test_whitener_methods(method: Any) -> None:
    x = _correlated()
    whitened = EmpiricalFisherWhitener(method).fit(x).transform(x)
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=1e-12)
    covariance = np.cov(whitened.T, bias=True)
    if method == "diag":
        np.testing.assert_allclose(np.diag(covariance), 1.0, rtol=1e-9)
    else:
        np.testing.assert_allclose(covariance, np.eye(3), atol=1e-9)


def test_whitener_pca_rank_and_null_directions() -> None:
    x = _correlated()
    assert EmpiricalFisherWhitener("pca", rank=2).fit(x).transform(x).shape == (5000, 2)
    u = np.random.default_rng(1).standard_normal(1000)
    duplicated = np.column_stack((u, u)) / np.sqrt(2.0)
    whitened = EmpiricalFisherWhitener("full", center=False).fit(duplicated)
    covariance = np.cov(whitened.transform(duplicated).T, bias=True)
    np.testing.assert_allclose(covariance, [[0.5, 0.5], [0.5, 0.5]], atol=0.05)


def test_whitener_weights_match_repeated_rows() -> None:
    x = _correlated(200)
    weights = np.random.default_rng(2).integers(1, 4, size=200)
    weighted = EmpiricalFisherWhitener().fit(x, weights=weights)
    repeated = EmpiricalFisherWhitener().fit(np.repeat(x, weights, axis=0))
    assert weighted.matrix_ is not None and repeated.matrix_ is not None
    assert weighted.mean_ is not None and repeated.mean_ is not None
    np.testing.assert_allclose(weighted.matrix_, repeated.matrix_, rtol=1e-9)
    np.testing.assert_allclose(weighted.mean_, repeated.mean_, rtol=1e-9)


def test_whitener_round_trips_through_pickle() -> None:
    x = _correlated(500)
    whitener = EmpiricalFisherWhitener("full").fit(x)
    restored = pickle.loads(pickle.dumps(whitener))
    np.testing.assert_array_equal(restored.transform(x), whitener.transform(x))


def test_whitener_errors() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        EmpiricalFisherWhitener().transform(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="method"):
        EmpiricalFisherWhitener("zca").fit(np.zeros((3, 2)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two samples"):
        EmpiricalFisherWhitener().fit(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="weights"):
        EmpiricalFisherWhitener().fit(np.zeros((3, 2)), weights=[1.0, -1.0, 1.0])
    whitener = EmpiricalFisherWhitener().fit(_correlated(50))
    with pytest.raises(ValueError, match="columns"):
        whitener.transform(np.zeros((2, 4)))
