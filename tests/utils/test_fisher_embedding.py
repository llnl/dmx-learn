"""Tests for the mixture-level Fisher embedding."""

import warnings
from typing import Any, List, cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dmx import bstats, stats
from dmx.stats.pdist import ProbabilityDistribution
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherEmbedding,
    FisherScoreOptions,
    WhiteningMethod,
    fisher_embedding,
)


class _UnscoredGaussian(stats.GaussianDistribution):
    """Gaussian with the base-class (unsupported) Fisher methods restored."""

    seq_fisher_score = ProbabilityDistribution.seq_fisher_score
    fisher_dim = ProbabilityDistribution.fisher_dim


def _heterogeneous_mixture() -> stats.MixtureDistribution:
    """Two composite components with a Gaussian and a categorical field."""
    return stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(-2.0, 1.0),
                    stats.CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
                )
            ),
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(2.0, 2.0),
                    stats.CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
                )
            ),
        ],
        [0.4, 0.6],
    )


def _separated_gaussians(count: int = 5) -> stats.MixtureDistribution:
    """Gaussian mixture whose components are far apart."""
    return stats.MixtureDistribution(
        [stats.GaussianDistribution(20.0 * k, 1.0) for k in range(count)],
        np.full(count, 1.0 / count),
    )


def _bstats_mixture() -> bstats.MixtureDistribution:
    """Bayesian Gaussian mixture with three components."""
    return bstats.MixtureDistribution(
        (
            bstats.GaussianDistribution(-3.0, 1.0),
            bstats.GaussianDistribution(0.0, 0.5),
            bstats.GaussianDistribution(3.0, 2.0),
        ),
        [0.3, 0.3, 0.4],
    )


def _sample(model: Any, count: int, seed: int = 1) -> List[Any]:
    """Draw records from a model."""
    return list(model.sampler(seed).sample(count))


def _embed(model: Any, data: List[Any], **kwargs: Any) -> FisherEmbedding:
    """Embed records with ``seq_fisher_embedding`` on one encoded sequence."""
    return cast(
        FisherEmbedding,
        model.seq_fisher_embedding(model.dist_to_encoder().seq_encode(data), **kwargs),
    )


def _posteriors(model: Any, data: List[Any]) -> np.ndarray:
    """Return posteriors of records."""
    return np.asarray(model.seq_posterior(model.dist_to_encoder().seq_encode(data)))


def _scores(model: Any, data: List[Any], index: int) -> np.ndarray:
    """Return dense component scores of records."""
    encoded = model.dist_to_encoder().seq_encode(data).data
    scores = model.components[index].seq_fisher_score(encoded)
    return scores.toarray() if hasattr(scores, "toarray") else np.asarray(scores)


MODELS = [
    pytest.param(_heterogeneous_mixture, id="stats-composite"),
    pytest.param(_bstats_mixture, id="bstats-gaussian"),
]


@pytest.mark.parametrize("make", MODELS)
def test_output_is_csr_with_documented_slices(make: Any) -> None:
    model = make()
    data = _sample(model, 40)

    embedding = _embed(model, data)

    count = len(model.components)
    dims = [component.fisher_dim() for component in model.components]
    assert isinstance(embedding.matrix, csr_matrix)
    assert embedding.matrix.dtype == np.float32
    assert embedding.shape == (40, count + sum(dims))
    assert embedding.posterior_slice == slice(0, count)
    offset = count
    for index, dim in enumerate(dims):
        assert embedding.component_slices[index] == slice(offset, offset + dim)
        fields = model.components[index].fisher_field_slices()
        assert embedding.field_slices[index] == {
            name: slice(offset + part.start, offset + part.stop)
            for name, part in fields.items()
        }
        offset += dim
    assert embedding.top_m == 3
    assert embedding.whiteners is None


@pytest.mark.parametrize("make", MODELS)
def test_zero_scale_gives_posterior_features(make: Any) -> None:
    model = make()
    data = _sample(model, 30)

    embedding = _embed(model, data, scale=0.0)

    dense = embedding.matrix.toarray()
    count = len(model.components)
    expected = np.sqrt(_posteriors(model, data)).astype(np.float32)
    np.testing.assert_array_equal(dense[:, :count], expected)
    assert not np.any(dense[:, count:])
    assert embedding.scale == 0.0


@pytest.mark.parametrize("make", MODELS)
def test_default_scale_is_inverse_effective_dimension(make: Any) -> None:
    model = make()
    data = _sample(model, 30)

    embedding = _embed(model, data)

    dims = np.asarray([component.fisher_dim() for component in model.components])
    effective = np.mean(_posteriors(model, data) @ dims)
    assert embedding.scale == pytest.approx(1.0 / effective, rel=1e-12)


def test_identical_posteriors_give_scaled_score_distance() -> None:
    # The categorical field is the same in both components, so the posterior
    # depends only on the Gaussian field.
    categorical = {"a": 0.5, "b": 0.3, "c": 0.2}
    model = stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(-1.0, 1.0),
                    stats.CategoricalDistribution(categorical),
                )
            ),
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(1.0, 1.5),
                    stats.CategoricalDistribution(categorical),
                )
            ),
        ],
        [0.5, 0.5],
    )
    data = [(0.3, "a"), (0.3, "c")]
    scale = 0.4

    embedding = _embed(model, data, scale=scale, top_m=None, dtype=np.float64)

    posteriors = _posteriors(model, data)
    np.testing.assert_allclose(posteriors[0], posteriors[1], rtol=0, atol=1e-15)
    expected = scale * sum(
        posteriors[0, k]
        * np.sum((_scores(model, data, k)[0] - _scores(model, data, k)[1]) ** 2)
        for k in range(2)
    )
    rows = embedding.matrix.toarray()
    assert np.sum((rows[0] - rows[1]) ** 2) == pytest.approx(expected, rel=1e-12)


def test_single_component_distance_is_scaled_score_distance() -> None:
    model = stats.MixtureDistribution([stats.GaussianDistribution(0.5, 2.0)], [1.0])
    data = [-1.0, 0.25, 3.0]

    embedding = _embed(model, data, dtype=np.float64)

    scores = _scores(model, data, 0)
    rows = embedding.matrix.toarray()
    for i in range(3):
        for j in range(3):
            assert np.sum((rows[i] - rows[j]) ** 2) == pytest.approx(
                embedding.scale * np.sum((scores[i] - scores[j]) ** 2), abs=1e-12
            )


@pytest.mark.parametrize("make", MODELS)
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_chunked_equals_unchunked(make: Any, chunk_size: int) -> None:
    model = make()
    data = _sample(model, 45)

    reference = _embed(model, data)
    chunked = fisher_embedding(model, data, chunk_size=chunk_size)

    assert chunked.scale == reference.scale
    assert chunked.component_slices == reference.component_slices
    assert chunked.field_slices == reference.field_slices
    np.testing.assert_array_equal(chunked.matrix.toarray(), reference.matrix.toarray())
    assert chunked.matrix.dtype == np.float32


def test_top_m_truncation_loses_little_on_separated_mixture() -> None:
    model = _separated_gaussians(5)
    data = _sample(model, 200)

    full = _embed(model, data, top_m=5, dtype=np.float64).matrix.toarray()
    truncated = _embed(model, data, top_m=3, dtype=np.float64).matrix.toarray()

    assert np.max(np.abs(full - truncated)) < 1e-6


def test_top_m_keeps_at_most_m_score_blocks() -> None:
    model = stats.MixtureDistribution(
        [stats.GaussianDistribution(float(k), 1.0) for k in range(4)],
        np.full(4, 0.25),
    )
    data = _sample(model, 50)

    embedding = _embed(model, data, top_m=2)

    dense = embedding.matrix.toarray()
    posteriors = _posteriors(model, data)
    for row, posterior in zip(dense, posteriors):
        kept = [
            index
            for index, columns in embedding.component_slices.items()
            if np.any(row[columns])
        ]
        assert len(kept) <= 2
        assert set(kept) <= set(np.argsort(-posterior, kind="stable")[:2])


def test_zero_weight_components_are_skipped() -> None:
    model = stats.MixtureDistribution(
        [
            stats.GaussianDistribution(-1.0, 1.0),
            stats.GaussianDistribution(0.0, 1.0),
            stats.GaussianDistribution(1.0, 1.0),
        ],
        [0.5, 0.0, 0.5],
    )
    data = [-1.0, 0.0, 2.0]

    embedding = _embed(model, data, top_m=None)

    assert set(embedding.component_slices) == {0, 2}
    assert embedding.shape == (3, 3 + 4)
    assert not np.any(embedding.matrix.toarray()[:, 1])


def test_like_reuses_feature_space() -> None:
    model = _heterogeneous_mixture()
    data = _sample(model, 30)
    reference = fisher_embedding(model, data, top_m=1, whiten="diag")

    transformed = fisher_embedding(model, data[:10], like=reference, chunk_size=4)

    assert transformed.scale == reference.scale
    assert transformed.top_m == 1
    assert transformed.whiteners is reference.whiteners
    np.testing.assert_array_equal(
        transformed.matrix.toarray(), reference.matrix[:10].toarray()
    )


def test_like_rejects_conflicting_arguments() -> None:
    model = _heterogeneous_mixture()
    data = _sample(model, 10)
    reference = fisher_embedding(model, data)

    with pytest.raises(ValueError, match="like"):
        fisher_embedding(model, data, like=reference, scale=1.0)
    with pytest.raises(ValueError, match="layout"):
        fisher_embedding(_separated_gaussians(2), [0.0, 1.0], like=reference)


@pytest.mark.parametrize("method", ["full", "diag"])
def test_data_whiteners_match_weighted_fit(method: WhiteningMethod) -> None:
    model = _heterogeneous_mixture()
    data = _sample(model, 60)

    embedding = fisher_embedding(model, data, whiten=method, chunk_size=13)

    posteriors = _posteriors(model, data)
    assert embedding.whiteners is not None
    for index, whitener in enumerate(embedding.whiteners):
        expected = EmpiricalFisherWhitener(method).fit(
            _scores(model, data, index), posteriors[:, index]
        )
        assert whitener is not None
        np.testing.assert_allclose(
            np.asarray(whitener.mean_), np.asarray(expected.mean_), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(whitener.matrix_), np.asarray(expected.matrix_), atol=1e-8
        )
    unchunked = fisher_embedding(model, data, whiten=method)
    np.testing.assert_allclose(
        embedding.matrix.toarray(), unchunked.matrix.toarray(), rtol=1e-6, atol=1e-6
    )


def test_pca_whitener_changes_block_width_and_drops_field_slices() -> None:
    model = _heterogeneous_mixture()
    data = _sample(model, 60)

    embedding = fisher_embedding(
        model, data, whiten=EmpiricalFisherWhitener("pca", rank=2)
    )

    assert embedding.component_slices == {0: slice(2, 4), 1: slice(4, 6)}
    assert embedding.field_slices == {0: {}, 1: {}}


def test_posterior_only_skips_unscored_components_and_warns_once() -> None:
    model = stats.MixtureDistribution(
        [stats.GaussianDistribution(-1.0, 1.0), _UnscoredGaussian(1.0, 1.0)],
        [0.5, 0.5],
    )
    data = _sample(model, 12)
    options = FisherScoreOptions(on_missing="posterior_only")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedding = fisher_embedding(model, data, options=options, chunk_size=5)

    assert len(caught) == 1
    assert "_UnscoredGaussian" in str(caught[0].message)
    assert set(embedding.component_slices) == {0}
    assert embedding.shape == (12, 2 + 2)
    with pytest.raises(NotImplementedError):
        fisher_embedding(model, data)


def test_stats_and_bstats_embeddings_agree() -> None:
    stats_model = stats.MixtureDistribution(
        [stats.GaussianDistribution(-3.0, 1.0), stats.GaussianDistribution(0.0, 0.5)],
        [0.3, 0.7],
    )
    bstats_model = bstats.MixtureDistribution(
        (bstats.GaussianDistribution(-3.0, 1.0), bstats.GaussianDistribution(0.0, 0.5)),
        [0.3, 0.7],
    )
    data = [-3.5, -1.0, 0.2, 1.4]

    expected = fisher_embedding(stats_model, data, dtype=np.float64).matrix.toarray()
    actual = fisher_embedding(bstats_model, data, dtype=np.float64).matrix.toarray()

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"scale": -1.0}, "scale"),
        ({"scale": float("nan")}, "scale"),
        ({"top_m": 0}, "top_m"),
        ({"whiteners": [None]}, "whiteners"),
    ],
)
def test_invalid_arguments_raise(kwargs: Any, match: str) -> None:
    model = _separated_gaussians(2)

    with pytest.raises(ValueError, match=match):
        _embed(model, [0.0, 1.0], **kwargs)


def test_fisher_embedding_input_checks() -> None:
    model = _separated_gaussians(2)

    with pytest.raises(TypeError, match="seq_fisher_embedding"):
        fisher_embedding(stats.GaussianDistribution(0.0, 1.0), [0.0])
    with pytest.raises(ValueError, match="at least one record"):
        fisher_embedding(model, [])
    with pytest.raises(ValueError, match="chunk_size"):
        fisher_embedding(model, [0.0], chunk_size=0)


def test_stats_seq_fisher_embedding_requires_mixture_encoding() -> None:
    model: Any = _separated_gaussians(2)
    encoded = stats.GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([0.0])

    with pytest.raises(TypeError, match="MixtureEncodedDataSequence"):
        model.seq_fisher_embedding(encoded)


def test_whitener_fit_covariance_matches_fit() -> None:
    rng = np.random.RandomState(3)
    scores = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 3))
    weights = rng.uniform(size=50)
    fitted = EmpiricalFisherWhitener("full").fit(scores, weights)
    normalized = weights / weights.sum()
    mean = normalized @ scores
    covariance = ((scores - mean) * normalized[:, None]).T @ (scores - mean)

    from_moments = EmpiricalFisherWhitener("full").fit_covariance(covariance, mean)

    np.testing.assert_allclose(
        np.asarray(from_moments.matrix_), np.asarray(fitted.matrix_)
    )
    np.testing.assert_allclose(np.asarray(from_moments.mean_), np.asarray(fitted.mean_))
    with pytest.raises(ValueError, match="square"):
        EmpiricalFisherWhitener().fit_covariance(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="mean"):
        EmpiricalFisherWhitener().fit_covariance(np.eye(2), np.zeros(3))


def test_sparse_component_scores() -> None:
    # dense_category_limit=1 makes the categorical scores CSR.
    model = stats.MixtureDistribution(
        [
            stats.CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.2, "c": 0.7}),
        ],
        [0.5, 0.5],
    )
    data = ["a", "b", "c", "c"]
    options = FisherScoreOptions(dense_category_limit=1)
    encoded = model.dist_to_encoder().seq_encode(data)
    assert hasattr(model.components[0].seq_fisher_score(encoded.data, options), "tocsr")

    embedding = _embed(model, data, scale=0.5, options=options, dtype=np.float64)

    posteriors = _posteriors(model, data)
    rows = embedding.matrix.toarray()
    for index, columns in embedding.component_slices.items():
        scores: Any = model.components[index].seq_fisher_score(encoded.data, options)
        expected = scores.toarray() * np.sqrt(0.5 * posteriors[:, index])[:, None]
        np.testing.assert_allclose(rows[:, columns], expected, atol=1e-14)
    whitened = fisher_embedding(model, data, options=options, whiten="diag")
    assert whitened.shape == embedding.shape


def test_component_without_posterior_mass_is_not_whitened() -> None:
    model = stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(0.0, 1.0),
                    stats.CategoricalDistribution({"a": 1.0}),
                )
            ),
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(0.0, 1.0),
                    stats.CategoricalDistribution({"b": 1.0}),
                )
            ),
        ],
        [0.5, 0.5],
    )
    data = [(-1.0, "a"), (0.5, "a"), (2.0, "a")]

    embedding = fisher_embedding(model, data, whiten="full")

    assert embedding.whiteners is not None
    assert embedding.whiteners[0] is not None
    assert embedding.whiteners[1] is None


def test_rows_do_not_depend_on_chunking_with_reused_whiteners() -> None:
    model = _heterogeneous_mixture()
    data = _sample(model, 50)
    reference = fisher_embedding(model, data, whiten="full", dtype=np.float64)

    chunked = fisher_embedding(model, data, like=reference, chunk_size=3)

    np.testing.assert_array_equal(chunked.matrix.toarray(), reference.matrix.toarray())


def test_high_dimensional_gaussian_scores_do_not_depend_on_chunking() -> None:
    # A multi-right-hand-side triangular solve rounds differently by batch
    # size at this dimension; the scores must not.
    rng = np.random.RandomState(4)
    dim = 64
    factor = rng.normal(size=(dim, dim))
    dist = stats.MultivariateGaussianDistribution(
        rng.normal(size=dim), factor @ factor.T + dim * np.eye(dim)
    )
    data = list(dist.sampler(1).sample(40))
    encoder = dist.dist_to_encoder()

    reference = dist.seq_fisher_score(encoder.seq_encode(data))
    chunked = np.vstack(
        [
            dist.seq_fisher_score(encoder.seq_encode(data[start : start + 7]))
            for start in range(0, len(data), 7)
        ]
    )

    np.testing.assert_array_equal(chunked, reference)
