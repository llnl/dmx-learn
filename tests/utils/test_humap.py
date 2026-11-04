"""Test integration with UMAP."""

# humap's overloads return four values unless return_embedding=True; pylint
# only sees the five-value branch of the implementation.
# pylint: disable=unbalanced-tuple-unpacking

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import umap  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix
from umap.distances import hellinger  # type: ignore[import-untyped]

from dmx import stats
from dmx.utils import humap as humap_module
from dmx.utils.fisher import FisherEmbedding, FisherScoreOptions, fisher_embedding
from dmx.utils.humap import (
    DEFAULT_UMAP,
    HUMAP_FISHER_OPTIONS,
    Representation,
    humap,
    humap_transform,
)
from dmx.utils.progress import ProgressEvent

DATA_DIR = "tests/data"


def _data() -> List[Tuple[str, float]]:
    """Load the heterogeneous example records."""
    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
        return list(pickle.load(f))


def _umap_kwargs() -> Dict[str, Any]:
    """Return fresh UMAP options with a fixed seed."""
    return {"n_neighbors": 15, "min_dist": 0.2, "random_state": 42}


@pytest.fixture(name="fitted", scope="module")
def fixture_fitted() -> Tuple[List[Tuple[str, float]], Any]:
    """Fit the default bstats DPM once for the tests that reuse it."""
    data = _data()
    _, model, _, _ = humap(
        data, seed=10, max_its=50, verbose=False, umap_kwargs=_umap_kwargs()
    )
    return data, model


def test_humap() -> None:
    """Test for humap using automatic fitting."""
    data = _data()

    humap(data, seed=10, umap_kwargs=_umap_kwargs())


def test_default_call_matches_umap_on_posteriors(fitted: Any) -> None:
    data, model = fitted

    result = humap(data, mix_model=model, verbose=False, umap_kwargs=_umap_kwargs())

    assert len(result) == 4
    embeddings, returned_model, fit, features = result
    posteriors = model.seq_posterior(model.seq_encode(data))
    assert returned_model is model
    assert isinstance(features, np.ndarray)
    np.testing.assert_array_equal(features, posteriors)
    assert fit.metric == "hellinger"
    expected = umap.UMAP(**{**DEFAULT_UMAP, **_umap_kwargs()}).fit_transform(posteriors)
    np.testing.assert_array_equal(embeddings, expected)


def test_fisher_mode_does_not_change_later_default_calls(fitted: Any) -> None:
    data, model = fitted
    defaults = dict(DEFAULT_UMAP)

    _, _, fisher_fit, _ = humap(
        data, mix_model=model, verbose=False, representation="fisher"
    )
    _, _, posterior_fit, _ = humap(data, mix_model=model, verbose=False)

    assert fisher_fit.metric == "euclidean"
    assert posterior_fit.metric == "hellinger"
    assert DEFAULT_UMAP == defaults
    assert not hasattr(humap_module, "umap_model")


def test_fisher_mode_on_default_bstats_dpm_path() -> None:
    data = _data()
    events: List[ProgressEvent] = []

    embeddings, model, fit, features, embedding = humap(
        data,
        seed=10,
        max_its=50,
        verbose=False,
        umap_kwargs=_umap_kwargs(),
        progress_callback=events.append,
        representation="fisher",
        return_embedding=True,
    )

    assert embeddings.shape == (len(data), 2)
    assert type(model).__module__ == "dmx.bstats.mixture"
    assert fit.metric == "euclidean"
    assert isinstance(embedding, FisherEmbedding)
    assert isinstance(features, csr_matrix) and features.dtype == np.float32
    assert features is embedding.matrix
    assert features.shape[0] == len(data)
    assert features.shape[1] > model.num_components
    [event] = [event for event in events if event.event == "embedding"]
    assert event.source == "humap"
    assert event.metrics["scale"] == embedding.scale
    assert event.metrics["p_eff"] == pytest.approx(1.0 / embedding.scale, rel=1e-5)
    assert event.metrics["top_m"] == 3
    assert event.metrics["balance_fields"] is True
    assert event.metrics["feature_dim"] == features.shape[1]


def test_fisher_mode_with_supplied_stats_mixture() -> None:
    data = _data()
    levels = sorted({label for label, _ in data})
    model = stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.CategoricalDistribution(
                        {level: 1.0 / len(levels) for level in levels}
                    ),
                    stats.GaussianDistribution(mean, 1.0),
                )
            )
            for mean in (-1.0, 0.0, 1.0)
        ],
        [0.3, 0.4, 0.3],
    )

    embeddings, returned_model, fit, features = humap(
        data,
        mix_model=model,
        verbose=False,
        umap_kwargs=_umap_kwargs(),
        representation="fisher",
        fisher_scale=0.5,
        top_m=None,
    )

    assert returned_model is model
    assert embeddings.shape == (len(data), 2)
    assert fit.metric == "euclidean"
    dims = sum(component.fisher_dim() for component in model.components)
    assert features.shape == (len(data), 3 + dims)


def test_zero_fisher_scale_reproduces_posterior_geometry(fitted: Any) -> None:
    data, model = fitted

    _, _, posterior_fit, posteriors = humap(
        data, mix_model=model, verbose=False, umap_kwargs=_umap_kwargs()
    )
    _, _, fisher_fit, features = humap(
        data,
        mix_model=model,
        verbose=False,
        umap_kwargs=_umap_kwargs(),
        representation="fisher",
        fisher_scale=0.0,
    )

    # Euclidean distance between sqrt-posteriors is sqrt(2) times UMAP's
    # Hellinger distance between posteriors, so the kNN graphs agree up to
    # ties between near-duplicate posteriors.
    dense = csr_matrix(features).toarray().astype(np.float64)
    rng = np.random.RandomState(0)
    for i, j in rng.randint(len(data), size=(200, 2)):
        assert np.linalg.norm(dense[i] - dense[j]) == pytest.approx(
            np.sqrt(2.0) * hellinger(posteriors[i], posteriors[j]), abs=1e-6
        )
    posterior_edges = posterior_fit.graph_.toarray() != 0
    fisher_edges = fisher_fit.graph_.toarray() != 0
    assert np.mean(posterior_edges == fisher_edges) > 0.999


@pytest.mark.parametrize("representation", ["posterior", "fisher"])
def test_humap_transform_places_training_records_near_fit(
    fitted: Any, representation: Representation
) -> None:
    data, model = fitted

    embeddings, _, fit, _, embedding = humap(
        data,
        mix_model=model,
        verbose=False,
        umap_kwargs=_umap_kwargs(),
        representation=representation,
        return_embedding=True,
    )
    placed = humap_transform(fit, model, data, embedding)

    assert (embedding is None) == (representation == "posterior")
    distances = np.linalg.norm(placed - embeddings, axis=1)
    assert np.median(distances) < 0.05 * float(np.mean(embeddings.std(axis=0)))


def test_caller_metric_is_kept_and_defaults_filled(fitted: Any) -> None:
    data, model = fitted
    options = {"metric": "cosine", "random_state": 1}

    _, _, fit, _ = humap(
        data,
        mix_model=model,
        verbose=False,
        umap_kwargs=options,
        representation="fisher",
    )

    assert fit.metric == "cosine"
    assert options["n_neighbors"] == DEFAULT_UMAP["n_neighbors"]


def test_unknown_representation_raises() -> None:
    with pytest.raises(ValueError, match="representation"):
        humap(_data(), representation="raw")  # type: ignore[call-overload]


def test_fisher_mode_balances_fields_by_default(fitted: Any) -> None:
    data, model = fitted

    _, _, _, features, embedding = humap(
        data,
        mix_model=model,
        verbose=False,
        representation="fisher",
        return_embedding=True,
    )

    assert embedding is not None
    assert embedding.options == HUMAP_FISHER_OPTIONS
    assert embedding.options.balance_fields
    expected = fisher_embedding(model, data, options=HUMAP_FISHER_OPTIONS)
    np.testing.assert_array_equal(
        csr_matrix(features).toarray(), expected.matrix.toarray()
    )


def test_explicit_options_keep_pure_fisher_weighting(fitted: Any) -> None:
    data, model = fitted
    pure = FisherScoreOptions()

    _, _, _, features, embedding = humap(
        data,
        mix_model=model,
        verbose=False,
        representation="fisher",
        fisher_options=pure,
        return_embedding=True,
    )

    assert embedding is not None and not embedding.options.balance_fields
    expected = fisher_embedding(model, data, options=pure)
    np.testing.assert_array_equal(
        csr_matrix(features).toarray(), expected.matrix.toarray()
    )
