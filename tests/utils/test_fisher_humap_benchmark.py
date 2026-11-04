"""Regression tests for the Fisher humap benchmark tools and its main result.

The full benchmark is ``_benchmarks/fisher_humap/run_benchmark.py``; see
``_benchmarks/fisher-humap.md``.
"""

import numpy as np
import pytest

from dmx.utils.fisher import FisherScoreOptions, fisher_embedding
from dmx.utils.humap import humap
from tests.utils.fisher_benchmark import (
    embedding_report,
    fisher_hellinger_reference,
    generate_ring_clusters,
    knn_graph_agreement,
    oracle_mixture,
    raw_features,
    true_coordinates,
    within_cluster_rank_correlation,
)


def test_generator_is_deterministic_and_keeps_ground_truth() -> None:
    first = generate_ring_clusters(200, seed=3)
    second = generate_ring_clusters(200, seed=3)

    assert first.records == second.records
    assert first.cluster.shape == first.angle.shape == (200,)
    assert np.all((first.angle >= 0.0) & (first.angle < 2.0 * np.pi))
    assert set(first.cluster) == {0, 1, 2}
    assert true_coordinates(first).shape == (200, 5)
    assert raw_features(first.records, first.spec.levels).shape == (200, 9)


def test_reference_matches_feature_map_for_small_scale() -> None:
    data = generate_ring_clusters(120, seed=2)
    model = oracle_mixture(data)
    scale = 1.0e-6
    phi = fisher_embedding(
        model, data.records, scale=scale, top_m=None, dtype=np.float64
    ).matrix.toarray()

    reference = fisher_hellinger_reference(model, data.records, scale=scale)

    same = (data.cluster[:, None] == data.cluster[None, :]) & ~np.eye(120, dtype=bool)
    squared = ((phi[:, None, :] - phi[None, :, :]) ** 2).sum(axis=-1)
    ratio = 2.0 * reference[same] / squared[same]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01)
    assert np.allclose(reference, reference.T)
    assert np.all(np.diag(reference) == 0.0)
    assert knn_graph_agreement(phi, reference) > 0.85


def test_reference_rejects_large_scale() -> None:
    data = generate_ring_clusters(20, seed=0)

    with pytest.raises(ValueError, match="scale"):
        fisher_hellinger_reference(oracle_mixture(data), data.records, scale=0.3)


def test_rank_correlation_scores_true_layout_highly() -> None:
    data = generate_ring_clusters(300, seed=4)
    layout = np.column_stack([np.cos(data.angle), np.sin(data.angle)])
    layout += 10.0 * data.cluster[:, None]

    assert within_cluster_rank_correlation(layout, data) > 0.99
    report = embedding_report(layout, data)
    assert report["knn_recall"] > 0.9
    assert report["trustworthiness"] > 0.99


def test_balanced_fisher_humap_beats_posterior_humap() -> None:
    """Fisher humap recovers within-cluster structure that posteriors lose.

    Averaged over three seeds, ``balance_fields=True`` beats posterior humap
    on within-cluster rank correlation by about 0.2 in the benchmark; the
    test requires half of that.
    """
    balanced = FisherScoreOptions(balance_fields=True)
    gaps = []
    for seed in (1, 2, 3):
        data = generate_ring_clusters(400, seed=seed)
        model = oracle_mixture(data)
        posterior, *_ = humap(
            data.records,
            mix_model=model,
            verbose=False,
            umap_kwargs={"random_state": seed},
        )
        fisher, *_ = humap(
            data.records,
            mix_model=model,
            verbose=False,
            umap_kwargs={"random_state": seed},
            representation="fisher",
            fisher_options=balanced,
        )
        scores = {
            "posterior": within_cluster_rank_correlation(posterior, data),
            "fisher": within_cluster_rank_correlation(fisher, data),
        }
        gaps.append(scores["fisher"] - scores["posterior"])

    assert np.mean(gaps) > 0.1, gaps
