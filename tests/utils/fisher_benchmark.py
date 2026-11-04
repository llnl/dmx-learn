"""Synthetic data, exact reference metric, and quality metrics for Fisher humap.

Shared by ``tests/utils/test_fisher_humap_benchmark.py`` and
``_benchmarks/fisher_humap/run_benchmark.py``; not part of the package.

- :func:`generate_ring_clusters` draws heterogeneous records
  ``(x, y, level, count)`` from clusters with known internal structure: each
  cluster is a ring in the two Gaussian fields, the categorical level follows
  the position on the ring, and the Poisson count rises and falls with it.
  The cluster and the angle on the ring are returned as ground truth.
- :func:`oracle_mixture` builds the ``dmx.stats`` mixture a perfect fit
  would give (one composite component per cluster, moment matched), so tests
  do not depend on a fitting run.
- :func:`fisher_hellinger_reference` is the exact squared Hellinger distance
  between the records' locally tilted models ``Q_i`` (roadmap section 2.1),
  with closed-form Bhattacharyya coefficients for Gaussian, Poisson and
  categorical leaves. It costs ``O(N^2)`` and is only for small ``N``.
- The metric functions score an embedding against the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from dmx import stats
from dmx.utils._fisher_kernels import categorical_probabilities
from dmx.utils.fisher import FisherScoreOptions, fisher_embedding

Record = Tuple[float, float, str, int]


@dataclass(frozen=True)
class RingClusterSpec:
    """Parameters of the ring-cluster generator.

    Attributes:
        centers: Ring center ``(x, y)`` of each cluster.
        radius: Ring radius.
        noise: Standard deviation of Gaussian noise added to ``x`` and ``y``.
        levels: Categorical levels; level ``l`` covers the ``l``-th equal arc
            of the ring.
        level_purity: Probability that a record's level is its arc's level;
            otherwise the level is uniform.
        rates: Mean Poisson count of each cluster.
        rate_swing: Relative swing of the count rate around the ring:
            ``rate * (1 + rate_swing * sin(angle))``.
        weights: Cluster probabilities.
    """

    centers: Tuple[Tuple[float, float], ...] = ((-12.0, 0.0), (0.0, 12.0), (12.0, 0.0))
    radius: float = 3.0
    noise: float = 0.3
    levels: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
    level_purity: float = 0.85
    rates: Tuple[float, ...] = (3.0, 8.0, 15.0)
    rate_swing: float = 0.5
    weights: Tuple[float, ...] = (0.3, 0.3, 0.4)


@dataclass
class RingClusterData:
    """Records drawn by :func:`generate_ring_clusters`, with ground truth.

    Attributes:
        records: ``(x, y, level, count)`` tuples.
        cluster: True cluster of each record.
        angle: True angle of each record on its ring, in ``[0, 2 pi)``.
        spec: Generator parameters.
    """

    records: List[Record]
    cluster: np.ndarray
    angle: np.ndarray
    spec: RingClusterSpec = field(default_factory=RingClusterSpec)


def generate_ring_clusters(
    count: int, spec: Optional[RingClusterSpec] = None, seed: int = 0
) -> RingClusterData:
    """Draw ``count`` records from ring-shaped clusters.

    Args:
        count: Number of records.
        spec: Generator parameters; None uses :class:`RingClusterSpec`.
        seed: Seed of the random state.

    Returns:
        The records and their true clusters and angles.
    """
    spec = RingClusterSpec() if spec is None else spec
    rng = np.random.RandomState(seed)
    weights = np.asarray(spec.weights, dtype=float)
    cluster = rng.choice(len(spec.centers), size=count, p=weights / weights.sum())
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    centers = np.asarray(spec.centers, dtype=float)[cluster]
    x = centers[:, 0] + spec.radius * np.cos(angle) + rng.normal(0, spec.noise, count)
    y = centers[:, 1] + spec.radius * np.sin(angle) + rng.normal(0, spec.noise, count)
    num_levels = len(spec.levels)
    arc = np.minimum((angle / (2.0 * np.pi) * num_levels).astype(int), num_levels - 1)
    level = np.where(
        rng.uniform(size=count) < spec.level_purity,
        arc,
        rng.randint(num_levels, size=count),
    )
    rates = np.asarray(spec.rates, dtype=float)[cluster]
    counts = rng.poisson(rates * (1.0 + spec.rate_swing * np.sin(angle)))
    records = [
        (float(a), float(b), spec.levels[int(c)], int(d))
        for a, b, c, d in zip(x, y, level, counts)
    ]
    return RingClusterData(records, cluster, angle, spec)


def true_coordinates(data: RingClusterData, separation: float = 10.0) -> np.ndarray:
    """Return Euclidean coordinates of the ground truth.

    Each record is ``[separation * onehot(cluster), cos(angle), sin(angle)]``:
    records of one cluster lie on a unit circle by their true angle, and
    clusters are ``separation * sqrt(2)`` apart.
    """
    num_clusters = len(data.spec.centers)
    onehot = np.eye(num_clusters)[data.cluster] * separation
    return np.hstack([onehot, np.cos(data.angle)[:, None], np.sin(data.angle)[:, None]])


def oracle_mixture(data: RingClusterData) -> stats.MixtureDistribution:
    """Return the mixture a perfect fit to ``data`` would give.

    One composite component per true cluster with a Gaussian for each of
    ``x`` and ``y``, a categorical with add-one smoothed level frequencies,
    and a Poisson, all moment matched on the cluster's records.
    """
    components = []
    weights = []
    for k in range(len(data.spec.centers)):
        rows = [r for r, c in zip(data.records, data.cluster) if c == k]
        values = np.asarray([[r[0], r[1], r[3]] for r in rows], dtype=float)
        counts = {level: 1.0 for level in data.spec.levels}
        for row in rows:
            counts[row[2]] += 1.0
        total = sum(counts.values())
        components.append(
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(
                        float(values[:, 0].mean()), float(values[:, 0].var())
                    ),
                    stats.GaussianDistribution(
                        float(values[:, 1].mean()), float(values[:, 1].var())
                    ),
                    stats.CategoricalDistribution(
                        {level: c / total for level, c in counts.items()}
                    ),
                    stats.PoissonDistribution(float(values[:, 2].mean())),
                )
            )
        )
        weights.append(len(rows) / len(data.records))
    return stats.MixtureDistribution(components, weights)


def raw_features(records: Sequence[Record], levels: Sequence[str]) -> np.ndarray:
    """Return standardized numeric fields with one-hot levels, for a baseline."""
    numeric = np.asarray([[r[0], r[1], r[3]] for r in records], dtype=float)
    numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)
    position = {level: i for i, level in enumerate(levels)}
    onehot = np.zeros((len(records), len(levels)))
    onehot[np.arange(len(records)), [position[r[2]] for r in records]] = 1.0
    return np.hstack([numeric, onehot])


# --- Exact reference metric ---------------------------------------------------


def _gaussian_bc(x: np.ndarray, dist: Any, eps: float) -> np.ndarray:
    """Return BC between the tilted Gaussians of every pair of records."""
    mean = (1.0 - eps) * dist.mu + eps * x
    var = (1.0 - eps) * dist.sigma2 + eps * (1.0 - eps) * (x - dist.mu) ** 2
    v_sum = var[:, None] + var[None, :]
    return np.asarray(
        np.sqrt(2.0 * np.sqrt(var[:, None] * var[None, :]) / v_sum)
        * np.exp(-((mean[:, None] - mean[None, :]) ** 2) / (4.0 * v_sum))
    )


def _poisson_bc(x: np.ndarray, dist: Any, eps: float) -> np.ndarray:
    """Return BC between the tilted Poissons of every pair of records."""
    root = np.sqrt((1.0 - eps) * dist.lam + eps * x)
    return np.asarray(np.exp(-0.5 * (root[:, None] - root[None, :]) ** 2))


def _categorical_bc(
    values: Sequence[Any], dist: Any, eps: float, floor: float
) -> np.ndarray:
    """Return BC between the tilted categoricals of every pair of records.

    Level probabilities are floored as in the categorical Fisher score; a
    value outside ``pmap`` tilts toward an extra "other" level.
    """
    weights = list(dist.pmap.values())
    if dist.default_value > 0.0:
        weights.append(dist.default_value)
    probs = np.append(categorical_probabilities(weights, floor), 0.0)
    position = {level: i for i, level in enumerate(dist.pmap)}
    other = len(probs) - 2 if dist.default_value > 0.0 else len(probs) - 1
    index = np.asarray([position.get(v, other) for v in values])
    tilted = np.tile((1.0 - eps) * probs, (len(index), 1))
    tilted[np.arange(len(index)), index] += eps
    root = np.sqrt(tilted)
    return np.asarray(root @ root.T)


def _component_bc(
    component: Any, columns: Sequence[np.ndarray], eps: float, floor: float
) -> np.ndarray:
    """Return the BC matrix of one component, a leaf or composite of leaves."""
    if isinstance(component, stats.CompositeDistribution):
        leaves = component.dists
    else:
        leaves = (component,)
        columns = [columns[0]] if len(columns) == 1 else [np.asarray(columns)]
    result = np.ones((len(columns[0]), len(columns[0])))
    for leaf, values in zip(leaves, columns):
        if isinstance(leaf, stats.GaussianDistribution):
            result *= _gaussian_bc(np.asarray(values, dtype=float), leaf, eps)
        elif isinstance(leaf, stats.PoissonDistribution):
            result *= _poisson_bc(np.asarray(values, dtype=float), leaf, eps)
        elif isinstance(leaf, stats.CategoricalDistribution):
            result *= _categorical_bc(list(values), leaf, eps, floor)
        else:
            raise NotImplementedError(
                f"No closed-form reference for {type(leaf).__name__}."
            )
    return result


def fisher_hellinger_reference(
    model: stats.MixtureDistribution,
    data: Sequence[Any],
    *,
    scale: Optional[float] = None,
    options: Optional[FisherScoreOptions] = None,
) -> np.ndarray:
    """Return exact squared Hellinger distances between local models ``Q_i``.

    ``Q_i(k, x) = z_ik f_k(x | eta_ik)``, where component ``k`` is tilted
    toward ``x_i`` in mean parameters, ``mu = (1 - eps) mu_k + eps T(x_i)``,
    and ``eps = 2 sqrt(c)`` matches the Fisher embedding with scale ``c``:
    for small ``eps``, ``2 H^2(Q_i, Q_j) ~ ||phi_i - phi_j||^2`` with
    ``top_m=None``. Components must be Gaussian, Poisson or categorical
    leaves, or composites of them.

    Args:
        model: Mixture with supported components.
        data: Records, as accepted by ``model``.
        scale: Embedding scale ``c``; None uses the embedding's default
            ``1 / p_eff`` on ``data``. Must be below ``1/4`` so the tilt
            weights stay in ``[0, 1)``.
        options: Score options; only ``prob_floor`` is used.

    Returns:
        Symmetric ``(N, N)`` matrix of ``H^2`` with zero diagonal.

    Raises:
        ValueError: If ``scale >= 1/4``.
        NotImplementedError: If a component has an unsupported leaf.
    """
    options = FisherScoreOptions() if options is None else options
    if scale is None:
        scale = fisher_embedding(model, data, options=options, top_m=1).scale
    if not 0.0 <= scale < 0.25:
        raise ValueError(f"The reference needs 0 <= scale < 1/4, got {scale}.")
    eps = 2.0 * np.sqrt(scale)
    posteriors = model.seq_posterior(model.dist_to_encoder().seq_encode(data))
    root_z = np.sqrt(posteriors)
    columns = (
        [np.asarray(column, dtype=object) for column in zip(*data)]
        if (isinstance(data[0], tuple))
        else [np.asarray(data)]
    )
    overlap = np.zeros((len(data), len(data)))
    for k, component in enumerate(model.components):
        if model.w[k] == 0.0:
            continue
        overlap += np.outer(root_z[:, k], root_z[:, k]) * _component_bc(
            component, columns, eps, options.prob_floor
        )
    distances = np.clip(1.0 - overlap, 0.0, None)
    np.fill_diagonal(distances, 0.0)
    return np.asarray(0.5 * (distances + distances.T))


# --- Embedding quality metrics ------------------------------------------------


def trustworthiness_continuity(
    true_coords: np.ndarray, embedding: np.ndarray, k: int = 15
) -> Tuple[float, float]:
    """Return trustworthiness and continuity of an embedding.

    Trustworthiness penalizes embedding neighbours that are far in the ground
    truth; continuity penalizes true neighbours that the embedding separates.
    """
    trust = float(trustworthiness(true_coords, embedding, n_neighbors=k))
    cont = float(trustworthiness(embedding, true_coords, n_neighbors=k))
    return trust, cont


def _neighbours(points: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of each point's ``k`` nearest other points."""
    search = NearestNeighbors(n_neighbors=k + 1).fit(points)
    return np.asarray(search.kneighbors(return_distance=False)[:, 1:])


def _distance_neighbours(distances: np.ndarray, k: int) -> np.ndarray:
    """Return each row's ``k`` nearest other indices from a distance matrix."""
    masked = distances + np.diag(np.full(len(distances), np.inf))
    return np.argsort(masked, axis=1, kind="stable")[:, :k]


def neighbour_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Return the mean fraction of shared neighbours between two index sets."""
    k = first.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(first, second)]))


def knn_recall(true_coords: np.ndarray, embedding: np.ndarray, k: int = 15) -> float:
    """Return the fraction of each record's true ``k`` neighbours kept."""
    return neighbour_overlap(_neighbours(true_coords, k), _neighbours(embedding, k))


def knn_graph_agreement(
    features: np.ndarray, reference_distances: np.ndarray, k: int = 15
) -> float:
    """Return kNN overlap between Euclidean features and a distance matrix."""
    return neighbour_overlap(
        _neighbours(features, k), _distance_neighbours(reference_distances, k)
    )


def within_cluster_rank_correlation(
    embedding: np.ndarray,
    data: RingClusterData,
    pairs: int = 4000,
    seed: int = 0,
) -> float:
    """Return the Spearman correlation of distances within true clusters.

    For random pairs of records in the same true cluster, compares their
    embedding distance with their true chord distance ``2 sin(dtheta / 2)``
    on the ring. Clusters are weighted by size. A representation that
    collapses each cluster scores near 0; one that keeps the ring scores
    near 1.
    """
    rng = np.random.RandomState(seed)
    scores: List[float] = []
    sizes: List[int] = []
    for k in np.unique(data.cluster):
        rows = np.flatnonzero(data.cluster == k)
        if len(rows) < 3:
            continue
        first = rng.choice(rows, pairs)
        second = rng.choice(rows, pairs)
        keep = first != second
        first, second = first[keep], second[keep]
        chord = 2.0 * np.abs(np.sin((data.angle[first] - data.angle[second]) / 2.0))
        spread = np.linalg.norm(embedding[first] - embedding[second], axis=1)
        scores.append(float(spearmanr(spread, chord).statistic))
        sizes.append(len(rows))
    return float(np.average(scores, weights=sizes))


def embedding_report(
    embedding: np.ndarray, data: RingClusterData, k: int = 15
) -> Dict[str, float]:
    """Return every ground-truth metric of a 2-D embedding."""
    coords = true_coordinates(data)
    trust, cont = trustworthiness_continuity(coords, embedding, k)
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "knn_recall": knn_recall(coords, embedding, k),
        "within_cluster_rank_corr": within_cluster_rank_correlation(embedding, data),
    }
