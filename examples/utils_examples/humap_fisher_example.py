"""Compare posterior and Fisher humap embeddings side by side.

``humap`` embeds records by their mixture posteriors by default. With
``representation="fisher"`` it also uses each record's whitened Fisher score
under its most probable components, so records are spread out by where they
sit inside their cluster, not only by which cluster they belong to.

This example fits one mixture per dataset with ``humap``'s default
Dirichlet-process path, then embeds the same records both ways:

- synthetic records ``(x, y, level, count)`` from three ring-shaped clusters
  (the benchmark family in ``_benchmarks/fisher-humap.md``), colored by the
  true angle on the ring;
- the repository's session data in ``data/embedding_data``, colored by the
  user who produced each session.

It writes ``humap_fisher_example.png`` (needs the ``viz`` extra for
matplotlib). Run it from the repository root::

    python examples/utils_examples/humap_fisher_example.py

It takes well under a minute on a laptop.
"""

import ast
import csv
import os
from typing import Any, List, Sequence, Tuple

import numpy as np

from dmx.utils.humap import humap, humap_transform
from dmx.utils.optional import require_optional

DATA_PATH = os.path.join("data", "embedding_data", "embedding_train_data.csv")
OUTPUT = "humap_fisher_example.png"
UMAP_KWARGS = {"n_neighbors": 15, "min_dist": 0.1}
LEVELS = ("a", "b", "c", "d", "e", "f")


def ring_clusters(
    count: int, rng: np.random.RandomState
) -> Tuple[List[Tuple[float, float, str, int]], np.ndarray]:
    """Draw records from three ring-shaped clusters.

    Each cluster is a ring in ``(x, y)``; the categorical level names the arc
    of the ring the record is on (with some noise), and the Poisson count
    rises and falls around the ring.

    Args:
        count: Number of records.
        rng: Random state.

    Returns:
        The records and each record's true angle on its ring.
    """
    centers = np.array([[-12.0, 0.0], [0.0, 12.0], [12.0, 0.0]])
    rates = np.array([3.0, 8.0, 15.0])
    cluster = rng.choice(3, size=count, p=[0.3, 0.3, 0.4])
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    x = centers[cluster, 0] + 3.0 * np.cos(angle) + rng.normal(0.0, 0.3, count)
    y = centers[cluster, 1] + 3.0 * np.sin(angle) + rng.normal(0.0, 0.3, count)
    arc = np.minimum((angle / (2.0 * np.pi) * len(LEVELS)).astype(int), 5)
    level = np.where(rng.uniform(size=count) < 0.85, arc, rng.randint(6, size=count))
    counts = rng.poisson(rates[cluster] * (1.0 + 0.5 * np.sin(angle)))
    records = [
        (float(a), float(b), LEVELS[int(c)], int(d))
        for a, b, c, d in zip(x, y, level, counts)
    ]
    return records, angle


def session_records() -> Tuple[List[Tuple[Any, ...]], List[str]]:
    """Load the repository's session data and the user of each session.

    Returns:
        Records ``(session_time, key, failed_calls, timer, processes)`` and
        the user labels, which are kept out of the records.
    """
    with open(DATA_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        (
            float(row["session_time"]),
            row["key"],
            int(row["failed_calls"]),
            float(row["timer"]),
            ast.literal_eval(row["processes"]),
        )
        for row in rows
    ]
    return records, [row["user"] for row in rows]


def embed_both_ways(
    records: Sequence[Any], seed: int
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Fit a mixture once and embed the records with both representations.

    Args:
        records: Heterogeneous records.
        seed: Seed for mixture fitting and UMAP.

    Returns:
        The posterior embedding, the Fisher embedding, and the outputs needed
        to place new records: ``(fit, model, fisher_embedding)``.
    """
    posterior_coords, model, _, _ = humap(
        records,
        seed=seed,
        max_components=20,
        max_its=200,
        verbose=False,
        umap_kwargs={**UMAP_KWARGS, "random_state": seed},
    )
    fisher_coords, _, fit, features, embedding = humap(
        records,
        mix_model=model,
        verbose=False,
        umap_kwargs={**UMAP_KWARGS, "random_state": seed},
        representation="fisher",
        return_embedding=True,
    )
    assert embedding is not None
    print(
        f"  {model.num_components} components; Fisher features "
        f"{features.shape[1]} columns, "
        f"{embedding.matrix.nnz / features.shape[0]:.0f} "
        f"nonzeros per row, scale c = {embedding.scale:.4f}"
    )
    return posterior_coords, fisher_coords, (fit, model, embedding)


def plot(
    synthetic: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sessions: Tuple[np.ndarray, np.ndarray, List[str]],
) -> None:
    """Draw posterior and Fisher embeddings of both datasets side by side.

    Args:
        synthetic: Posterior and Fisher coordinates, and the true angles.
        sessions: Posterior and Fisher coordinates, and the user labels.
    """
    plt = require_optional("matplotlib.pyplot", "viz", "humap_fisher_example")
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for column, title in enumerate(("Posterior humap", "Fisher humap")):
        points = synthetic[column]
        shown = axes[0, column].scatter(
            points[:, 0],
            points[:, 1],
            c=synthetic[2],
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * np.pi,
            s=8,
            linewidths=0,
        )
        axes[0, column].set_title(f"{title}: ring clusters", loc="left")
    colorbar = figure.colorbar(shown, ax=axes[0, :], fraction=0.03)
    colorbar.set_label("true angle on the ring")

    # Four users: color plus marker shape, so identity is not color alone.
    users = sorted(set(sessions[2]))
    styles = zip(users, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"], "osD^")
    labels = np.asarray(sessions[2])
    for user, color, marker in styles:
        rows = labels == user
        for column in range(2):
            points = sessions[column][rows]
            axes[1, column].scatter(
                points[:, 0],
                points[:, 1],
                color=color,
                marker=marker,
                s=14,
                linewidths=0,
                label=user,
            )
    for column, title in enumerate(("Posterior humap", "Fisher humap")):
        axes[1, column].set_title(f"{title}: session data", loc="left")
    axes[1, 1].legend(frameon=False, title="user", loc="best")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


def main() -> None:
    """Embed both datasets, place new records, and plot the comparison."""
    rng = np.random.RandomState(7)

    print("Ring clusters:")
    records, angle = ring_clusters(900, rng)
    synthetic = embed_both_ways(records, seed=1)

    # Place new records in the fitted Fisher embedding.
    new_records, _ = ring_clusters(100, rng)
    fit, model, embedding = synthetic[2]
    placed = humap_transform(fit, model, new_records, embedding)
    print(f"  placed {placed.shape[0]} new records with humap_transform")

    print("Session data:")
    sessions, users = session_records()
    session_coords = embed_both_ways(sessions, seed=2)

    plot(
        (synthetic[0], synthetic[1], angle),
        (session_coords[0], session_coords[1], users),
    )


if __name__ == "__main__":
    main()
