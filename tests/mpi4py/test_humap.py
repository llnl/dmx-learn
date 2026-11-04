"""Tests for mpi4py automatic model fitting and humap embeddings.

Run with mpiexec -n 4 pytest test_humap.py

"""

import os
import pickle

import numpy as np
from mpi4py import MPI  # pylint: disable=no-name-in-module
from umap import UMAP  # type: ignore[import-untyped]

from dmx.bstats import MixtureDistribution as BMix
from dmx.mpi4py.utils.humap import humap_mpi
from dmx.stats import CategoricalDistribution as Categorical
from dmx.stats import CompositeDistribution as Composite
from dmx.stats import GaussianDistribution as Gaussian
from dmx.stats import MixtureDistribution as Mix
from dmx.utils.fisher import fisher_embedding
from dmx.utils.humap import HUMAP_FISHER_OPTIONS

DATA_DIR = "tests/data"


def test_humap_mpi() -> None:
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:

        with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
            data = pickle.load(f)

    else:
        data = None

    umap_kwargs = {
        "n_neighbors": 15,
        "min_dist": 0.2,
        "random_state": 42,  # Set your desired seed here
    }

    results = humap_mpi(data=data, seed=10, umap_kwargs=umap_kwargs)

    if world_rank == 0:
        sz = len(data)
        if results is None:
            raise RuntimeError("humap_mpi did not return results on rank 0.")
        embs, mix_model, umap_fit, posteriors = results

        assert isinstance(embs, np.ndarray) and embs.shape == (
            sz,
            2,
        ), "Embeddings dims mismatch"
        assert isinstance(umap_fit, UMAP), "UMAP fit not returned."
        assert isinstance(
            mix_model, BMix
        ), "humap should return a bstats.MixtureDistribution object."
        assert (
            isinstance(posteriors, np.ndarray) and posteriors.shape[0] == sz
        ), "Posterior dimension mismatch."
    else:
        assert results is None, f"Did not return None on worker {world_rank}."


def test_humap_mpi_fisher_matches_local_features() -> None:
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    data = None
    if world_rank == 0:
        with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
            data = pickle.load(f)
    levels = comm.bcast(
        sorted({label for label, _ in data}) if data is not None else None, root=0
    )
    model = Mix(
        [
            Composite(
                (
                    Categorical({level: 1.0 / len(levels) for level in levels}),
                    Gaussian(mean, 1.0),
                )
            )
            for mean in (-1.0, 0.0, 1.0)
        ],
        [0.3, 0.4, 0.3],
    )

    results = humap_mpi(
        data=data,
        mix_model=model,
        umap_kwargs={"random_state": 42},
        verbose=False,
        representation="fisher",
        return_embedding=True,
    )

    ok = None
    if world_rank == 0:
        assert results is not None and data is not None
        embs, _, umap_fit, features, embedding = results
        expected = fisher_embedding(model, data, options=HUMAP_FISHER_OPTIONS)
        ok = (
            embs.shape == (len(data), 2)
            and umap_fit.metric == "euclidean"
            and features is embedding.matrix
            and embedding.scale == expected.scale
            and np.array_equal(features.toarray(), expected.matrix.toarray())
        )
    else:
        ok = results is None
    assert all(comm.allgather(ok)), "humap_mpi Fisher features differ from local."
