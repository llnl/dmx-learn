"""Tests for mpi4py automatic model fitting and humap embeddings.

Run with mpiexec -n 4 pytest test_humap.py

"""

import os
import pickle

import numpy as np
from mpi4py import MPI  # pylint: disable=no-name-in-module
from umap import UMAP

from dmx.bstats import MixtureDistribution as BMix
from dmx.mpi4py.utils.humap import humap_mpi

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
