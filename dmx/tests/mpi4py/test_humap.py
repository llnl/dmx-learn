"""Tests for mpi4py automatic model fitting and humap embeddings.

Run with mpiexec -n 4 pytest test_humap.py

"""
import os
import pickle
from dmx.bstats import *
from dmx.mpi4py.bstats import *
from dmx.mpi4py.utils.humap import humap_mpi
import numpy as np
from mpi4py import MPI

DATA_DIR = "dmx/tests/data"
ANSWER_DIR = "dmx/tests/answerkeys"


def test_humap_mpi() -> None:
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:

        with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), 'rb') as f:
            data = pickle.load(f)

    else:
        data = None

    umap_kwargs = {
        'n_neighbors': 15,
        'min_dist': 0.2,
        'random_state': 42  # Set your desired seed here
    }


    results = humap_mpi(data=data, seed=10, umap_kwargs=umap_kwargs)
    
    if world_rank == 0:
        embeddings, mix_model, fit, posteriors = results

        with open(os.path.join(ANSWER_DIR, "testOutput_humap_mpi_n4.pkl"), 'rb') as f:
            answer_dict = pickle.load(f)

        assert np.all(answer_dict['embeddings'] == embeddings)
        assert str(mix_model) == answer_dict['mix_model']
        assert str(fit) == answer_dict['fit']
        assert np.all(posteriors == answer_dict['posteriors'])