"""Tests for mpi4py support on stats estimation functions.

Tests were run with mpiexec -n 4 pytest test_estimation

"""
import os
import pickle
import pytest
from dmx.stats import *
from dmx.mpi4py.utils.estimation import optimize_mpi, best_of_mpi
import numpy as np
from mpi4py import MPI

DATA_DIR = "dml/tests/data"
ANSWER_DIR = "dml/tests/answerkeys"

def test_optimize_mpi() -> None:
    """Test to ensure optimize works with mpi4py call."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, 'testInput_optimize_mpi_n4.pkl'), 'rb') as f:
            data = pickle.load(f)

        with open(os.path.join(ANSWER_DIR, 'testOutput_optimize_mpi_n4.pkl'), 'rb') as f:
            answer = pickle.load(f)
    else:
        data = None

    rng = np.random.RandomState(1)
    est0 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
    est = MixtureEstimator([est0] * 2)

    model = optimize_mpi(data, est, rng=rng)

    if world_rank == 0:
        assert str(model) == str(answer)
    else:
        assert model == None


def test_best_of_mpi() -> None:
    """Tests mpi4py on best of model fitting."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, 'testInput_optimize_mpi_n4.pkl'), 'rb') as f:
            data = pickle.load(f)

        data, vdata = data[:-10], data[-10:]
    else:
        data = None
        vdata = None

    with open(os.path.join(ANSWER_DIR, 'testOutput_best_of_mpi_n4.pkl'), 'rb') as f:
        answer = pickle.load(f)

    rng = np.random.RandomState(1)
    est0 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
    est = MixtureEstimator([est0] * 2)

    model = best_of_mpi(
        data=data,
        vdata=vdata,
        est=est, 
        max_its=100, 
        max_its_cnt=10, 
        init_p=0.10, 
        delta=1.0e-6, 
        trials=5, 
        rng=rng)

    if world_rank == 0:
        assert str(model) == str(answer)