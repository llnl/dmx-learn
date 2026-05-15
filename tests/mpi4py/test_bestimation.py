"""Tests for mpi4py functionality on bstats estimation functions.

All tests are run with mpiexec -n 4 pytest test_bestimation

"""

import os
import pickle

import numpy as np
from mpi4py import MPI

from dmx.bstats import *
from dmx.bstats.bestimation import empirical_kl_divergence
from dmx.mpi4py.bstats import *
from dmx.mpi4py.utils.bestimation import optimize_mpi

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"


def test_bestimation_optimize_mpi() -> None:
    """Test bstats optimize mpi call with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, "testInput_mpi_b_optimize.pkl"), "rb") as f:
            true_model = pickle.load(f)
        data = true_model.sampler(10).sample(4000)
        est = true_model.estimator()
    else:
        data = None
        est = None

    est = comm.bcast(est, root=0)
    rng = np.random.RandomState(1)

    model = optimize_mpi(data, est, max_its=10000, print_iter=10000, rng=rng)

    if world_rank == 0:
        enc_data = seq_encode(data=data, model=true_model)
        kl, _, _ = empirical_kl_divergence(
            dist1=true_model, dist2=model, enc_data=enc_data
        )
        assert (
            kl <= 1.0e-2
        ), f"Model estimate did not converge under empirical KL: {kl}."
    else:
        assert model == None, f"Model was broadcast to worker!"
