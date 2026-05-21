"""Tests for mpi4py support on stats estimation functions.

Tests were run with mpiexec -n 4 pytest test_estimation

"""

# pylint: disable=duplicate-code

import os
import pickle

import numpy as np
from mpi4py import MPI  # pylint: disable=no-name-in-module

from dmx.mpi4py.utils.estimation import best_of_mpi, optimize_mpi
from dmx.stats import seq_encode
from dmx.utils.estimation import empirical_kl_divergence

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"


def test_optimize_mpi() -> None:
    """Test to ensure optimize works with mpi4py call."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, "testInput_mpi_optimize.pkl"), "rb") as f:
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
        assert model is None, "Model was broadcast to worker!"


def test_best_of_mpi() -> None:
    """Tests mpi4py on best of model fitting."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, "testInput_mpi_optimize.pkl"), "rb") as f:
            true_model = pickle.load(f)
            data = true_model.sampler(10).sample(4000)

            data, vdata = data[:-400], data[-400:]
            est = true_model.estimator()
    else:
        data = None
        vdata = None
        est = None

    est = comm.bcast(est, root=0)
    rng = np.random.RandomState(1)

    model = best_of_mpi(
        data=data,
        vdata=vdata,
        est=est,
        max_its=100,
        print_iter=1000,
        max_its_cnt=1000,
        init_p=0.10,
        delta=1.0e-6,
        trials=5,
        rng=rng,
    )

    if world_rank == 0:
        enc_data = seq_encode(data=data, model=true_model)
        kl, _, _ = empirical_kl_divergence(
            dist1=true_model, dist2=model, enc_data=enc_data
        )
        assert (
            kl <= 1.0e-2
        ), f"Model estimate did not converge under empirical KL: {kl}."
    else:
        assert model is None, "Model was broadcast to worker!"
