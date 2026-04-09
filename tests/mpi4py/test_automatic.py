"""Tests for automatic.py with mpi4py support using 4 cores."""

import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI

from dmx.bstats import *
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture_mpi(case_id: int) -> None:
    """Tests if pipeline for creating estimator and estiamting a DPM works."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), "rb") as f:
        data = pickle.load(f)

    model = get_dpm_mixture_mpi(data, rng=np.random.RandomState(1))

    with open(
        os.path.join(
            ANSWER_DIR, f"testOutput_automatic_get_dpm_mixture_mpi_n4_case{case_id}.txt"
        ),
        "r",
    ) as f:
        answer = f.read()

    assert answer == str(model)
