"""Tests for automatic.py with mpi4py support using 4 cores."""
import os
import pytest
import pickle
from mpi4py import MPI
import numpy as np
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi
from dmx.bstats import *

DATA_DIR = "dmx/tests/data"
ANSWER_DIR = "dmx/tests/answerkeys"


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture_mpi(case_id: int) -> None:
    """Tests if pipeline for creating estimator and estiamting a DPM works."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    model = get_dpm_mixture_mpi(data, rng=np.random.RandomState(1))

    with open(os.path.join(ANSWER_DIR, f"testOutput_automatic_get_dpm_mixture_mpi_n4_case{case_id}.txt"), 'r') as f:
        answer = f.read()
    
    assert answer == str(model)

