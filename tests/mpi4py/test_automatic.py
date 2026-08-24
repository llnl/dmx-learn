"""Tests for automatic.py with mpi4py support using 4 cores."""

# pylint: disable=duplicate-code

import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI  # pylint: disable=no-name-in-module

from dmx.bstats import MixtureDistribution
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi

DATA_DIR = "tests/data"


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture_mpi(case_id: int) -> None:
    """Test that MPI automatic fitting returns a usable bstats mixture."""
    comm = MPI.COMM_WORLD
    comm.Get_rank()

    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), "rb") as f:
        data = pickle.load(f)

    model = get_dpm_mixture_mpi(data, rng=np.random.RandomState(1))

    assert isinstance(model, MixtureDistribution)
    assert model.num_components == len(model.components)
    assert model.num_components > 0
    np.testing.assert_allclose(model.w.sum(), 1.0)
