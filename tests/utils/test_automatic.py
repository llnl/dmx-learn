"""Tests for utils/automatic.py"""

import os
import pickle

import numpy as np
import pytest

from dmx.utils.automatic import get_dpm_mixture

DATA_DIR = "tests/data"


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture(case_id: int) -> None:
    """Test that the DPM estimation pipeline runs successfully."""
    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), "rb") as f:
        data = pickle.load(f)

    get_dpm_mixture(data, rng=np.random.RandomState(1))
    assert True
