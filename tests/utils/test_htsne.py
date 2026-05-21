"""Tests for heterogenous SNE functionality"""

import os
import pickle

from dmx.utils.htsne import htsne

DATA_DIR = "tests/data"


def test_htsne() -> None:
    """Test that HTSNE runs successfully with data-only input."""
    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
        data = pickle.load(f)

    htsne(data, seed=10)
    assert True
