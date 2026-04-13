"""Tests for heterogenous SNE functionality"""

import os
import pickle

import numpy as np

from dmx.stats import *
from dmx.utils.htsne import htsne

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"


def test_htsne() -> None:
    """Test if HTSNE behaves as expected with data only input."""
    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
        data = pickle.load(f)

    answer = np.load(os.path.join(ANSWER_DIR, "testOutput_htsne.npy"))
    rv = htsne(data, seed=10)

    # Use approximate equality to handle platform/version differences in numerical operations
    # rtol=1e-5 allows ~0.001% relative difference, atol=1e-7 handles near-zero values
    assert np.allclose(answer, rv, rtol=1e-5, atol=1e-7)
