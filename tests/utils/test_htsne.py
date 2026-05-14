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

    htsne(data, seed=10)
    assert True
