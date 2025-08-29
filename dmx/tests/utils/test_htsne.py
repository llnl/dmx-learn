"""Tests for heterogenous SNE functionality"""
import os
import pickle
from dmx.stats import *
from dmx.utils.htsne import htsne
import numpy as np

DATA_DIR = "dmx/tests/data"
ANSWER_DIR = "dmx/tests/answerkeys"

def test_htsne() -> None:
    """Test if HTSNE behaves as expected with data only input."""
    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    answer = np.load(os.path.join(ANSWER_DIR, "testOutput_htsne.npy"))
    rv = htsne(data, seed=10)

    assert np.all(answer == rv)