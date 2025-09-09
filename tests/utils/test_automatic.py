"""Tests for utils/automatic.py"""
import os
import pytest
import pickle
import numpy as np
from dmx.utils.automatic import get_dpm_mixture, get_estimator
from dmx.bstats import *

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"

@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture(case_id: int) -> None:
    """Tests if pipeline for creating estimator and estiamting a DPM works."""
    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    model = get_dpm_mixture(data, rng=np.random.RandomState(1))

    with open(os.path.join(ANSWER_DIR, f"testOutput_automatic{case_id}.txt"), 'r') as f:
        answer = f.read()
    
    assert answer == str(model)

