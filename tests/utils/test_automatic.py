"""Tests for utils/automatic.py"""

import os
import pickle

import numpy as np
import pytest

from dmx.bstats import CategoricalEstimator, MixtureDistribution
from dmx.utils.automatic import get_dpm_mixture, get_estimator

DATA_DIR = "tests/data"


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture(case_id: int) -> None:
    """Test that the DPM estimation pipeline runs successfully."""
    with open(os.path.join(DATA_DIR, f"testInput_automatic{case_id}.pkl"), "rb") as f:
        data = pickle.load(f)

    model = get_dpm_mixture(data, rng=np.random.RandomState(1))
    assert isinstance(model, MixtureDistribution)


def test_get_estimator_returns_bstats_estimator() -> None:
    """Route automatic estimation through bstats when requested."""
    estimator = get_estimator(["a", "b"], use_bstats=True)

    assert isinstance(estimator, CategoricalEstimator)
