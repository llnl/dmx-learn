"""Test integration with UMAP."""

import os
import pickle

import numpy as np

from dmx.stats import *
from dmx.utils.humap import humap

DATA_DIR = "tests/data"
ANSWER_DIR = "tests/answerkeys"


def test_humap() -> None:
    """Test for humap using automatic fitting."""

    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
        data = pickle.load(f)

    umap_kwargs = {
        "n_neighbors": 15,
        "min_dist": 0.2,
        "random_state": 42,  # Set your desired seed here
    }

    embeddings, mix_model, fit, posteriors = humap(
        data, seed=10, umap_kwargs=umap_kwargs
    )

    with open(os.path.join(ANSWER_DIR, "testOutput_humap.pkl"), "rb") as f:
        answer_dict = pickle.load(f)

    # Use approximate equality for numerical arrays to handle platform/version differences
    # rtol=1e-5 allows ~0.001% relative difference, atol=1e-7 handles near-zero values
    assert np.allclose(answer_dict["embeddings"], embeddings, rtol=1e-5, atol=1e-7)
    # assert str(mix_model) == answer_dict["mix_model"]
    # assert str(fit) == answer_dict["fit"]
    assert np.allclose(posteriors, answer_dict["posteriors"], rtol=1e-5, atol=1e-7)
