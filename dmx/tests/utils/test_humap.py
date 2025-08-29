"""Test integration with UMAP."""
import os
import pickle
from dmx.stats import *
from dmx.utils.humap import humap
import numpy as np

DATA_DIR = "dmx/tests/data"
ANSWER_DIR = "dmx/tests/answerkeys"

def test_humap() -> None:
    """Test for humap using automatic fitting."""

    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), 'rb') as f:
        data = pickle.load(f)

    umap_kwargs = {
    'n_neighbors': 15,
    'min_dist': 0.2,
    'random_state': 42  # Set your desired seed here
    }

    embeddings, mix_model, fit, posteriors = humap(data, seed=10, umap_kwargs=umap_kwargs)

    with open(os.path.join(ANSWER_DIR, "testOutput_humap.pkl"), "rb") as f:
        answer_dict = pickle.load(f)

    assert np.all(answer_dict['embeddings'] == embeddings)
    assert str(mix_model) == answer_dict['mix_model']
    assert str(fit) == answer_dict['fit']
    assert np.all(posteriors == answer_dict['posteriors'])