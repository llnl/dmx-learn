"""Test integration with UMAP."""

import os
import pickle

from dmx.utils.humap import humap

DATA_DIR = "tests/data"


def test_humap() -> None:
    """Test for humap using automatic fitting."""

    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), "rb") as f:
        data = pickle.load(f)

    umap_kwargs = {
        "n_neighbors": 15,
        "min_dist": 0.2,
        "random_state": 42,  # Set your desired seed here
    }

    humap(data, seed=10, umap_kwargs=umap_kwargs)

    assert True
