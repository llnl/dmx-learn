"""Update the data and answer keys for htsne and humap tests."""

import pickle

import numpy as np

from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)
from dmx.utils.htsne import htsne
from dmx.utils.humap import humap

if __name__ == "__main__":
    rng = np.random.RandomState(121)
    # create a three state mixture
    mu = np.linspace(-3.0, 3.0, 3)
    p = np.asarray([0.1, 0.3, 0.6])
    vals = ["a", "b", "c"]
    comps = []

    for i in range(3):
        pmap = {v: p[(i + j) % 3].item() for v, j in zip(vals, range(3))}
        d0 = CategoricalDistribution(pmap=pmap)
        d1 = GaussianDistribution(mu=mu[i], sigma2=1.0)

        comps.append(CompositeDistribution(dists=[d0, d1]))

    dist = MixtureDistribution(comps, w=np.ones(3) / 3.0)

    data = dist.sampler(rng.randint(low=0, high=2**31 - 1)).sample(300)

    # write out htsne data file
    with open("tests/data/testInput_htsne.pkl", "wb") as f:
        pickle.dump(data, f)

    # run htsne
    htsne_answerkey = htsne(data, seed=10)
    np.save("tests/data/testOutput_htsne.npy", htsne_answerkey)

    # update humap
    umap_kwargs = {
        "n_neighbors": 15,
        "min_dist": 0.2,
        "random_state": 42,  # Set your desired seed here
    }

    embeddings, _, _, _ = humap(data, seed=10, umap_kwargs=umap_kwargs)

    np.save("tests/answerkeys/testOutput_humap.npy", embeddings)

    # Model Dump for mpi4py
    with open("tests/data/testInput_mpi_optimize.pkl", "wb") as f:
        pickle.dump(dist, f)
