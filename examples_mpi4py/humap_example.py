"""Example of fitting heterogenous umap fitting with mpi4py.

Run with mpiexec -n 4 python3 examples_mpi4py/humap_example.py

"""

# pylint: disable=duplicate-code

import os
import pickle

from mpi4py import MPI  # pylint: disable=no-name-in-module

from dmx.mpi4py.utils.humap import humap_mpi
from dmx.mpi4py.utils.optsutil import pickle_on_master

PATH_TO_DATA = "examples_mpi4py/data"

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()


if __name__ == "__main__":

    if world_rank == 0:

        with open(os.path.join(PATH_TO_DATA, "sample_data.pkl"), "rb") as f:
            data = pickle.load(f)

    else:
        data = None

    # These are the parameters that are passed to UMAP fit
    umap_kwargs = {"n_neighbors": 15, "min_dist": 0.2, "random_state": 42}

    results = humap_mpi(data=data, seed=1, umap_kwargs=umap_kwargs)

    # you can access the results on the master node
    if world_rank == 0:
        # UMAP embeddings, mixture model fit, the UMAP fit, and the posteriors.
        embeddings, mix_model, fit, posteriors = results

        rv = {
            "embeddings": embeddings,
            "mix_model": mix_model,
            "umap_fit": fit,
            "posteriors": posteriors,
        }

    else:
        rv = None

    # save results on master
    pickle_on_master(rv, "humap_mpi_results.pkl")
