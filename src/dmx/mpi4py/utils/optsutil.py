"""Helper functions for mpi4py."""

import pickle
from typing import Any

from dmx.mpi4py.utils import get_runtime_attr


def pickle_on_master(x: Any, filename: str) -> None:
    """Function for saving input to pickle file on master node."""
    mpi = get_runtime_attr("mpi4py", "MPI")
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        if x is None:
            raise ValueError("Input cannot be None on rank 0.")

        with open(filename, "wb") as f:
            pickle.dump(x, f)
