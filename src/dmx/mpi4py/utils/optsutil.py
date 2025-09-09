"""Helper functions for mpi4py"""
import pickle
from typing import Any
from mpi4py import MPI

def pickle_on_master(x: Any, filename: str) -> None:
    """Function for saving input to pickle file on master node."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        if x is None:
            raise Exception("Input can not be None on Rank 0.")

        with open(filename, "wb") as f:
            pickle.dump(x, f)
