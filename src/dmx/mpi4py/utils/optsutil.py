"""Helper functions for mpi4py."""

import pickle
from typing import Any

from dmx.mpi4py.utils import get_runtime_attr


def pickle_on_master(x: Any, filename: str) -> None:
    """Write an object to a pickle file only on rank 0.

    This helper is not collective; it only checks the caller's
    `MPI.COMM_WORLD` rank. Rank 0 requires a non-`None` object and opens
    `filename` in overwrite mode. Worker ranks return without opening the file.
    Standard pickle security rules apply: only unpickle files from trusted
    sources.

    Args:
        x (Any): Object to pickle on rank 0.
        filename (str): Destination file path.

    Raises:
        ValueError: If `x` is `None` on rank 0.
        OSError: If rank 0 cannot open or write `filename`.
        pickle.PickleError: If rank 0 cannot pickle `x`.
    """
    mpi = get_runtime_attr("mpi4py", "MPI")
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        if x is None:
            raise ValueError("Input cannot be None on rank 0.")

        with open(filename, "wb") as f:
            pickle.dump(x, f)
