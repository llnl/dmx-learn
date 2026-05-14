"""Automatic estimations for input data files. Use in auto-estimation step of htsne."""

from importlib import import_module
from typing import Any, Optional, Sequence

import numpy as np

from dmx.bstats import ParameterEstimator
from dmx.bstats.mixture import MixtureDistribution
from dmx.utils.automatic import get_estimator


# Keep MPI and related optional imports runtime-loaded for lintability.
def _get_attr(module_name: str, attr_name: str) -> Any:
    """Load an attribute at runtime from a module."""
    return getattr(import_module(module_name), attr_name)


# Keep the current helper call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def get_dpm_mixture_mpi(
    data: Sequence[Any],
    estimator: Optional[ParameterEstimator] = None,
    max_comp: int = 20,
    rng: Optional[np.random.RandomState] = None,
    max_its: int = 1000,
    print_iter: int = 100,
    mix_threshold_count: int = 0.5,
) -> MixtureDistribution:
    """Gets a Dirichlet Process Mixture model for the data.

    Args:
        data (Sequence[Any]): The data to model.
        estimator (Optional[ParameterEstimator]): The base estimator to use.
        max_comp (int): Maximum number of components in the mixture.
        rng (Optional[numpy.random.RandomState]): Random number generator.
        max_its (int): Maximum number of iterations for optimization.
        print_iter (int): Frequency of printing iteration progress.
        mix_threshold_count (float): Threshold for component weights.

    Returns:
        MixtureDistribution: A mixture distribution model.
    """
    mpi = _get_attr("mpi4py", "MPI")
    dirichlet_process_mixture_estimator = _get_attr(
        "dmx.bstats.dpm", "DirichletProcessMixtureEstimator"
    )
    optimize_mpi = _get_attr("dmx.mpi4py.utils.bestimation", "optimize_mpi")

    # Get MPI communicator, rank, and size
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        est = (
            estimator if estimator is not None else get_estimator(data, use_bstats=True)
        )
    else:
        est = None

    # broadcast estimator to each worker
    est = comm.bcast(est, root=0)

    est = dirichlet_process_mixture_estimator([est] * max_comp)

    # the model should live on world_rank == 0
    mix_model = optimize_mpi(data, est, max_its=max_its, rng=rng, print_iter=print_iter)

    if world_rank == 0:
        thresh = mix_threshold_count / len(data)
        mix_comps = [
            mix_model.components[i] for i in np.flatnonzero(mix_model.w >= thresh)
        ]
        mix_weights = mix_model.w[mix_model.w >= thresh]

        print(str(mix_weights))
        print(f"# Components = {len(mix_comps)}")
        mix_dist = MixtureDistribution(mix_comps, mix_weights)
    else:
        mix_dist = None

    mix_dist = comm.bcast(mix_dist, root=0)

    return mix_dist
