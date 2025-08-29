"""Automatic estimations for input data files. Use in auto-estimation step of htsne."""
from typing import Optional, Any, Sequence
from mpi4py import MPI

import numpy as np

from dmx.utils.automatic import get_estimator
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats import ParameterEstimator

def get_dpm_mixture_mpi(
        data: Sequence[Any], 
        estimator: Optional[ParameterEstimator] = None, 
        max_comp: int = 20, 
        rng: Optional[np.random.RandomState] = None, 
        max_its: int = 1000, 
        print_iter: int = 100, 
        mix_threshold_count: int = 0.5
    ) -> MixtureDistribution:
    """Gets a Dirichlet Process Mixture model for the data.

    Args:
        data (Sequence[Any]): The data to model.
        estimator (Optional[ParameterEstimator]): The base estimator to use.
        max_comp (int): Maximum number of components in the mixture.
        rng (Optinal[numpy.random.RandomState]): Random number generator.
        max_its (int): Maximum number of iterations for optimization.
        print_iter (int): Frequency of printing iteration progress.
        mix_threshold_count (float): Threshold for component weights.

    Returns:
        MixtureDistribution: A mixture distribution model.
    """
    from dmx.bstats.dpm import DirichletProcessMixtureEstimator
    from dmx.bstats.mixture import MixtureDistribution
    from dmx.mpi4py.utils.bestimation import optimize_mpi

    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        est = estimator if estimator else get_estimator(data, use_bstats=True)
    else:
        est = None

    # broadcast estimator to each worker
    est = comm.bcast(est, root=0)

    est = DirichletProcessMixtureEstimator([est]*max_comp)

    # the model should live on world_rank == 0
    mix_model = optimize_mpi(data, est, max_its=max_its, rng=rng, print_iter=print_iter)

    if world_rank == 0:
        thresh = mix_threshold_count/len(data)
        mix_comps = [mix_model.components[i] for i in np.flatnonzero(mix_model.w >= thresh)]
        mix_weights = mix_model.w[mix_model.w >= thresh]

        print(str(mix_weights))
        print('# Components = %d' % (len(mix_comps)))
        mix_dist = MixtureDistribution(mix_comps, mix_weights)
    else:
        mix_dist = None

    mix_dist = comm.bcast(mix_dist, root=0)

    return mix_dist