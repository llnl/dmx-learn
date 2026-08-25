"""Automatic MPI estimation helpers used by heterogeneous embedding workflows."""

from typing import Any, Optional, Sequence, cast

import numpy as np

from dmx.bstats import MixtureDistribution, ParameterEstimator
from dmx.bstats.dpm import DirichletProcessMixtureEstimator
from dmx.mpi4py.utils import get_runtime_attr
from dmx.utils.automatic import get_estimator


# Keep the current helper call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def get_dpm_mixture_mpi(
    data: Optional[Sequence[Any]],
    estimator: Optional[ParameterEstimator] = None,
    max_comp: int = 20,
    rng: Optional[np.random.RandomState] = None,
    max_its: int = 1000,
    print_iter: int = 100,
    mix_threshold_count: float = 0.5,
) -> MixtureDistribution:
    """Fit and prune a Dirichlet process mixture with MPI.

    This is collective over `MPI.COMM_WORLD`; every rank must call it in the
    same order. Raw `data` is required only on rank 0, where the component
    estimator is inferred when omitted. The estimator is broadcast to workers,
    DPM fitting is performed collectively, pruning is computed on rank 0, and
    the pruned mixture is broadcast back to every rank. Rank 0 prints the
    retained weights and component count.

    Args:
        data (Optional[Sequence[Any]]): Data to model. Must be defined on rank 0.
        estimator (Optional[ParameterEstimator]): Base estimator to use. If
            omitted, rank 0 infers it from `data`.
        max_comp (int): Maximum number of components in the mixture.
        rng (Optional[numpy.random.RandomState]): Random number generator used
            during mixture optimization.
        max_its (int): Maximum number of iterations for optimization.
        print_iter (int): Frequency of printing iteration progress.
        mix_threshold_count (float): Component-count threshold. On rank 0 this
            is divided by `len(data)` to form the minimum retained weight.

    Returns:
        MixtureDistribution: Pruned mixture distribution on every rank.

    Raises:
        ValueError: If `data` is `None` on rank 0.
        RuntimeError: If collective DPM optimization does not return a model on
            rank 0.
    """
    mpi = get_runtime_attr("mpi4py", "MPI")
    optimize_mpi = get_runtime_attr("dmx.mpi4py.utils.bestimation", "optimize_mpi")

    # Get MPI communicator, rank, and size
    comm = mpi.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        if data is None:
            raise ValueError("Data must be defined on rank 0.")
        est = (
            estimator if estimator is not None else get_estimator(data, use_bstats=True)
        )
    else:
        est = None

    # broadcast estimator to each worker
    est = comm.bcast(est, root=0)

    est = DirichletProcessMixtureEstimator([est] * max_comp)

    # the model should live on world_rank == 0
    mix_model = optimize_mpi(data, est, max_its=max_its, rng=rng, print_iter=print_iter)

    if world_rank == 0:
        assert data is not None
        if mix_model is None:
            raise RuntimeError("DPM mixture optimization did not return a model.")
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

    mix_dist = cast(MixtureDistribution, comm.bcast(mix_dist, root=0))

    return mix_dist
