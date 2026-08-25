"""MPI helpers for heterogeneous UMAP embeddings."""

from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import umap  # type: ignore[import-untyped]
from numpy.random import RandomState
from umap import UMAP

from dmx.bstats import MixtureDistribution as BMixtureDistribution
from dmx.bstats.pdist import ParameterEstimator
from dmx.mpi4py.utils import get_runtime_attr
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi
from dmx.stats import MixtureDistribution

T = TypeVar("T")
MIX_TYPE = Union[MixtureDistribution, BMixtureDistribution]
MPI = get_runtime_attr("mpi4py", "MPI")


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def humap_mpi(
    data: Optional[Sequence[T]],
    max_components: int = 30,
    mix_threshold_count: float = 0.5,
    max_its: int = 1000,
    print_iter: int = 100,
    seed: Optional[int] = None,
    comp_estimator: Optional[ParameterEstimator] = None,
    mix_model: Optional[MIX_TYPE] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Any, MIX_TYPE, UMAP, np.ndarray]]:
    """Fit UMAP on posteriors of a DPM mixture model.

    This function participates in MPI collectives only when it fits a mixture
    model. In that branch, every rank must call it in the same order and raw
    `data` is required only on rank 0. If `mix_model` is precomputed, pass it
    consistently on all ranks so every process follows the same collective
    branch. Mixture fitting returns a model on all ranks, but posterior
    evaluation and UMAP fitting happen only on rank 0.

    Args:
        data (Optional[Sequence[T]]): Input data sequence. Must be defined
            on rank 0.
        max_components (int): Maximum number of components for the mixture model.
        mix_threshold_count (float): Threshold for mixture component selection.
        max_its (int): Maximum number of DPM fitting iterations.
        print_iter (int): Number of iteration to print while fitting the DPM.
        seed (Optional[int]): Random seed for mixture fitting. UMAP randomness
            is controlled through `umap_kwargs`, such as `random_state`.
        comp_estimator (Optional[ParameterEstimator]): Component estimator
            for mixture model.
        mix_model (Optional[MIX_TYPE]): Precomputed mixture model. Supplying one
            skips collective mixture fitting.
        umap_kwargs (Optional[Dict[str, Any]]): Kwargs for UMAP fit. The
            dictionary is mutated in place to set `metric="hellinger"`.

    Returns:
        Optional[Tuple[Any, MIX_TYPE, UMAP, numpy.ndarray]]: On rank 0, returns
        embeddings with shape `(len(data), n_components)`, the mixture model,
        fitted UMAP object, and posterior matrix with shape
        `(len(data), mix_model.num_components)`. Worker ranks return `None`.

    Raises:
        ValueError: If `data` is missing on rank 0, `max_components <= 1`, or
            the mixture has no components.
        RuntimeError: If collective mixture fitting does not return a model.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0 and data is None:
        raise ValueError("Data must be defined on rank 0 for humap.")

    rng = RandomState(seed) if seed is not None else RandomState()
    if max_components <= 1:
        max_components = int(max_components)
        raise ValueError("max_components must be and integer greater than 1.")

    # Fit DPM to data using comp_estimator if passed.
    if mix_model is None:
        mix_model = get_dpm_mixture_mpi(
            data=data,
            estimator=comp_estimator,
            max_comp=max_components,
            rng=rng,
            max_its=max_its,
            print_iter=print_iter,
            mix_threshold_count=mix_threshold_count,
        )
    if mix_model is None:
        raise RuntimeError("DPM mixture fitting did not return a model.")

    # Mixture must have at least one comp!
    if mix_model.num_components == 0:
        raise ValueError("Something is broken. Mixture model has zero components.")

    if world_rank == 0:
        assert data is not None
        # This is until all bstats is updated!
        if isinstance(mix_model, BMixtureDistribution):
            enc_data = mix_model.seq_encode(data)
        else:
            encoder = mix_model.dist_to_encoder()
            enc_data = encoder.seq_encode(data)

        # Posterior and log comp density for each point [z | x] and [x | z]
        posteriors = mix_model.seq_posterior(enc_data)

        # Currently only take hellinger
        if umap_kwargs is None:
            umap_kwargs = {}
        umap_kwargs["metric"] = "hellinger"

        fit = umap.UMAP(**umap_kwargs)
        embeddings = fit.fit_transform(posteriors)

        return embeddings, mix_model, fit, posteriors

    # Returns None for worker nodes
    return None
