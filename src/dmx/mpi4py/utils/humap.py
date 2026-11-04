"""MPI helpers for heterogeneous UMAP embeddings."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import umap  # type: ignore[import-untyped]

from dmx.bstats import MixtureDistribution as BMixtureDistribution
from dmx.bstats.pdist import ParameterEstimator
from dmx.mpi4py.utils import get_runtime_attr, resolve_rng_mpi
from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi
from dmx.mpi4py.utils.fisher import fisher_embedding_mpi
from dmx.stats import MixtureDistribution
from dmx.utils.fisher import FisherEmbedding, FisherScoreOptions
from dmx.utils.humap import (
    Features,
    Representation,
    report_embedding,
    resolve_humap_fisher_options,
    umap_options,
)
from dmx.utils.progress import ProgressCallback, ProgressReporter

T = TypeVar("T")
MIX_TYPE = Union[MixtureDistribution, BMixtureDistribution]
MPI = get_runtime_attr("mpi4py", "MPI")

logger = logging.getLogger(__name__)


# Keep the current public call signature stable; new options are keyword-only.
# pylint: disable-next=too-many-positional-arguments,too-many-arguments,too-many-locals
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
    verbose: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    representation: Representation = "posterior",
    fisher_scale: Optional[float] = None,
    top_m: Optional[int] = 3,
    fisher_options: Optional[FisherScoreOptions] = None,
    return_embedding: bool = False,
) -> Optional[Tuple[Any, ...]]:
    """Fit UMAP on posteriors or Fisher features of a DPM mixture model.

    This function participates in MPI collectives when it fits a mixture
    model or when ``representation="fisher"``. In those cases every rank must
    call it in the same order, with the same arguments, and raw `data` is
    required only on rank 0. If `mix_model` is precomputed, pass it
    consistently on all ranks so every process follows the same collective
    branch. Mixture fitting returns a model on all ranks. Posteriors and the
    UMAP fit are computed only on rank 0. Fisher features are computed in
    parallel: rank 0 scatters the records in contiguous blocks, every rank
    embeds its block with
    :func:`~dmx.mpi4py.utils.fisher.fisher_embedding_mpi`, and the rows are
    gathered back to rank 0; they equal the rows :func:`dmx.utils.humap.humap`
    would compute. See :func:`dmx.utils.humap.humap` for the Fisher options.

    Args:
        data (Optional[Sequence[T]]): Input data sequence. Must be defined
            on rank 0.
        max_components (int): Maximum number of components for the mixture model.
        mix_threshold_count (float): Threshold for mixture component selection.
        max_its (int): Maximum number of DPM fitting iterations.
        print_iter (int): Number of iteration to print while fitting the DPM.
        seed (Optional[int]): Seed for mixture fitting, used on every rank.
            When ``None`` and a mixture is fitted, rank 0 draws a fresh seed
            from OS entropy and broadcasts it; the seed used is sent to
            ``progress_callback`` as a ``"start"`` event and logged at DEBUG
            on rank 0. UMAP randomness is controlled through `umap_kwargs`,
            such as `random_state`.
        comp_estimator (Optional[ParameterEstimator]): Component estimator
            for mixture model.
        mix_model (Optional[MIX_TYPE]): Precomputed mixture model. Supplying one
            skips collective mixture fitting.
        umap_kwargs (Optional[Dict[str, Any]]): Kwargs for UMAP fit. Missing
            defaults are added in place; the default ``metric`` is
            ``"hellinger"`` for posteriors and ``"euclidean"`` for Fisher
            features, and a metric set by the caller is kept.
        verbose (bool): Write mixture-fitting progress to standard output.
            With ``False`` nothing is written; messages are still logged at
            DEBUG.
        progress_callback (Optional[ProgressCallback]): Optional function
            receiving the :class:`~dmx.utils.progress.ProgressEvent` objects
            from mixture fitting, including a ``"start"`` event with the seed,
            and in Fisher mode an ``"embedding"`` event, on rank 0 only.
        representation: ``"posterior"`` (default) or ``"fisher"``.
        fisher_scale: Weight ``c`` of the Fisher score blocks; None uses
            ``1 / p_eff`` over all records.
        top_m: Fisher score blocks kept per record; None keeps all.
        fisher_options: Options for the components' Fisher scores. None uses
            :data:`dmx.utils.humap.HUMAP_FISHER_OPTIONS`
            (``balance_fields=True``), as :func:`dmx.utils.humap.humap` does.
        return_embedding: Also return the
            :class:`~dmx.utils.fisher.FisherEmbedding` (None for posteriors)
            on rank 0, for :func:`dmx.utils.humap.humap_transform`.

    Returns:
        Optional[Tuple[Any, ...]]: On rank 0, returns embeddings with shape
        `(len(data), n_components)`, the mixture model, fitted UMAP object,
        and the features UMAP was fitted on: the posterior matrix with shape
        `(len(data), mix_model.num_components)`, or the CSR Fisher feature
        matrix. With ``return_embedding`` the Fisher embedding (or None) is
        appended. Worker ranks return `None`.

    Raises:
        ValueError: If `data` is missing on rank 0, `max_components <= 1`,
            `seed` is out of range, `representation` is unknown, or the
            mixture has no components.
        RuntimeError: If collective mixture fitting does not return a model.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0 and data is None:
        raise ValueError("Data must be defined on rank 0 for humap.")

    if max_components <= 1:
        max_components = int(max_components)
        raise ValueError("max_components must be and integer greater than 1.")

    # Validate before any collective, identically on every rank.
    options = umap_options(umap_kwargs, representation)

    reporter = ProgressReporter(
        "humap_mpi", logger, verbose=verbose, callback=progress_callback
    )
    # Fit DPM to data using comp_estimator if passed.
    if mix_model is None:
        rng, used_seed = resolve_rng_mpi(comm, seed)
        if world_rank == 0:
            reporter.start(used_seed)
        mix_model = get_dpm_mixture_mpi(
            data=data,
            estimator=comp_estimator,
            max_comp=max_components,
            rng=rng,
            max_its=max_its,
            print_iter=print_iter,
            mix_threshold_count=mix_threshold_count,
            verbose=verbose,
            progress_callback=progress_callback,
        )
    if mix_model is None:
        raise RuntimeError("DPM mixture fitting did not return a model.")

    # Mixture must have at least one comp!
    if mix_model.num_components == 0:
        raise ValueError("Something is broken. Mixture model has zero components.")

    embedding: Optional[FisherEmbedding] = None
    if representation == "fisher":
        local_data = comm.scatter(
            _contiguous_blocks(data, comm.Get_size()) if world_rank == 0 else None,
            root=0,
        )
        embedding = fisher_embedding_mpi(
            mix_model,
            local_data,
            comm,
            scale=fisher_scale,
            top_m=top_m,
            options=resolve_humap_fisher_options(fisher_options),
        )

    if world_rank == 0:
        assert data is not None
        features: Features
        if embedding is not None:
            report_embedding(reporter, embedding)
            features = embedding.matrix
        else:
            # This is until all bstats is updated!
            if isinstance(mix_model, BMixtureDistribution):
                enc_data = mix_model.seq_encode(data)
            else:
                encoder = mix_model.dist_to_encoder()
                enc_data = encoder.seq_encode(data)

            # Posterior and log comp density for each point [z | x] and [x | z]
            features = mix_model.seq_posterior(enc_data)

        fit = umap.UMAP(**options)
        embeddings = fit.fit_transform(features)

        if return_embedding:
            return embeddings, mix_model, fit, features, embedding
        return embeddings, mix_model, fit, features

    # Returns None for worker nodes
    return None


def _contiguous_blocks(data: Optional[Sequence[T]], parts: int) -> List[Sequence[T]]:
    """Split ``data`` into ``parts`` contiguous blocks of near-equal size."""
    assert data is not None
    count = len(data)
    bounds = [count * part // parts for part in range(parts + 1)]
    return [data[bounds[part] : bounds[part + 1]] for part in range(parts)]
