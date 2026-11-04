"""Embed heterogeneous observations using a fitted mixture and UMAP.

:func:`humap` represents each record by its mixture posteriors (the default)
or by its Fisher-geometry features (``representation="fisher"``), which also
keep where the record sits inside its component; see
``docs/utils/fisher_embedding.rst``. :func:`humap_transform` places new
records in an existing embedding.
"""

import logging
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import umap  # type: ignore[import-untyped]
from numpy.random import RandomState
from scipy.sparse import csr_matrix
from umap import UMAP

from dmx.utils.automatic import MixtureModel, encode_mixture_data, prepare_mixture_model
from dmx.utils.fisher import FisherEmbedding, FisherScoreOptions, fisher_embedding
from dmx.utils.progress import ProgressCallback, ProgressReporter
from dmx.utils.seeding import resolve_seed

logger = logging.getLogger(__name__)

DATUM_TYPE = TypeVar("DATUM_TYPE")

Representation = Literal["posterior", "fisher"]
Features = Union[np.ndarray, csr_matrix]

DEFAULT_UMAP = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.10,
    "metric": "hellinger",
}

_REPRESENTATIONS = ("posterior", "fisher")


def umap_options(
    umap_kwargs: Optional[Dict[str, Any]], representation: Representation
) -> Dict[str, Any]:
    """Return UMAP options with defaults filled in for ``representation``.

    Missing defaults are inserted into a caller-supplied dictionary in place,
    as :func:`humap` documents. Without one, a fresh copy of
    :data:`DEFAULT_UMAP` is used, so per-call changes never leak into later
    calls. The default metric is ``"hellinger"`` for posteriors and
    ``"euclidean"`` for Fisher features; a metric set by the caller is kept.

    Raises:
        ValueError: If ``representation`` is unknown.
    """
    if representation not in _REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {_REPRESENTATIONS}, "
            f"got {representation!r}."
        )
    options = {} if umap_kwargs is None else umap_kwargs
    defaults = dict(DEFAULT_UMAP)
    if representation == "fisher":
        defaults["metric"] = "euclidean"
    for key, value in defaults.items():
        options.setdefault(key, value)
    return options


def report_embedding(reporter: ProgressReporter, embedding: FisherEmbedding) -> None:
    """Log the Fisher features' scale, ``p_eff``, ``top_m`` and dimension.

    Sends an ``"embedding"`` event with those values as metrics. ``p_eff`` is
    recomputed from the posterior block, so it is reported even when the
    scale was given explicitly.
    """
    matrix = embedding.matrix
    posteriors = np.square(matrix[:, embedding.posterior_slice].toarray(), dtype=float)
    widths = np.zeros(posteriors.shape[1])
    for index, columns in embedding.component_slices.items():
        widths[index] = columns.stop - columns.start
    p_eff = float((posteriors @ widths).mean()) if posteriors.size else 0.0
    metrics = {
        "representation": "fisher",
        "scale": embedding.scale,
        "p_eff": p_eff,
        "top_m": embedding.top_m,
        "feature_dim": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
    }
    reporter.logger.debug(
        "%s Fisher features: scale=%g p_eff=%g top_m=%s feature_dim=%d nnz=%d",
        reporter.source,
        metrics["scale"],
        p_eff,
        metrics["top_m"],
        metrics["feature_dim"],
        metrics["nnz"],
    )
    reporter.event("embedding", metrics=metrics)


@overload
# pylint: disable-next=too-many-positional-arguments
def humap(
    data: Sequence[DATUM_TYPE],
    max_components: int = ...,
    mix_threshold_count: float = ...,
    max_its: int = ...,
    print_iter: int = ...,
    seed: Optional[int] = ...,
    comp_estimator: Optional[Any] = ...,
    mix_model: Optional[MixtureModel] = ...,
    umap_kwargs: Optional[Dict[str, Any]] = ...,
    verbose: bool = ...,
    progress_callback: Optional[ProgressCallback] = ...,
    *,
    representation: Representation = ...,
    fisher_scale: Optional[float] = ...,
    top_m: Optional[int] = ...,
    fisher_options: Optional[FisherScoreOptions] = ...,
    return_embedding: Literal[False] = ...,
) -> Tuple[Any, MixtureModel, UMAP, Features]:
    """Return the four-value result when ``return_embedding`` is false."""


@overload
# pylint: disable-next=too-many-positional-arguments
def humap(
    data: Sequence[DATUM_TYPE],
    max_components: int = ...,
    mix_threshold_count: float = ...,
    max_its: int = ...,
    print_iter: int = ...,
    seed: Optional[int] = ...,
    comp_estimator: Optional[Any] = ...,
    mix_model: Optional[MixtureModel] = ...,
    umap_kwargs: Optional[Dict[str, Any]] = ...,
    verbose: bool = ...,
    progress_callback: Optional[ProgressCallback] = ...,
    *,
    representation: Representation = ...,
    fisher_scale: Optional[float] = ...,
    top_m: Optional[int] = ...,
    fisher_options: Optional[FisherScoreOptions] = ...,
    return_embedding: Literal[True],
) -> Tuple[Any, MixtureModel, UMAP, Features, Optional[FisherEmbedding]]:
    """Return the result with the Fisher embedding appended."""


# Keep the current public call signature stable; new options are keyword-only.
# pylint: disable-next=too-many-positional-arguments,too-many-arguments,too-many-locals
def humap(
    data: Sequence[DATUM_TYPE],
    max_components: int = 30,
    mix_threshold_count: float = 0.5,
    max_its: int = 1000,
    print_iter: int = 100,
    seed: Optional[int] = None,
    comp_estimator: Optional[Any] = None,
    mix_model: Optional[MixtureModel] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    representation: Representation = "posterior",
    fisher_scale: Optional[float] = None,
    top_m: Optional[int] = 3,
    fisher_options: Optional[FisherScoreOptions] = None,
    return_embedding: bool = False,
) -> Union[
    Tuple[Any, MixtureModel, UMAP, Features],
    Tuple[Any, MixtureModel, UMAP, Features, Optional[FisherEmbedding]],
]:
    """Fit UMAP to mixture posteriors or Fisher-geometry features.

    A supplied mixture is reused; otherwise a pruned Dirichlet-process mixture
    is fitted. ``seed`` controls only mixture fitting. UMAP randomness is
    controlled independently through ``umap_kwargs`` (for example,
    ``random_state``). Missing default UMAP options are inserted into a
    caller-supplied dictionary in place.

    With ``representation="fisher"`` each record is represented by
    ``[sqrt(z_i), sqrt(c z_ik) s_ik for its top_m components]``, its
    posteriors plus whitened Fisher scores under its most probable
    components, so records of one component are no longer all at distance
    zero. The features are a sparse matrix, compared with the Euclidean
    metric by default. ``fisher_scale=0`` gives the posterior geometry. See
    ``docs/utils/fisher_embedding.rst`` and
    :func:`dmx.utils.fisher.fisher_embedding`.

    Args:
        data: Nonempty heterogeneous observations.
        max_components: Maximum components used when fitting a mixture.
        mix_threshold_count: Minimum effective component count retained after
            mixture fitting.
        max_its: Maximum mixture-fitting iterations.
        print_iter: Interval for mixture-fitting progress output.
        seed: Seed for the random state used in mixture fitting. ``None``
            draws a fresh seed from OS entropy; the seed used is sent to
            ``progress_callback`` as a ``"start"`` event and logged at DEBUG.
            UMAP randomness is separate: set ``umap_kwargs["random_state"]``.
        comp_estimator: Optional estimator for one mixture component.
        mix_model: Optional pre-fitted mixture distribution.
        umap_kwargs: Options passed to :class:`umap.UMAP`. Defaults are added
            for missing keys; the default ``metric`` is ``"hellinger"`` for
            posteriors and ``"euclidean"`` for Fisher features.
        verbose: Write progress messages to standard output. With
            ``False`` nothing is written; messages are still logged at DEBUG.
        progress_callback: Optional function receiving a
            :class:`~dmx.utils.progress.ProgressEvent` for each mixture-fitting
            update and the fitted mixture; see
            :func:`dmx.utils.automatic.get_dpm_mixture`. In Fisher mode it also
            receives an ``"embedding"`` event with the scale, ``p_eff``,
            ``top_m`` and feature dimension.
        representation: ``"posterior"`` (default) or ``"fisher"``.
        fisher_scale: Weight ``c`` of the Fisher score blocks. None uses
            ``1 / p_eff``, with ``p_eff`` the posterior-weighted mean score
            dimension. Ignored for posteriors.
        top_m: Score blocks kept per record, for its most probable components;
            None keeps all. Ignored for posteriors.
        fisher_options: Options for the components' Fisher scores. Ignored for
            posteriors.
        return_embedding: Also return the
            :class:`~dmx.utils.fisher.FisherEmbedding` (None for posteriors),
            for use with :func:`humap_transform`.

    Returns:
        The embedding with shape ``(len(data), n_components)``, fitted mixture
        model, fitted UMAP object, and the features UMAP was fitted on: the
        posterior matrix with shape ``(len(data), n_mixture_components)``, or
        the CSR Fisher feature matrix. With ``return_embedding``, the
        Fisher embedding (or None) is appended.

    Raises:
        ValueError: If ``max_components`` is not an integer greater than one,
            ``representation`` is unknown, a Fisher option is out of range, or
            UMAP rejects its inputs or options.
        RuntimeError: If mixture fitting produces no components.
        NotImplementedError: In Fisher mode, if a component has no Fisher
            score and ``fisher_options.on_missing == "raise"``.
    """
    options = umap_options(umap_kwargs, representation)
    seed = resolve_seed(seed)
    reporter = ProgressReporter(
        "humap", logger, verbose=verbose, callback=progress_callback
    )
    reporter.start(seed)
    rng = RandomState(seed)
    mix_model, _, posteriors = prepare_mixture_model(
        data,
        rng,
        max_components,
        mix_threshold_count,
        max_its,
        print_iter,
        comp_estimator,
        mix_model,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    embedding: Optional[FisherEmbedding] = None
    features: Features = posteriors
    if representation == "fisher":
        embedding = fisher_embedding(
            mix_model,
            data,
            scale=fisher_scale,
            top_m=top_m,
            options=fisher_options,
        )
        report_embedding(reporter, embedding)
        features = embedding.matrix

    fit = umap.UMAP(**options)
    embeddings = fit.fit_transform(features)

    if return_embedding:
        return embeddings, mix_model, fit, features, embedding
    return embeddings, mix_model, fit, features


def humap_transform(
    fit: UMAP,
    mix_model: MixtureModel,
    new_data: Sequence[DATUM_TYPE],
    embedding: Optional[FisherEmbedding] = None,
) -> Any:
    """Place new records in an embedding fitted by :func:`humap`.

    Computes the same representation the fit used, from the same mixture, and
    calls ``fit.transform``.

    Args:
        fit: Fitted UMAP object returned by :func:`humap`.
        mix_model: Mixture returned by :func:`humap`.
        new_data: Records to place.
        embedding: The :class:`~dmx.utils.fisher.FisherEmbedding` returned by
            ``humap(..., representation="fisher", return_embedding=True)``.
            None means the fit used posteriors.

    Returns:
        Coordinates with shape ``(len(new_data), n_components)``.
    """
    features: Features
    if embedding is None:
        features = mix_model.seq_posterior(encode_mixture_data(new_data, mix_model))
    else:
        features = fisher_embedding(mix_model, new_data, like=embedding).matrix
    return fit.transform(features)
