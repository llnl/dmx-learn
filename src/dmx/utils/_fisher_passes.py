"""Private passes over local records behind the Fisher embedding functions.

:func:`dmx.utils.fisher.fisher_embedding`, its MPI version in
:mod:`dmx.mpi4py.utils.fisher`, and its Spark version in
:mod:`dmx.utils.spark` run the same passes over the records each process
holds and differ only in how the per-process results are added:

1. with ``whiten``, :func:`whitening_pass` returns posterior-weighted score
   moments packed in one float64 vector; vectors add elementwise;
2. with the default scale, :func:`scale_terms` of the posteriors returns an
   exact integer sum and a count; terms add elementwise and give the same
   scale however the records were split;
3. :func:`rows_pass` computes the rows of the local records.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack

from dmx.utils._fisher_mixture import (
    ScaleTerms,
    ScoreMoments,
    add_score_moments,
    check_embedding_args,
    embedding_layout,
    fit_component_whiteners,
    mixture_component_data,
    scale_from_terms,
    scale_terms,
)

__all__ = [
    "EmbeddingSetup",
    "ScaleTerms",
    "deduplicated_warnings",
    "fit_whiteners",
    "make_embedding",
    "posterior_pass",
    "resolve_setup",
    "rows_pass",
    "scale_dims",
    "scale_from_terms",
    "scale_terms",
    "whitening_pass",
]
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherEmbedding,
    FisherScoreOptions,
    WhiteningMethod,
    resolve_fisher_options,
)

Whiteners = Optional[List[Optional[EmpiricalFisherWhitener]]]


class EmbeddingSetup(NamedTuple):
    """Resolved arguments of a Fisher embedding call.

    Attributes:
        scale: Explicit scale, or None for the default.
        top_m: Score blocks kept per record.
        options: Score options.
        dtype: Output dtype.
        whiteners: Whiteners reused from ``like``, if any.
        template: Unfitted whitener to fit on the data, if ``whiten`` was
            given.
    """

    scale: Optional[float]
    top_m: Optional[int]
    options: FisherScoreOptions
    dtype: Any
    whiteners: Whiteners
    template: Optional[EmpiricalFisherWhitener]


def resolve_setup(
    model: Any,
    *,
    scale: Optional[float],
    top_m: Optional[int],
    options: Optional[FisherScoreOptions],
    chunk_size: Optional[int],
    whiten: Optional[Any],
    like: Optional[FisherEmbedding],
    dtype: Any,
) -> EmbeddingSetup:
    """Validate the arguments and apply ``like``.

    Raises:
        TypeError: If ``model`` has no ``seq_fisher_embedding``.
        ValueError: If an argument is out of range, or ``scale``, ``options``
            or ``whiten`` is given with ``like``.
    """
    if not hasattr(model, "seq_fisher_embedding"):
        raise TypeError(
            f"{type(model).__name__} has no seq_fisher_embedding(); use a "
            "dmx.stats or dmx.bstats MixtureDistribution."
        )
    whiteners: Whiteners = None
    if like is not None:
        if scale is not None or options is not None or whiten is not None:
            raise ValueError("scale, options and whiten come from like; omit them.")
        scale, top_m, options = like.scale, like.top_m, like.options
        whiteners, dtype = like.whiteners, like.matrix.dtype
    check_embedding_args(scale, top_m)
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    template = None
    if whiten is not None:
        template = (
            whiten
            if isinstance(whiten, EmpiricalFisherWhitener)
            else EmpiricalFisherWhitener(_whitening_method(whiten))
        )
    return EmbeddingSetup(
        scale=scale,
        top_m=top_m,
        options=resolve_fisher_options(options),
        dtype=np.float32 if dtype is None else dtype,
        whiteners=whiteners,
        template=template,
    )


def _whitening_method(method: Any) -> WhiteningMethod:
    """Return ``method`` if it names a whitening method."""
    if method not in ("full", "diag", "pca"):
        raise ValueError(f"Unknown whitening method {method!r}.")
    return method  # type: ignore[no-any-return]


def encoded_chunks(
    model: Any, data: Sequence[Any], chunk_size: Optional[int]
) -> Iterator[Any]:
    """Yield the mixture encoding of consecutive chunks of ``data``."""
    count = len(data)
    if count == 0:
        return
    step = count if chunk_size is None else chunk_size
    encoder = model.dist_to_encoder()
    for start in range(0, count, step):
        yield encoder.seq_encode(data[start : start + step])


def posterior_pass(
    model: Any, data: Sequence[Any], chunk_size: Optional[int]
) -> List[np.ndarray]:
    """Return the posteriors of each chunk of ``data``."""
    return [
        np.asarray(model.seq_posterior(encoded), dtype=np.float64)
        for encoded in encoded_chunks(model, data, chunk_size)
    ]


def whitening_pass(
    model: Any,
    data: Sequence[Any],
    chunk_size: Optional[int],
    options: FisherScoreOptions,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return packed score moments of ``data`` and the posteriors of each chunk.

    The vector is all zeros for empty ``data``, so it can be added to the
    vectors of other processes.
    """
    dims = embedding_layout(model, options, None).dims
    moments: Optional[List[Optional[ScoreMoments]]] = None
    posteriors = []
    for encoded in encoded_chunks(model, data, chunk_size):
        chunk = np.asarray(model.seq_posterior(encoded), dtype=np.float64)
        posteriors.append(chunk)
        moments = add_score_moments(
            model, mixture_component_data(encoded), chunk, dims, options, moments
        )
    return pack_moments(moments, dims), posteriors


def pack_moments(
    moments: Optional[Sequence[Optional[ScoreMoments]]],
    dims: Sequence[Optional[int]],
) -> np.ndarray:
    """Pack per-component ``(total, first, second)`` moments into one vector."""
    parts = []
    for index, dim in enumerate(dims):
        if not dim:
            continue
        entry = None if moments is None else moments[index]
        if entry is None:
            parts.append(np.zeros(1 + dim + dim * dim))
        else:
            parts.append(np.concatenate(([entry[0]], entry[1], entry[2].ravel())))
    return np.concatenate(parts) if parts else np.zeros(0)


def unpack_moments(
    vector: np.ndarray, dims: Sequence[Optional[int]]
) -> List[Optional[ScoreMoments]]:
    """Invert :func:`pack_moments`."""
    moments: List[Optional[ScoreMoments]] = []
    offset = 0
    for dim in dims:
        if not dim:
            moments.append(None)
            continue
        total = float(vector[offset])
        first = vector[offset + 1 : offset + 1 + dim]
        second = vector[offset + 1 + dim : offset + 1 + dim + dim * dim]
        moments.append((total, first.copy(), second.reshape(dim, dim).copy()))
        offset += 1 + dim + dim * dim
    if offset != len(vector):
        raise ValueError(
            f"Expected {offset} packed moment values, got {len(vector)}; the "
            "processes may hold different models."
        )
    return moments


def fit_whiteners(
    model: Any, setup: EmbeddingSetup, vector: np.ndarray
) -> List[Optional[EmpiricalFisherWhitener]]:
    """Fit the per-component whiteners from moments summed over all records."""
    assert setup.template is not None
    dims = embedding_layout(model, setup.options, None).dims
    return fit_component_whiteners(setup.template, unpack_moments(vector, dims))


def scale_dims(
    model: Any, options: FisherScoreOptions, whiteners: Whiteners
) -> List[Optional[int]]:
    """Return the score block widths that enter ``p_eff``."""
    return embedding_layout(model, options, whiteners).dims


def rows_pass(
    model: Any,
    data: Sequence[Any],
    chunk_size: Optional[int],
    *,
    scale: Optional[float],
    setup: EmbeddingSetup,
    whiteners: Whiteners,
) -> Tuple[csr_matrix, Optional[float]]:
    """Return the rows of ``data`` and the scale used.

    With ``scale=None`` there must be at most one chunk, whose own posteriors
    give the default scale. Empty ``data`` gives a matrix with no rows and
    scale None.
    """
    pieces = [
        model.seq_fisher_embedding(
            encoded,
            scale=scale,
            top_m=setup.top_m,
            options=setup.options,
            dtype=setup.dtype,
            whiteners=whiteners,
        )
        for encoded in encoded_chunks(model, data, chunk_size)
    ]
    if not pieces:
        width = embedding_layout(model, setup.options, whiteners).width
        return csr_matrix((0, width), dtype=setup.dtype), scale
    if scale is None and len(pieces) > 1:
        raise ValueError("rows_pass() needs a scale for more than one chunk.")
    if len(pieces) == 1:
        return pieces[0].matrix, pieces[0].scale
    return sparse_vstack([piece.matrix for piece in pieces], format="csr"), scale


def make_embedding(
    model: Any,
    matrix: Optional[csr_matrix],
    scale: float,
    setup: EmbeddingSetup,
    whiteners: Whiteners,
    like: Optional[FisherEmbedding],
) -> FisherEmbedding:
    """Wrap rows in a :class:`FisherEmbedding` with the model's layout.

    A ``matrix`` of None gives an embedding with no rows, which still carries
    the scale, slices and whiteners for use as ``like``.

    Raises:
        ValueError: If ``like`` was computed with a different layout.
    """
    layout = embedding_layout(model, setup.options, whiteners)
    if like is not None and (
        layout.width != like.matrix.shape[1]
        or layout.component_slices != like.component_slices
    ):
        raise ValueError("like was computed with a different model layout.")
    if matrix is None:
        matrix = csr_matrix((0, layout.width), dtype=setup.dtype)
    return FisherEmbedding(
        matrix=matrix,
        scale=float(scale),
        top_m=setup.top_m,
        posterior_slice=slice(0, len(model.components)),
        component_slices=layout.component_slices,
        field_slices=layout.field_slices,
        options=setup.options,
        whiteners=whiteners,
    )


@contextlib.contextmanager
def deduplicated_warnings(stacklevel: int = 4) -> Iterator[None]:
    """Collect warnings raised inside the block and re-emit each one once.

    Chunked passes would otherwise repeat the same warning for every chunk.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for category, message in dict.fromkeys(
        (item.category, str(item.message)) for item in caught
    ):
        warnings.warn(message, category, stacklevel=stacklevel)
