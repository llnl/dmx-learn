"""MPI computation of Fisher-geometry embeddings.

Every rank embeds the records it holds. The only collectives are one
``allreduce`` of the exact ``p_eff`` terms for the default scale, one
``allreduce`` of the score moments when ``whiten`` is given, and the final
``gather`` of the rows. See :func:`dmx.utils.fisher.fisher_embedding` for
the embedding itself.
"""

from typing import Any, Optional, Sequence

import numpy as np
from scipy.sparse import vstack as sparse_vstack

from dmx.mpi4py.utils import get_runtime_attr
from dmx.utils import _fisher_passes as passes
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherEmbedding,
    FisherScoreOptions,
    WhiteningMethod,
)

__all__ = ["fisher_embedding_mpi"]


# Mirrors fisher_embedding, plus the communicator arguments.
# pylint: disable-next=too-many-arguments,too-many-locals
def fisher_embedding_mpi(
    model: Any,
    local_data: Sequence[Any],
    comm: Optional[Any] = None,
    *,
    scale: Optional[float] = None,
    top_m: Optional[int] = 3,
    options: Optional[FisherScoreOptions] = None,
    chunk_size: Optional[int] = None,
    whiten: Optional[WhiteningMethod | EmpiricalFisherWhitener] = None,
    like: Optional[FisherEmbedding] = None,
    dtype: Any = None,
    gather: bool = True,
    root: int = 0,
) -> Optional[FisherEmbedding]:
    """Compute the Fisher embedding of records spread over MPI ranks.

    Collective over ``comm``: every rank must call it with the same model and
    arguments, each passing its own records (possibly none). The rows equal
    those of :func:`dmx.utils.fisher.fisher_embedding` on all records in rank
    order, bit for bit, because the default scale is computed from exact sums.
    With ``whiten``, the score moments are summed across ranks in floating
    point, so the whiteners, and the rows, match the local result only up to
    summation order.

    Args:
        model: Fitted mixture, identical on every rank (broadcast it first
            if needed).
        local_data: This rank's records; must support ``len`` and slicing.
        comm: Communicator; None means ``MPI.COMM_WORLD``.
        scale: Weight ``c`` of the score blocks. None computes
            ``1 / p_eff`` over the records of all ranks.
        top_m: Score blocks kept per record. None keeps all.
        options: Score options passed to every component.
        chunk_size: Records per chunk on each rank. None embeds a rank's
            records at once.
        whiten: Fit per-component whiteners on the records of all ranks, as
            in :func:`dmx.utils.fisher.fisher_embedding`.
        like: Reuse a previous embedding's scale, ``top_m``, options,
            whiteners and dtype. No collective is needed except the gather.
        dtype: Output dtype; None means float32, or the dtype of ``like``.
        gather: Gather all rows on ``root``. With False every rank returns
            the embedding of its own records.
        root: Rank that receives the gathered rows.

    Returns:
        With ``gather``, the embedding of all records on ``root`` and None on
        other ranks; otherwise each rank's own embedding. All embeddings
        carry the same scale and whiteners, so any of them can be passed as
        ``like``.

    Raises:
        TypeError: If ``model`` has no ``seq_fisher_embedding``.
        ValueError: If an argument is out of range, or ``scale``,
            ``options`` or ``whiten`` is given with ``like``.
        NotImplementedError: If a component has no score and
            ``options.on_missing == "raise"``.
    """
    mpi = get_runtime_attr("mpi4py", "MPI")
    if comm is None:
        comm = mpi.COMM_WORLD
    setup = passes.resolve_setup(
        model,
        scale=scale,
        top_m=top_m,
        options=options,
        chunk_size=chunk_size,
        whiten=whiten,
        like=like,
        dtype=dtype,
    )
    with passes.deduplicated_warnings():
        whiteners = setup.whiteners
        posteriors = None
        if setup.template is not None:
            vector, posteriors = passes.whitening_pass(
                model, local_data, chunk_size, setup.options
            )
            vector = comm.allreduce(vector, op=mpi.SUM)
            whiteners = passes.fit_whiteners(model, setup, vector)
        used_scale = setup.scale
        if used_scale is None:
            if posteriors is None:
                posteriors = passes.posterior_pass(model, local_data, chunk_size)
            dims = passes.scale_dims(model, setup.options, whiteners)
            # Python integers in an object array add elementwise and exactly.
            terms = np.empty(2, dtype=object)
            terms[:] = passes.scale_terms(posteriors, dims)
            total, count = comm.allreduce(terms, op=mpi.SUM)
            used_scale = passes.scale_from_terms((int(total), int(count)))
        matrix, _ = passes.rows_pass(
            model,
            local_data,
            chunk_size,
            scale=used_scale,
            setup=setup,
            whiteners=whiteners,
        )
        local = passes.make_embedding(model, matrix, used_scale, setup, whiteners, like)
    if not gather:
        return local
    pieces = comm.gather(local.matrix, root=root)
    if comm.Get_rank() != root:
        return None
    local.matrix = sparse_vstack(pieces, format="csr", dtype=local.matrix.dtype)
    return local
