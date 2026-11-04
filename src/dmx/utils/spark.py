"""Optional PySpark support for the Spark-backed code paths.

PySpark is not a core dependency of dmx-learn; it is installed with the
``spark`` (or ``distributed``/``all``) extra. Core modules use :func:`is_rdd`
to accept RDDs without importing PySpark, and Spark-only modules call
:func:`require_pyspark` at import time so a missing installation fails with an
actionable message instead of a bare ``ModuleNotFoundError``.

:func:`fisher_embedding_rdd` computes Fisher-geometry embeddings of the
records of an RDD; it uses only RDD methods and does not import PySpark.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherEmbedding,
    FisherScoreOptions,
    WhiteningMethod,
)
from dmx.utils.optional import install_hint, require_optional

if TYPE_CHECKING:
    from pyspark import RDD
    from typing_extensions import TypeIs

SPARK_INSTALL_HINT = install_hint("spark")


def require_pyspark(feature: str = "This feature") -> ModuleType:
    """Import PySpark or raise an error explaining how to install it.

    Args:
        feature: Name of the module or feature that needs PySpark, used in the
            error message.

    Returns:
        The imported ``pyspark`` module.

    Raises:
        ImportError: If PySpark is not installed.
    """
    return require_optional("pyspark", "spark", feature, package="PySpark")


def is_rdd(obj: Any) -> TypeIs[RDD[Any]]:
    """Return whether ``obj`` is a PySpark RDD, without importing PySpark.

    An RDD can only exist once PySpark has been imported, so when PySpark is
    absent from ``sys.modules`` the answer is ``False`` and PySpark is never
    loaded.

    Args:
        obj: Object to test.

    Returns:
        ``True`` if ``obj`` is an instance of ``pyspark.RDD``.
    """
    pyspark = sys.modules.get("pyspark")
    rdd_type = getattr(pyspark, "RDD", None)
    return isinstance(rdd_type, type) and isinstance(obj, rdd_type)


# Mirrors fisher_embedding; arguments are keyword-only.
# pylint: disable-next=too-many-arguments,too-many-locals
def fisher_embedding_rdd(
    model: Any,
    rdd: RDD[Any],
    *,
    scale: Optional[float] = None,
    top_m: Optional[int] = 3,
    options: Optional[FisherScoreOptions] = None,
    chunk_size: Optional[int] = None,
    whiten: Optional[Union[WhiteningMethod, EmpiricalFisherWhitener]] = None,
    like: Optional[FisherEmbedding] = None,
    dtype: Any = None,
) -> Tuple[RDD[Tuple[int, csr_matrix]], FisherEmbedding]:
    """Compute the Fisher embedding of the records of an RDD.

    Each partition embeds its records with ``mapPartitions``. With the
    default scale, one job sums the exact ``p_eff`` terms of all partitions
    first; with ``whiten``, one job sums the score moments before that. The
    rows equal those of :func:`dmx.utils.fisher.fisher_embedding` on
    ``rdd.collect()`` bit for bit, except that with ``whiten`` the moments
    are summed in floating point, so rows match only up to summation order.

    Args:
        model: Fitted mixture; it is broadcast to the executors.
        rdd: Records to embed.
        scale: Weight ``c`` of the score blocks. None computes
            ``1 / p_eff`` over all records.
        top_m: Score blocks kept per record. None keeps all.
        options: Score options passed to every component.
        chunk_size: Records per chunk inside a partition. None embeds a
            partition at once.
        whiten: Fit per-component whiteners on all records, as in
            :func:`dmx.utils.fisher.fisher_embedding`.
        like: Reuse a previous embedding's scale, ``top_m``, options,
            whiteners and dtype; no preliminary job is run.
        dtype: Output dtype; None means float32, or the dtype of ``like``.

    Returns:
        A lazy RDD of ``(index, row)`` pairs, where ``index`` is the record's
        position in ``rdd`` and ``row`` a ``1 x width`` CSR matrix, and a
        :class:`~dmx.utils.fisher.FisherEmbedding` with no rows that holds
        the scale, slices and whiteners. Pass it as ``like`` to embed more
        records in the same feature space.

    Raises:
        TypeError: If ``model`` has no ``seq_fisher_embedding``.
        ValueError: If an argument is out of range, or ``scale``,
            ``options`` or ``whiten`` is given with ``like``.
    """
    # Imported here so this module stays light for the core packages.
    # pylint: disable-next=import-outside-toplevel
    from dmx.utils import _fisher_passes as passes

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
    shared = rdd.context.broadcast(model)
    with passes.deduplicated_warnings():
        whiteners = setup.whiteners
        if setup.template is not None:
            vector = rdd.mapPartitions(
                lambda records: [
                    passes.whitening_pass(
                        shared.value, list(records), chunk_size, setup.options
                    )[0]
                ]
            ).treeReduce(np.add)
            whiteners = passes.fit_whiteners(model, setup, vector)
        used_scale = setup.scale
        if used_scale is None:
            dims = passes.scale_dims(model, setup.options, whiteners)
            terms = rdd.mapPartitions(
                lambda records: [
                    passes.scale_terms(
                        passes.posterior_pass(shared.value, list(records), chunk_size),
                        dims,
                    )
                ]
            ).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
            used_scale = passes.scale_from_terms(terms)
        layout = passes.make_embedding(model, None, used_scale, setup, whiteners, like)

    def embed_partition(
        items: Iterable[Tuple[Any, int]],
    ) -> Iterator[Tuple[int, csr_matrix]]:
        pairs = list(items)
        matrix, _ = passes.rows_pass(
            shared.value,
            [record for record, _ in pairs],
            chunk_size,
            scale=used_scale,
            setup=setup,
            whiteners=whiteners,
        )
        for row, (_, index) in enumerate(pairs):
            yield index, matrix[row]

    return rdd.zipWithIndex().mapPartitions(embed_partition), layout
