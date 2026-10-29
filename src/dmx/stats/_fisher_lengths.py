"""Private helpers for Fisher scores of variable-length dmx.stats families.

Kept apart from :mod:`dmx.utils._fisher_kernels` because they inspect
distribution classes, which the shared kernels never import.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from dmx.stats.binomial import BinomialDistribution
from dmx.stats.categorical import CategoricalDistribution
from dmx.stats.geometric import GeometricDistribution
from dmx.stats.intrange import IntegerCategoricalDistribution
from dmx.stats.null_dist import NullDistribution
from dmx.stats.poisson import PoissonDistribution
from dmx.utils._fisher_kernels import (
    hstack_scores,
    length_weights,
    scale_rows,
    sum_rows,
)
from dmx.utils.fisher import FisherScoreArray, FisherScoreOptions


def length_mean(len_dist: Any) -> float:
    """Return the mean of a length distribution, for marginal scaling.

    Supported: Poisson, Geometric (support ``1, 2, ...``), Binomial
    (shifted by ``min_val``), IntegerCategorical, and Categorical over
    numeric lengths (normalized over ``pmap``; ``default_value`` mass on
    unlisted lengths is ignored).

    Args:
        len_dist: Length distribution of a sequence or multinomial.

    Returns:
        ``E[n]``.

    Raises:
        ValueError: If the mean of ``len_dist`` is not known here.
    """
    if isinstance(len_dist, PoissonDistribution):
        return float(len_dist.lam)
    if isinstance(len_dist, GeometricDistribution):
        return 1.0 / float(len_dist.p)
    if isinstance(len_dist, BinomialDistribution):
        shift = len_dist.min_val if len_dist.min_val is not None else 0
        return float(len_dist.n * len_dist.p + shift)
    if isinstance(len_dist, IntegerCategoricalDistribution):
        p = np.asarray(len_dist.p_vec, dtype=np.float64)
        values = len_dist.min_val + np.arange(len(p))
        return float(np.dot(values, p) / p.sum())
    if isinstance(len_dist, CategoricalDistribution):
        lengths = np.asarray(list(len_dist.pmap), dtype=np.float64)
        p = np.asarray(list(len_dist.pmap.values()), dtype=np.float64)
        return float(np.dot(lengths, p) / p.sum())
    raise ValueError(
        "sequence_scaling='marginal' needs the mean length, which is not known "
        f"for {type(len_dist).__name__}; use sequence_scaling='conditional'."
    )


def length_score_dim(len_dist: Any, options: Optional[FisherScoreOptions]) -> int:
    """Return the length block's dimension: zero for a null length distribution."""
    if len_dist is None or isinstance(len_dist, NullDistribution):
        return 0
    return int(len_dist.fisher_dim(options))


def variable_length_score(
    element_scores: FisherScoreArray,
    owner: Any,
    lengths: Any,
    len_dist: Any,
    len_encoded: Any,
    options: FisherScoreOptions,
    weights: Optional[Any] = None,
) -> FisherScoreArray:
    """Return ``[sum of element scores / scale(n), length score]`` per record.

    Element rows are summed into their records (weighted by ``weights``,
    such as counts), each record is divided by ``sqrt(n)`` or ``sqrt(E[n])``
    according to ``options.sequence_scaling``, and the length distribution's
    own score is appended unless it is null.

    Args:
        element_scores: Scores of all elements, shape ``(M, d)``.
        owner: Record index of each element, shape ``(M,)``.
        lengths: Length or total count ``n`` of each record, shape ``(N,)``.
        len_dist: Length distribution.
        len_encoded: Encoded lengths for ``len_dist``.
        options: Resolved score options.
        weights: Optional weight of each element.

    Returns:
        Scores of shape ``(N, d + length_score_dim(len_dist))``.
    """
    n = len(lengths)
    mean = None
    if options.sequence_scaling == "marginal":
        mean = length_mean(len_dist)
    summed = sum_rows(element_scores, owner, n, weights)
    blocks = [
        scale_rows(summed, length_weights(lengths, options.sequence_scaling, mean))
    ]
    if length_score_dim(len_dist, options) > 0:
        blocks.append(len_dist.seq_fisher_score(len_encoded, options))
    return hstack_scores(blocks, n)


def part_slices(dist: Any, parts: Sequence[Tuple[str, int]]) -> Dict[str, slice]:
    """Return named column slices for a score made of consecutive parts.

    Parts with no columns are omitted. With a single remaining part the
    distribution keeps the default single-field slices.
    """
    nonempty = [(name, dim) for name, dim in parts if dim > 0]
    if len(nonempty) <= 1:
        name = getattr(dist, "name", None) or type(dist).__name__
        return {name: slice(0, sum(dim for _, dim in parts))}
    slices: Dict[str, slice] = {}
    offset = 0
    for name, dim in nonempty:
        slices[name] = slice(offset, offset + dim)
        offset += dim
    return slices
