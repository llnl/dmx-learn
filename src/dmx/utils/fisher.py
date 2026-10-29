"""Whitened Fisher scores for Fisher-geometry embeddings.

A distribution that supports Fisher-geometry embeddings returns, for each
observation ``x``, its **whitened score**::

    s(x) = I^{-1/2} (T(x) - E[T]),

where ``T`` are the distribution's sufficient statistics and ``I = Cov[T]``
is its Fisher information. Under the distribution, ``E[s] = 0`` and
``Cov[s]`` is the identity on the score's support (a projector for
over-complete encodings such as a categorical one-hot). Any orthonormal
basis is allowed, because downstream embeddings only use Euclidean distances
between scores.

Distributions expose the score through three methods on the
``ProbabilityDistribution`` base classes of :mod:`dmx.stats` and
:mod:`dmx.bstats`:

- ``seq_fisher_score(x, options=None)`` returns an ``(N, fisher_dim)``
  float64 array for an encoded sequence ``x``, or a float64 CSR matrix for
  high-cardinality encodings (see :data:`FisherScoreArray`).
- ``fisher_dim(options=None)`` returns the score dimension.
- ``fisher_field_slices(options=None)`` maps readable field names to column
  slices of the score, for attributing distances to fields.

Combinators recurse into their children with the same
:class:`FisherScoreOptions`. Parameters shared through estimator ``keys`` do
not change the definition: each occurrence is scored with the parameters it
uses.

Closed-form formulas for individual families live in the private module
``dmx.utils._fisher_kernels``, shared by :mod:`dmx.stats` and :mod:`dmx.bstats`
so both packages return identical scores. They are not public API; each
distribution's ``seq_fisher_score`` docstring states its formula.

See ``docs/utils/fisher_embedding.rst`` for the embedding built from these
scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "FisherScoreArray",
    "FisherScoreOptions",
    "as_score_array",
    "fisher_score_unsupported",
    "has_fisher_score",
    "resolve_fisher_options",
]

FisherScoreArray = Union[np.ndarray, csr_matrix]
"""Return type of ``seq_fisher_score``: a dense array or a sparse CSR matrix.

Families with many levels, such as categoricals above
``FisherScoreOptions.dense_category_limit``, return CSR so each row stores only
its nonzero entries. Either form has shape ``(N, fisher_dim)`` and dtype
float64.
"""

SequenceScaling = Literal["conditional", "marginal"]
OnMissing = Literal["raise", "posterior_only"]

_SEQUENCE_SCALINGS = ("conditional", "marginal")
_ON_MISSING = ("raise", "posterior_only")


@dataclass(frozen=True)
class FisherScoreOptions:
    """Options shared by every Fisher score in a distribution tree.

    Combinators pass the same instance to their children, so one object
    configures a whole model. Options a family does not use are ignored.

    Attributes:
        prob_floor: Lower bound applied to probabilities before dividing by
            them, for example for rare or unseen categorical levels. Must lie
            in ``(0, 1)``.
        location_only: For multivariate Gaussian families, return only the
            location block of the score and drop the scale block.
        sequence_scaling: How variable-length families scale a summed element
            score of length ``n``. ``"conditional"`` divides by
            ``sqrt(n)``, so the score is whitened given the observed length.
            ``"marginal"`` divides by ``sqrt(E[n])`` under the length
            distribution, so longer records end up farther apart.
        balance_fields: Divide each composite field's block by the square
            root of its dimension, so fields with many parameters do not
            dominate distances. Off by default, which keeps the pure Fisher
            weighting.
        on_missing: What a mixture-level embedding does with a component
            whose distribution tree has no Fisher score. ``"raise"`` fails;
            ``"posterior_only"`` keeps only that component's posterior block
            and warns.
        dense_category_limit: Largest number of levels for which categorical
            families return a dense score. Larger supports use a sparse form.
            Must be at least 1.

    Raises:
        ValueError: If a value is outside its documented range.
    """

    prob_floor: float = 1.0e-6
    location_only: bool = False
    sequence_scaling: SequenceScaling = "conditional"
    balance_fields: bool = False
    on_missing: OnMissing = "raise"
    dense_category_limit: int = 256

    def __post_init__(self) -> None:
        """Validate option values."""
        if not 0.0 < self.prob_floor < 1.0:
            raise ValueError(f"prob_floor must lie in (0, 1), got {self.prob_floor!r}.")
        if self.sequence_scaling not in _SEQUENCE_SCALINGS:
            raise ValueError(
                f"sequence_scaling must be one of {_SEQUENCE_SCALINGS}, "
                f"got {self.sequence_scaling!r}."
            )
        if self.on_missing not in _ON_MISSING:
            raise ValueError(
                f"on_missing must be one of {_ON_MISSING}, " f"got {self.on_missing!r}."
            )
        if (
            isinstance(self.dense_category_limit, bool)
            or not isinstance(self.dense_category_limit, int)
            or self.dense_category_limit < 1
        ):
            raise ValueError(
                "dense_category_limit must be an integer of at least 1, "
                f"got {self.dense_category_limit!r}."
            )


def resolve_fisher_options(
    options: Optional[FisherScoreOptions],
) -> FisherScoreOptions:
    """Return ``options``, or the default options when it is None."""
    return FisherScoreOptions() if options is None else options


def fisher_score_unsupported(dist: Any, method: str) -> NotImplementedError:
    """Return the error raised by a distribution without a Fisher score.

    The base-class methods raise the returned exception, so every unsupported
    family reports the same message naming its class.

    Args:
        dist: Distribution whose method was called.
        method: Name of the method, for example ``"seq_fisher_score"``.

    Returns:
        Exception to raise.
    """
    return NotImplementedError(
        f"{type(dist).__name__} does not implement {method}(); "
        "see docs/utils/fisher_embedding.rst for supported distributions."
    )


def has_fisher_score(dist: Any, options: Optional[FisherScoreOptions] = None) -> bool:
    """Return True if a whole distribution tree supports Fisher scores.

    Support is reported by ``fisher_dim``: a distribution without a score
    raises :class:`NotImplementedError`, and combinators call ``fisher_dim``
    on their children, so one call checks the whole tree. Works for
    :mod:`dmx.stats` and :mod:`dmx.bstats` distributions alike.

    Args:
        dist: Distribution to check. Objects without ``fisher_dim`` are
            reported as unsupported.
        options: Options the scores would be computed with, since some
            options change the score dimension.

    Returns:
        True if ``dist.fisher_dim(options)`` succeeds.
    """
    fisher_dim = getattr(dist, "fisher_dim", None)
    if fisher_dim is None:
        return False
    try:
        fisher_dim(options)
    except NotImplementedError:
        return False
    return True


def as_score_array(values: Any, dim: int) -> np.ndarray:
    """Return score values as a C-contiguous ``(N, dim)`` float64 array.

    Score implementations use this to normalize their output. A one-dimensional input
    is treated as one score column, which is only valid when ``dim == 1``.

    Args:
        values: Score values with shape ``(N,)`` or ``(N, dim)``.
        dim: Expected score dimension.

    Returns:
        Array of shape ``(N, dim)`` with dtype float64.

    Raises:
        ValueError: If ``values`` cannot be read as ``(N, dim)``.
    """
    rv = np.asarray(values, dtype=np.float64)
    if rv.ndim == 1 and dim == 1:
        rv = rv[:, None]
    if rv.ndim != 2 or rv.shape[1] != dim:
        raise ValueError(f"Expected Fisher scores of shape (N, {dim}), got {rv.shape}.")
    return np.ascontiguousarray(rv)
