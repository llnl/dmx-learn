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

import warnings
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "EmpiricalFisherWhitener",
    "FisherScoreArray",
    "FisherScoreOptions",
    "as_score_array",
    "component_score_dims",
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
        on_missing: What a mixture does with a component whose distribution
            tree has no Fisher score. ``"raise"`` fails; ``"posterior_only"``
            keeps only that component's posterior information and warns once
            per call.
        dense_category_limit: Largest number of levels for which categorical
            families return a dense score. Larger supports use a sparse form.
            Must be at least 1.
        whitening_samples: Number of draws from the model used to estimate
            the covariance of scores that have no closed-form Fisher
            information, such as nested mixtures. Must be at least 2.
        whitening_seed: Seed for those draws, so the whitening, and hence the
            score, is a deterministic function of the model. Must be a
            nonnegative integer.

    Raises:
        ValueError: If a value is outside its documented range.
    """

    prob_floor: float = 1.0e-6
    location_only: bool = False
    sequence_scaling: SequenceScaling = "conditional"
    balance_fields: bool = False
    on_missing: OnMissing = "raise"
    dense_category_limit: int = 256
    whitening_samples: int = 4096
    whitening_seed: int = 0

    def __post_init__(self) -> None:
        """Validate option values."""
        for name, minimum in (
            ("dense_category_limit", 1),
            ("whitening_samples", 2),
            ("whitening_seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"{name} must be an integer of at least {minimum}, got {value!r}."
                )
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


def component_score_dims(
    components: Sequence[Any], options: FisherScoreOptions
) -> Tuple[List[Optional[int]], List[str]]:
    """Return each component's score dimension, or None where unsupported.

    With ``options.on_missing == "raise"`` an unsupported component raises
    :class:`NotImplementedError`. With ``"posterior_only"`` its dimension is
    None and the class that lacks a score is reported, so the caller can skip
    the component and warn once.

    Args:
        components: Component distributions of a mixture.
        options: Resolved score options.

    Returns:
        The dimensions, and the names of the classes without a score (empty
        when every component is supported).

    Raises:
        NotImplementedError: If a component has no score and
            ``options.on_missing == "raise"``.
    """
    dims: List[Optional[int]] = []
    unsupported: List[str] = []
    for component in components:
        try:
            dims.append(int(component.fisher_dim(options)))
        except NotImplementedError as err:
            if options.on_missing == "raise":
                raise
            dims.append(None)
            # The message starts with the class that has no score.
            name = str(err).split(" ", 1)[0] or type(component).__name__
            if name not in unsupported:
                unsupported.append(name)
    return dims, unsupported


def warn_unsupported(unsupported: Sequence[str]) -> None:
    """Warn once that components of these classes contribute no score."""
    if unsupported:
        warnings.warn(
            "Fisher scores skip components without a score (on_missing="
            f"'posterior_only'): {', '.join(unsupported)}.",
            UserWarning,
            stacklevel=3,
        )


WhiteningMethod = Literal["full", "diag", "pca"]


@dataclass
class EmpiricalFisherWhitener:
    """Whiten score vectors with a covariance estimated from samples.

    Used for scores whose Fisher information has no closed form, such as
    nested mixtures, and by mixture-level embeddings. ``fit`` estimates the
    (optionally weighted) covariance ``C`` of the samples; ``transform`` maps
    a score ``s`` to ``(s - mean) W`` so the transformed samples have identity
    covariance on the span of ``C``. Fitting is deterministic. A fitted
    whitener is a plain dataclass of arrays and pickles with the embedding
    that uses it.

    Attributes:
        method: ``"full"`` uses the symmetric (ZCA) whitening
            ``C^{-1/2}``, which keeps the dimension and stays as close as
            possible to the original coordinates; ``"diag"`` divides each
            coordinate by its standard deviation; ``"pca"`` projects onto the
            top ``rank`` principal directions and scales them to unit variance.
        rank: Number of directions kept by ``"pca"``; None keeps every
            direction with nonzero variance.
        center: Subtract the sample mean. Use False for scores whose mean is
            exactly zero under the model, so the transform stays linear.
        tol: Directions (or coordinates) with variance at most ``tol`` times
            the largest are treated as null and mapped to zero.
        mean_: Fitted mean, shape ``(d,)`` (zero when ``center`` is False).
        matrix_: Fitted whitening matrix ``W``, shape ``(d, k)``.
    """

    method: WhiteningMethod = "full"
    rank: Optional[int] = None
    center: bool = True
    tol: float = 1.0e-10
    mean_: Optional[np.ndarray] = field(default=None, repr=False)
    matrix_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(
        self, scores: Any, weights: Optional[Any] = None
    ) -> "EmpiricalFisherWhitener":
        """Estimate the whitening transform from sample scores.

        Args:
            scores: Sample scores, shape ``(n, d)`` with ``n >= 2``, dense or
                sparse.
            weights: Optional nonnegative weight per sample, such as
                posterior probabilities.

        Returns:
            This whitener, fitted.

        Raises:
            ValueError: If ``method`` is unknown, there are fewer than two
                samples, or the weights do not match the samples.
        """
        if self.method not in ("full", "diag", "pca"):
            raise ValueError(f"Unknown whitening method {self.method!r}.")
        x = _dense(scores)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError(
                f"Whitening needs at least two samples of shape (n, d), got {x.shape}."
            )
        w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, float)
        if w.shape != (x.shape[0],) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative, one per sample, not all 0.")
        w = w / w.sum()
        mean = w @ x if self.center else np.zeros(x.shape[1])
        centered = x - mean
        covariance = (centered * w[:, None]).T @ centered
        if self.method == "diag":
            variance = np.diag(covariance)
            keep = variance > self.tol * max(float(variance.max(initial=0.0)), 0.0)
            scale = np.zeros(len(variance))
            scale[keep] = 1.0 / np.sqrt(variance[keep])
            matrix = np.diag(scale)
        else:
            values, vectors = np.linalg.eigh(covariance)
            keep = values > self.tol * max(float(values.max(initial=0.0)), 0.0)
            if self.method == "full":
                inv_sqrt = np.zeros(len(values))
                inv_sqrt[keep] = 1.0 / np.sqrt(values[keep])
                matrix = (vectors * inv_sqrt) @ vectors.T
            else:
                order = np.argsort(values)[::-1]
                order = order[keep[order]]
                if self.rank is not None:
                    order = order[: self.rank]
                matrix = vectors[:, order] / np.sqrt(values[order])
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.matrix_ = np.asarray(matrix, dtype=np.float64)
        return self

    def transform(self, scores: Any) -> np.ndarray:
        """Return ``(scores - mean_) @ matrix_``.

        Args:
            scores: Scores of shape ``(n, d)``, dense or sparse.

        Returns:
            Whitened scores of shape ``(n, k)``.

        Raises:
            RuntimeError: If the whitener has not been fitted.
            ValueError: If ``scores`` has the wrong number of columns.
        """
        if self.mean_ is None or self.matrix_ is None:
            raise RuntimeError("EmpiricalFisherWhitener.transform() before fit().")
        x = _dense(scores)
        if x.ndim != 2 or x.shape[1] != self.matrix_.shape[0]:
            raise ValueError(
                f"Expected scores with {self.matrix_.shape[0]} columns, got {x.shape}."
            )
        return np.ascontiguousarray((x - self.mean_) @ self.matrix_)


def _dense(scores: Any) -> np.ndarray:
    """Return scores as a dense float64 array."""
    if hasattr(scores, "toarray"):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float64)
