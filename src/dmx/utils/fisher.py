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
  float64 array for an encoded sequence ``x``.
- ``fisher_dim(options=None)`` returns the score dimension.
- ``fisher_field_slices(options=None)`` maps readable field names to column
  slices of the score, for attributing distances to fields.

Combinators recurse into their children with the same
:class:`FisherScoreOptions`. Parameters shared through estimator ``keys`` do
not change the definition: each occurrence is scored with the parameters it
uses.

Closed-form score kernels for individual families are added to this module as
pure functions of the observations and parameters, so :mod:`dmx.stats` and
:mod:`dmx.bstats` share one implementation. Kernels take NumPy arrays and
scalar parameters, return the result of :func:`as_score_array`, and never
import distribution classes.

See ``docs/utils/fisher_embedding.rst`` for the embedding built from these
scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from dmx.utils.special import digamma, trigamma

__all__ = [
    "FisherScoreOptions",
    "as_score_array",
    "exponential_score",
    "fisher_score_unsupported",
    "gamma_score",
    "gaussian_score",
    "has_fisher_score",
    "resolve_fisher_options",
    "trigamma_minus_reciprocal",
]

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

    Score kernels use this to normalize their output. A one-dimensional input
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


# --- Score kernels -----------------------------------------------------------

_SQRT_HALF = float(np.sqrt(0.5))

# Shape above which trigamma(k) - 1/k is taken from its asymptotic series.
_TRIGAMMA_SERIES_SHAPE = 50.0


def gaussian_score(x: Any, mu: float, sigma2: float) -> np.ndarray:
    """Return the whitened score of a univariate Gaussian.

    With ``u = (x - mu) / sigma`` the score is ``[u, (u^2 - 1) / sqrt(2)]``:
    the location and scale directions, each with unit variance and
    uncorrelated under the Gaussian.

    Args:
        x: Observations, shape ``(N,)``.
        mu: Mean.
        sigma2: Positive variance.

    Returns:
        Array of shape ``(N, 2)``.
    """
    u = (np.asarray(x, dtype=np.float64) - mu) / np.sqrt(sigma2)
    return as_score_array(np.column_stack((u, (u * u - 1.0) * _SQRT_HALF)), 2)


def exponential_score(x: Any, beta: float) -> np.ndarray:
    """Return the whitened score of an exponential distribution with mean ``beta``.

    The score is ``x / beta - 1``.

    Args:
        x: Nonnegative observations, shape ``(N,)``.
        beta: Positive mean (scale).

    Returns:
        Array of shape ``(N, 1)``.
    """
    return as_score_array(np.asarray(x, dtype=np.float64) / beta - 1.0, 1)


def trigamma_minus_reciprocal(k: float) -> float:
    """Return ``trigamma(k) - 1/k`` accurately for large ``k``.

    The difference is positive for every ``k > 0`` and behaves like
    ``1 / (2 k^2)`` for large ``k``, so subtracting the two terms directly
    loses about ``log10(2 k)`` significant digits (relative error near
    ``2e-10`` at ``k = 1e6``). For ``k >= 50`` the asymptotic series is used
    instead, which stays accurate to rounding error.

    Args:
        k: Positive gamma shape.

    Returns:
        ``trigamma(k) - 1/k``.
    """
    if k < _TRIGAMMA_SERIES_SHAPE:
        return float(trigamma(k)) - 1.0 / k
    r = 1.0 / k
    r2 = r * r
    # 1/(2k^2) + 1/(6k^3) - 1/(30k^5) + 1/(42k^7) - 1/(30k^9), in Horner form.
    series = 1.0 / 6.0 + r2 * (-1.0 / 30.0 + r2 * (1.0 / 42.0 - r2 / 30.0))
    return float(r2 * (0.5 + r * series))


def gamma_score(x: Any, log_x: Any, k: float, theta: float) -> np.ndarray:
    """Return the whitened score of a gamma distribution with shape and scale.

    The sufficient statistics ``[x, log x]`` have mean
    ``[k theta, digamma(k) + log theta]`` and covariance
    ``[[k theta^2, theta], [theta, trigamma(k)]]``. The score is
    ``L^{-1} (T - mean)`` for the lower Cholesky factor ``L`` of that
    covariance:

    - ``s1 = (x - k theta) / (sqrt(k) theta)``
    - ``s2 = (log x - digamma(k) - log theta - s1 / sqrt(k))
      / sqrt(trigamma(k) - 1/k)``

    Args:
        x: Positive observations, shape ``(N,)``.
        log_x: ``log(x)``, shape ``(N,)``, as stored by the gamma encoder.
        k: Positive shape.
        theta: Positive scale.

    Returns:
        Array of shape ``(N, 2)``.
    """
    sqrt_k = np.sqrt(k)
    s1 = (np.asarray(x, dtype=np.float64) - k * theta) / (sqrt_k * theta)
    centered_log = np.asarray(log_x, dtype=np.float64) - (digamma(k) + np.log(theta))
    s2 = (centered_log - s1 / sqrt_k) / np.sqrt(trigamma_minus_reciprocal(k))
    return as_score_array(np.column_stack((s1, s2)), 2)
