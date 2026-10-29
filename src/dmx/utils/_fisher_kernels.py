"""Closed-form Fisher score kernels shared by dmx.stats and dmx.bstats.

This module is private: callers use ``seq_fisher_score`` on a distribution,
and each distribution's docstring states its formula. The kernels live here
rather than in the distribution modules because :mod:`dmx.stats` and
:mod:`dmx.bstats` implement the same families and never import each other;
one implementation keeps their scores identical.

Kernels are pure functions of observations (NumPy arrays) and scalar
parameters. They return the result of :func:`dmx.utils.fisher.as_score_array`
and never import distribution classes. Add a family's kernel here when its
issue implements the score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csr_matrix

from dmx.utils.fisher import FisherScoreArray, as_score_array
from dmx.utils.special import digamma, trigamma

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


def clamp_probability(p: float, floor: float) -> float:
    """Return ``p`` clipped to ``[floor, 1 - floor]``.

    Count and binary scores divide by ``sqrt(p (1 - p))`` or similar, which
    is zero at the ends of the parameter range; clipping keeps them finite.

    Args:
        p: Probability parameter.
        floor: Lower bound, typically ``FisherScoreOptions.prob_floor``.

    Returns:
        The clipped probability.
    """
    return min(max(float(p), floor), 1.0 - floor)


def bernoulli_score(x: Any, p: float, floor: float = 1.0e-6) -> np.ndarray:
    """Return the whitened Bernoulli score ``(x - p) / sqrt(p (1 - p))``.

    Args:
        x: Observations in ``{0, 1}``, shape ``(N,)``.
        p: Success probability, clipped with :func:`clamp_probability`.
        floor: Probability floor used for clipping.

    Returns:
        Array of shape ``(N, 1)``.
    """
    p = clamp_probability(p, floor)
    rv = (np.asarray(x, dtype=np.float64) - p) / np.sqrt(p * (1.0 - p))
    return as_score_array(rv, 1)


def poisson_score(x: Any, lam: float, floor: float = 1.0e-6) -> np.ndarray:
    """Return the whitened Poisson score ``(x - lam) / sqrt(lam)``.

    Args:
        x: Nonnegative counts, shape ``(N,)``.
        lam: Rate; values below ``floor`` are raised to ``floor``.
        floor: Smallest rate used.

    Returns:
        Array of shape ``(N, 1)``.
    """
    lam = max(float(lam), floor)
    return as_score_array((np.asarray(x, dtype=np.float64) - lam) / np.sqrt(lam), 1)


def geometric_score(x: Any, p: float, floor: float = 1.0e-6) -> np.ndarray:
    """Return the whitened score of a geometric distribution on ``1, 2, ...``.

    The count ``x`` has mean ``1 / p`` and variance ``(1 - p) / p^2``, so the
    score is ``(p x - 1) / sqrt(1 - p)``.

    Args:
        x: Counts of trials up to and including the first success, shape
            ``(N,)``.
        p: Success probability, clipped with :func:`clamp_probability`.
        floor: Probability floor used for clipping.

    Returns:
        Array of shape ``(N, 1)``.
    """
    p = clamp_probability(p, floor)
    rv = (p * np.asarray(x, dtype=np.float64) - 1.0) / np.sqrt(1.0 - p)
    return as_score_array(rv, 1)


def binomial_score(y: Any, n: int, p: float, floor: float = 1.0e-6) -> np.ndarray:
    """Return the whitened binomial score ``(y - n p) / sqrt(n p (1 - p))``.

    With ``n = 0`` every observation is the same and carries no information,
    so the score is zero.

    Args:
        y: Success counts in ``0..n``, shape ``(N,)``.
        n: Number of trials.
        p: Success probability, clipped with :func:`clamp_probability`.
        floor: Probability floor used for clipping.

    Returns:
        Array of shape ``(N, 1)``.
    """
    y = np.asarray(y, dtype=np.float64)
    if n <= 0:
        return as_score_array(np.zeros_like(y), 1)
    p = clamp_probability(p, floor)
    return as_score_array((y - n * p) / np.sqrt(n * p * (1.0 - p)), 1)


def categorical_probabilities(weights: Any, floor: float) -> np.ndarray:
    """Return level probabilities floored at ``floor`` and normalized.

    Flooring keeps ``1 / sqrt(q)`` finite for rare or zero-weight levels;
    normalizing afterwards keeps ``sqrt(q)`` a unit vector, so the dense score
    still has a projector covariance.

    Args:
        weights: Nonnegative level weights, shape ``(C,)``.
        floor: Smallest probability before normalizing.

    Returns:
        Probabilities of shape ``(C,)`` that sum to one.
    """
    q = np.maximum(np.asarray(weights, dtype=np.float64), floor)
    return np.asarray(q / q.sum())


def categorical_score(
    index: Any, probs: Any, floor: float, dense_limit: int
) -> FisherScoreArray:
    """Return the whitened score of categorical observations.

    With level probabilities ``q`` (from :func:`categorical_probabilities`),
    the dense score is ``e_x / sqrt(q_x) - sqrt(q)``. Its covariance is the
    projector ``I - sqrt(q) sqrt(q)^T`` and two different levels ``a`` and
    ``b`` are ``1/q_a + 1/q_b`` apart in squared distance.

    Above ``dense_limit`` levels the offset-free form ``e_x / sqrt(q_x)`` is
    returned as a CSR matrix with one nonzero per row. It differs from the
    dense form only by the constant ``sqrt(q)``, so distances between rows of
    the same distribution are unchanged.

    An observation with ``index == -1`` has zero probability under the
    distribution (no level matches it). Its row is the "no level" row:
    ``-sqrt(q)`` in the dense form, empty in the sparse form. Such records
    get zero posterior weight for this component, so the row only needs to
    be finite.

    Args:
        index: Level index of each observation in ``0..C-1``, or ``-1``,
            shape ``(N,)``.
        probs: Level weights, shape ``(C,)``.
        floor: Probability floor.
        dense_limit: Largest number of levels returned densely.

    Returns:
        Float64 array, or CSR matrix above ``dense_limit`` levels, of shape
        ``(N, C)``.
    """
    q = categorical_probabilities(probs, floor)
    index = np.asarray(index, dtype=np.intp)
    n, num_levels = len(index), len(q)
    rows = np.flatnonzero(index >= 0)
    cols = index[rows]
    values = 1.0 / np.sqrt(q[cols])
    if num_levels > dense_limit:
        return csr_matrix((values, (rows, cols)), shape=(n, num_levels))
    scores = np.tile(-np.sqrt(q), (n, 1))
    scores[rows, cols] += values
    return as_score_array(scores, num_levels)


def _gaussian_scale_block(u: np.ndarray) -> np.ndarray:
    """Return the whitened covariance score of standardized Gaussian rows ``u``.

    Columns run over the upper triangle ``i <= j`` in row-major order:
    ``(u_i^2 - 1) / sqrt(2)`` on the diagonal and ``u_i u_j`` off it. For
    ``u ~ N(0, I)`` these have unit variance and are uncorrelated with each
    other and with ``u``.
    """
    rows, cols = np.triu_indices(u.shape[1])
    block = u[:, rows] * u[:, cols]
    diagonal = rows == cols
    block[:, diagonal] = (block[:, diagonal] - 1.0) * _SQRT_HALF
    return np.asarray(block)


def mvn_score(
    x: Any, mu: Any, chol_lower: Any, location_only: bool = False
) -> np.ndarray:
    """Return the whitened score of a multivariate Gaussian.

    With ``u = L^{-1} (x - mu)`` for the lower Cholesky factor ``L`` of the
    covariance, the score is ``u`` followed, unless ``location_only``, by the
    scale block from :func:`_gaussian_scale_block`.

    Args:
        x: Observations, shape ``(N, d)``.
        mu: Mean, shape ``(d,)``.
        chol_lower: Lower Cholesky factor of the covariance, shape ``(d, d)``.
        location_only: Return only the location block.

    Returns:
        Array of shape ``(N, d)`` with ``location_only``, else
        ``(N, d + d (d + 1) / 2)``.
    """
    centered = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    u = solve_triangular(chol_lower, centered.T, lower=True).T
    dim = u.shape[1]
    if location_only:
        return as_score_array(u, dim)
    return as_score_array(
        np.hstack((u, _gaussian_scale_block(u))), dim + dim * (dim + 1) // 2
    )


def diagonal_gaussian_score(
    x: Any, mu: Any, variances: Any, location_only: bool = False
) -> np.ndarray:
    """Return the whitened score of a Gaussian with diagonal covariance.

    With ``u = (x - mu) / sigma`` per dimension, the score is ``u`` followed,
    unless ``location_only``, by ``(u^2 - 1) / sqrt(2)``: the univariate
    Gaussian score of each dimension, grouped as all location columns, then
    all scale columns.

    Args:
        x: Observations, shape ``(N, d)``.
        mu: Means, shape ``(d,)``.
        variances: Positive variances, shape ``(d,)``.
        location_only: Return only the location columns.

    Returns:
        Array of shape ``(N, d)`` with ``location_only``, else ``(N, 2 d)``.
    """
    u = (np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / (
        np.sqrt(np.asarray(variances, dtype=np.float64))
    )
    dim = u.shape[1]
    if location_only:
        return as_score_array(u, dim)
    return as_score_array(np.hstack((u, (u * u - 1.0) * _SQRT_HALF)), 2 * dim)


def dirichlet_score(log_x: Any, alpha: Any) -> np.ndarray:
    """Return the whitened score of a Dirichlet distribution.

    The sufficient statistics ``log x`` have mean
    ``digamma(alpha) - digamma(alpha_0)`` and covariance
    ``diag(trigamma(alpha)) - trigamma(alpha_0)``, with
    ``alpha_0 = sum(alpha)``. The score is ``L^{-1} (log x - mean)`` for the
    lower Cholesky factor ``L`` of that covariance.

    Args:
        log_x: ``log`` of the simplex observations, shape ``(N, d)``.
        alpha: Positive concentration parameters, shape ``(d,)``.

    Returns:
        Array of shape ``(N, d)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha_0 = float(alpha.sum())
    mean = digamma(alpha) - digamma(alpha_0)
    covariance = np.diag(trigamma(alpha)) - trigamma(alpha_0)
    chol_lower = np.linalg.cholesky(covariance)
    centered = np.asarray(log_x, dtype=np.float64) - mean
    u = solve_triangular(chol_lower, centered.T, lower=True).T
    return as_score_array(u, len(alpha))
