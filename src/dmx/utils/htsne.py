"""Embed heterogeneous observations with mixture-derived t-SNE affinities."""

from typing import Any, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.random import RandomState

from dmx.utils.automatic import MixtureModel, prepare_mixture_model

T = TypeVar("T")


def adj_perplexity(x: np.ndarray, ss: float) -> Tuple[float, np.ndarray]:
    """Convert a distance vector to probabilities at a given scale.

    Args:
        x: One-dimensional distance array of shape ``(n_neighbors,)``.
        ss: Positive distance scale. Its reciprocal acts as inverse
            temperature.

    Returns:
        Natural-log entropy and normalized probabilities with the same shape
        as ``x``.
    """
    s = 1 / ss
    P = -x * s
    M = P.max()
    P = np.exp(P - M)
    sumP = P.sum()
    H = np.log(sumP) + M + s * np.sum(x * P) / sumP
    P = P / sumP
    return H, P


def vec_perplexity(x: np.ndarray, s: float) -> float:
    """Compute entropy for a distance vector at a given scale.

    Args:
        x: One-dimensional distance array.
        s: Scale passed to :func:`adj_perplexity`.

    Returns:
        Natural-log entropy of the normalized probabilities.
    """
    H, _ = adj_perplexity(x, s)
    return H


def row_perplexity_solve(
    x: np.ndarray, a: float, s0: float, s1: float, d: int = 10
) -> float:
    """Bisect a scale interval to approach a target entropy.

    At most ``d`` recursive bisections are performed. If the target is outside
    the endpoint entropies, the corresponding endpoint is returned.

    Args:
        x: One-dimensional distance array.
        a: Target natural-log entropy.
        s0: Lower scale bound.
        s1: Upper scale bound.
        d: Remaining bisection depth.

    Returns:
        Selected scale in the closed interval ``[s0, s1]``.
    """
    if d == 0:
        return (s0 + s1) / 2.0
    s2 = (s0 + s1) / 2.0
    f0 = vec_perplexity(x, s0)
    f1 = vec_perplexity(x, s1)
    f2 = vec_perplexity(x, s2)

    if f0 >= a:
        return s0
    if f1 <= a:
        return s1
    if f2 > a:
        return row_perplexity_solve(x, a, s0, s2, d - 1)
    if f2 < a:
        return row_perplexity_solve(x, a, s2, s1, d - 1)
    return s2


def fix_row_perplexity(P: np.ndarray, a: float) -> np.ndarray:
    """Rescale each affinity row to a target perplexity.

    Diagonal entries are excluded, then restored as zeros. The input is read
    but not modified.

    Args:
        P: Square positive affinity matrix of shape ``(n_samples, n_samples)``.
        a: Positive target perplexity.

    Returns:
        Row-normalized matrix with the same shape as ``P``.
    """
    rv = np.zeros([P.shape[0]] * 2)
    ent_p = np.log2(a) * np.log(2)
    for i in range(P.shape[0]):
        x = P[i, :].copy()
        x /= x.sum()
        x = np.concatenate((x[:i], x[(i + 1) :]))
        x = -np.log(x)
        c = row_perplexity_solve(x, ent_p, 1.0e-12, 1000, 20)
        _, x = adj_perplexity(x, c)
        rv[i, :i] = x[:i]
        rv[i, (i + 1) :] = x[i:]

    return rv


def get_pmat_vlen(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray,
    targ_perplexity: Optional[float] = None,
) -> np.ndarray:
    """Construct affinities for variable-length mixture observations.

    Args:
        posterior_mat: Component posteriors with shape ``(n_samples,
            n_components)``.
        ll_mat: Component log densities with the same shape.
        targ_perplexity: Optional row perplexity target.

    Returns:
        Directed affinity matrix of shape ``(n_samples, n_samples)``, scaled
        by ``1 / n_samples`` and with a zero diagonal unless numerical flooring
        replaces it.
    """
    with np.errstate(divide="ignore"):

        n = len(posterior_mat)
        z_ij = posterior_mat
        l_ij = ll_mat
        v_ij = l_ij.max(axis=1, keepdims=True)
        g_ij = np.exp(l_ij - v_ij)
        p_ij = np.dot(g_ij, z_ij.T)
        np.fill_diagonal(p_ij, 0)
        np.log(p_ij, out=p_ij)
        p_ij += v_ij.T
        p_ij -= np.max(p_ij, axis=1, keepdims=True)
        np.exp(p_ij, out=p_ij)
        p_ij /= np.sum(p_ij, axis=1, keepdims=True)
        p_ij /= np.sum(p_ij, axis=0, keepdims=True)
        p_ij = p_ij.T

        np.maximum(p_ij, 1.0e-128, out=p_ij)
        if targ_perplexity is not None:
            p_ij = fix_row_perplexity(p_ij, targ_perplexity)

        p_ij /= n
        return np.asarray(p_ij)


def get_pmat(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray,
    targ_perplexity: Optional[float] = None,
    vlen: bool = False,
) -> np.ndarray:
    """Construct high-dimensional affinities from mixture responsibilities.

    Args:
        posterior_mat: Component posteriors with shape ``(n_samples,
            n_components)``.
        ll_mat: Component log densities with the same shape.
        targ_perplexity: Optional row perplexity target.
        vlen: Use the variable-length normalization implemented by
            :func:`get_pmat_vlen`.

    Returns:
        Directed affinity matrix of shape ``(n_samples, n_samples)``, scaled
        by ``1 / n_samples``.
    """
    if vlen:
        return get_pmat_vlen(posterior_mat, ll_mat, targ_perplexity)

    with np.errstate(divide="ignore"):

        n = len(posterior_mat)
        z_ij = posterior_mat
        l_ij = ll_mat
        v_ij = l_ij.max(axis=1, keepdims=True)
        g_ij = np.exp(l_ij - v_ij)
        p_ij = np.dot(g_ij, z_ij.T)
        np.fill_diagonal(p_ij, 0)
        np.log(p_ij, out=p_ij)
        p_ij += v_ij
        p_ij -= np.max(p_ij, axis=0, keepdims=True)
        np.exp(p_ij, out=p_ij)
        p_ij /= np.sum(p_ij, axis=0, keepdims=True)
        p_ij = p_ij.T

        np.maximum(p_ij, 1.0e-128, out=p_ij)
        if targ_perplexity is not None:
            p_ij = fix_row_perplexity(p_ij, targ_perplexity)

        p_ij /= n
        return np.asarray(p_ij)


def t_cond_prob_mat(tx: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized Student-t affinities in an embedding.

    Args:
        tx: Embedding coordinates of shape ``(n_samples, n_dimensions)``.
        alpha: Positive degrees-of-freedom parameter.

    Returns:
        A pair of ``(Q, kernel)`` matrices, each with shape ``(n_samples,
        n_samples)``. ``Q`` sums to one and has a zero diagonal.
    """
    n = tx.shape[0]

    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d_ij = np.dot(-2 * tx, tx.T)
    d_ij += rsum
    d_ij += rsum.T + 1
    np.power(d_ij, -(alpha + 1.0) / 2.0, out=d_ij)

    d_ij[np.arange(n), np.arange(n)] = 0
    q_ij = d_ij / np.sum(d_ij)

    return q_ij, d_ij


def t_cond_prob_mat_alpha(
    tx: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Student-t affinities and squared distances for alpha updates.

    Args:
        tx: Embedding coordinates of shape ``(n_samples, n_dimensions)``.
        alpha: Positive degrees-of-freedom parameter.

    Returns:
        Normalized affinity and squared-distance matrices, both with shape
        ``(n_samples, n_samples)``.
    """
    n = tx.shape[0]

    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d_ij = np.dot(-2.0 * tx, tx.T)
    d_ij += rsum
    d_ij += rsum.T

    c_ij = np.power((d_ij / alpha) + 1.0, -(alpha + 1.0) / 2.0)
    c_ij[np.arange(n), np.arange(n)] = 0
    c_ij /= np.sum(c_ij)

    return c_ij, d_ij


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def update_embed(
    P: np.ndarray,
    Y: np.ndarray,
    iY: np.ndarray,
    gains: np.ndarray,
    momentum: float,
    eta: float,
    alpha: float,
    min_gain: float,
    min_value: float = 1.0e-128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one momentum gradient update to an embedding.

    The coordinates are updated in place and recentered to zero column means.
    Sign changes between the gradient and previous increments increase gains;
    matching signs decay them, subject to ``min_gain``.

    Args:
        P: Target affinities of shape ``(n_samples, n_samples)``.
        Y: Coordinates of shape ``(n_samples, n_dimensions)``.
        iY: Previous coordinate increments with the same shape as ``Y``.
        gains: Per-coordinate gains with the same shape as ``Y``.
        momentum: Multiplier applied to previous increments.
        eta: Gradient learning rate.
        alpha: Positive degrees-of-freedom parameter.
        min_gain: Lower bound for adaptive gains.
        min_value: Numerical floor applied to low-dimensional affinities.

    Returns:
        Updated coordinates, increments, gains, and an affinity matrix of
        shape ``(n_samples, n_samples)``.
    """
    nn, mm = Y.shape

    Q, d_ij = t_cond_prob_mat(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    PQ = P - Q
    dC = np.zeros((nn, mm))

    for i in range(nn):
        dCi = PQ[i, :] * d_ij[i, :]
        dC[i, :] = np.dot((Y[i, None, :] - Y).T, dCi)
    dC *= (2.0 * alpha + 2.0) / alpha

    gains = (gains + 0.2) * ((dC > 0) != (iY > 0)) + (gains * 0.8) * (
        (dC > 0) == (iY > 0)
    )
    gains[gains < min_gain] = min_gain

    iY = momentum * iY - eta * (gains * dC)
    Y += iY
    Y -= np.mean(Y, axis=0, keepdims=True)

    return Y, iY, gains, Q


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def update_alpha(
    P: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    min_alpha: float,
    min_value: float,
    max_its: int = 30,
    step: float = 0.1,
    eps: float = 1.0e-4,
) -> float:
    """Optimize the Student-t degrees-of-freedom parameter.

    A bounded Newton-style update minimizes KL divergence, with each step
    limited to a relative change of ``step``. Iteration stops at ``max_its`` or
    when the KL reduction is less than ``eps``.

    Args:
        P: Target affinities of shape ``(n_samples, n_samples)``.
        Y: Coordinates of shape ``(n_samples, n_dimensions)``.
        alpha: Starting value.
        min_alpha: Lower bound for returned values.
        min_value: Numerical floor for low-dimensional affinities.
        max_its: Maximum alpha updates.
        step: Maximum relative change per update.
        eps: Minimum KL reduction required to continue.

    Returns:
        Optimized alpha value, no smaller than ``min_alpha``.
    """
    Q, e_ij = t_cond_prob_mat_alpha(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    PQ = P - Q
    GG = np.bitwise_and(P > 0, Q > 0)
    CC = np.log(P[GG] / Q[GG]) * P[GG]
    CC = CC.sum()
    its = 0

    while True:

        e_ij_a = (e_ij / alpha) + 1
        dCa = (
            np.log(e_ij_a) * 0.5
            + (((-alpha - 1.0) / (2.0 * alpha * alpha)) * e_ij / e_ij_a)
        ) * PQ
        dCa = dCa[GG].sum()

        if np.isfinite(dCa) and (dCa != 0):
            if (alpha - (CC / dCa)) < (alpha * (1.0 - step)):
                alpha *= 1.0 - step
            elif (alpha - (CC / dCa)) > (alpha * (1.0 + step)):
                alpha *= 1.0 + step
            elif np.abs(dCa) > 0.0:
                alpha = alpha - (CC / dCa)

        alpha = max(alpha, min_alpha)

        its += 1
        if its >= max_its:
            break

        CC_old = CC
        Q, e_ij = t_cond_prob_mat_alpha(Y, alpha)
        np.maximum(Q, min_value, out=Q)
        PQ = P - Q
        GG = np.bitwise_and(P > 0, Q > 0)
        CC = np.log(P[GG] / Q[GG]) * P[GG]
        CC = CC.sum()

        if CC_old - CC < eps:
            break

    return alpha


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def htsne(
    data: Sequence[T],
    emb_dim: int = 2,
    alpha: float = 1.0,
    max_components: int = 30,
    mix_threshold_count: float = 0.5,
    Y: Optional[np.ndarray] = None,
    perplexity: Optional[int] = None,
    max_its: int = 1000,
    print_iter: int = 100,
    eta: int = 500,
    momentum: float = 0.8,
    min_gain: float = 0.01,
    min_value: float = 1.0e-128,
    optimize_alpha: bool = False,
    min_alpha: float = 1.0e-6,
    max_alpha_its: int = 3,
    seed: Optional[int] = None,
    comp_estimator: Optional[Any] = None,
    mix_model: Optional[MixtureModel] = None,
    variable_length: bool = False,
) -> np.ndarray:
    """Embed heterogeneous observations using mixture-derived affinities.

    A supplied mixture is reused; otherwise a pruned Dirichlet-process mixture
    is fitted. Its component posteriors and log densities define the target
    affinity matrix. When ``Y`` is absent, ``seed`` controls both mixture
    fitting and a small Gaussian coordinate initialization, followed by 100
    early-exaggeration updates and then ``max_its`` regular updates. A supplied
    NumPy ``Y`` may be updated in place. Progress is printed to standard output.

    Args:
        data: Nonempty heterogeneous observations.
        emb_dim: Output dimensionality when ``Y`` is absent.
        alpha: Initial Student-t degrees-of-freedom parameter.
        max_components: Maximum components used when fitting a mixture.
        mix_threshold_count: Minimum effective component count retained after
            mixture fitting.
        Y: Optional initial coordinates of shape ``(len(data), emb_dim)``.
        perplexity: Optional target row perplexity for affinity
            construction.
        max_its: Number of regular embedding updates.
        print_iter: Positive interval for standard-output progress messages.
        eta: Gradient learning rate.
        momentum: Momentum for regular updates.
        min_gain: Lower bound for adaptive gradient gains.
        min_value: Numerical floor for affinities.
        optimize_alpha: Optimize ``alpha`` after each embedding update.
        min_alpha: Lower bound used during alpha optimization.
        max_alpha_its: Maximum alpha updates per embedding iteration.
        seed: Seed for a local legacy NumPy random state. ``None`` uses OS
            entropy.
        comp_estimator: Optional estimator for one mixture component.
        mix_model: Optional pre-fitted mixture distribution.
        variable_length: Use variable-length affinity normalization.

    Returns:
        Centered embedding coordinates with shape ``(len(data), emb_dim)``.

    Raises:
        ValueError: If ``max_components`` is not an integer greater than one.
        RuntimeError: If mixture fitting produces no components.
    """
    rng = RandomState(seed) if seed is not None else RandomState()
    mix_model, enc_data, z_ij = prepare_mixture_model(
        data=data,
        rng=rng,
        max_components=max_components,
        mix_threshold_count=mix_threshold_count,
        max_its=max_its,
        print_iter=print_iter,
        comp_estimator=comp_estimator,
        mix_model=mix_model,
    )
    # Log component density for each point [x | z]
    l_ij = mix_model.seq_component_log_density(enc_data)
    # Construct high-dim neighborhood matrix
    P = get_pmat(z_ij, l_ij, targ_perplexity=perplexity, vlen=variable_length)
    P = np.asarray(P)
    P += P.T
    P /= np.sum(P)
    np.maximum(P, min_value, out=P)
    if Y is None:
        nn = P.shape[0]
        Y = rng.randn(nn, emb_dim) * 1.0e-4
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))
        P *= 4
        for i in range(20):
            Y, iY, gains, Q = update_embed(
                P=P,
                Y=Y,
                iY=iY,
                gains=gains,
                momentum=0.5,
                eta=eta,
                alpha=alpha,
                min_gain=min_gain,
                min_value=min_value,
            )
        for i in range(80):
            Y, iY, gains, Q = update_embed(
                P=P,
                Y=Y,
                iY=iY,
                gains=gains,
                momentum=0.5,
                eta=eta,
                alpha=alpha,
                min_gain=min_gain,
                min_value=min_value,
            )
        P /= 4
    else:
        Y = np.asarray(Y)
        nn = Y.shape[0]
        emb_dim = Y.shape[1]
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))

    for i in range(1, max_its + 1):
        Y, iY, gains, Q = update_embed(
            P=P,
            Y=Y,
            iY=iY,
            gains=gains,
            momentum=momentum,
            eta=eta,
            alpha=alpha,
            min_gain=min_gain,
            min_value=min_value,
        )
        if optimize_alpha:
            alpha = update_alpha(
                P=P,
                Y=Y,
                alpha=alpha,
                min_alpha=min_alpha,
                min_value=min_value,
                max_its=max_alpha_its,
            )
        if (i % print_iter) == 0:
            KL = np.bitwise_and(P > 0, Q > 0)
            KL = np.dot(P[KL], (np.log(P[KL]) - np.log(Q[KL])))
            print(f"Iteration {i}: alpha = {alpha:f}, KL(P||Q)={KL}")

    return Y


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def dpmsne(
    P: Optional[np.ndarray] = None,
    emb_dim: int = 2,
    alpha: float = 1.0,
    Y: Optional[np.ndarray] = None,
    max_its: int = 1000,
    print_iter: int = 100,
    eta: int = 500,
    momentum: float = 0.8,
    min_gain: float = 0.01,
    min_value: float = 1.0e-128,
    optimize_alpha: bool = False,
    min_alpha: float = 1.0e-6,
    max_alpha_its: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Embed a precomputed affinity matrix with the t-SNE optimizer.

    ``P`` is converted with :func:`numpy.asarray` and normalized in place when
    possible. When ``Y`` is absent, ``seed`` controls Gaussian initialization,
    followed by 100 early-exaggeration updates and ``max_its`` regular updates.
    A supplied NumPy ``Y`` may be updated in place. Progress is printed to
    standard output.

    Args:
        P: Floating-point square affinity matrix with shape ``(n_samples,
            n_samples)``.
        emb_dim: Output dimensionality when ``Y`` is absent.
        alpha: Initial Student-t degrees-of-freedom parameter.
        Y: Optional initial coordinates of shape ``(n_samples, emb_dim)``.
        max_its: Number of regular embedding updates.
        print_iter: Positive interval for standard-output progress messages.
        eta: Gradient learning rate.
        momentum: Momentum for regular updates.
        min_gain: Lower bound for adaptive gradient gains.
        min_value: Numerical floor for affinities.
        optimize_alpha: Optimize ``alpha`` after each embedding update.
        min_alpha: Lower bound used during alpha optimization.
        max_alpha_its: Maximum alpha updates per embedding iteration.
        seed: Seed for Gaussian initialization. Ignored when ``Y`` is supplied.

    Returns:
        Centered embedding coordinates with shape ``(n_samples, emb_dim)``.
    """
    P = np.asarray(P)
    P /= np.sum(P)

    if Y is None:
        rng = np.random.RandomState(seed)
        nn = P.shape[0]
        Y = rng.randn(nn, emb_dim) * 1.0e-4
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))
        P *= 4
        for i in range(20):
            Y, iY, gains, Q = update_embed(
                P, Y, iY, gains, 0.5, eta, alpha, min_gain, min_value
            )
        for i in range(80):
            Y, iY, gains, Q = update_embed(
                P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value
            )
        P /= 4
    else:
        Y = np.asarray(Y)
        nn = Y.shape[0]
        emb_dim = Y.shape[1]
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))

    for i in range(1, max_its + 1):
        Y, iY, gains, Q = update_embed(
            P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value
        )
        if optimize_alpha:
            alpha = update_alpha(P, Y, alpha, min_alpha, min_value, max_alpha_its)
        if (i % print_iter) == 0:
            KL = np.bitwise_and(P > 0, Q > 0)
            KL = np.dot(P[KL], (np.log(P[KL]) - np.log(Q[KL])))
            print(f"Iteration {i}: alpha = {alpha:f}, KL(P||Q)={KL}")

    return Y
