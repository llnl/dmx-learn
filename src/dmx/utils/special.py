"""Defines log-pseudo-determinant and special-function helpers."""

import math
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any, Iterable, List, Optional, Union

import numpy as np

from dmx.arithmetic import exp


@lru_cache(maxsize=1)
def _scipy_special() -> ModuleType:
    return import_module("scipy.special")


def beta(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.beta` through a lazy import."""
    return _scipy_special().beta(*args, **kwargs)


def betaln(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.betaln` through a lazy import."""
    return _scipy_special().betaln(*args, **kwargs)


def digamma(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.digamma` through a lazy import."""
    return _scipy_special().digamma(*args, **kwargs)


def gamma(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.gamma` through a lazy import."""
    return _scipy_special().gamma(*args, **kwargs)


def gammaln(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.gammaln` through a lazy import."""
    return _scipy_special().gammaln(*args, **kwargs)


def ive(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.ive` through a lazy import."""
    return _scipy_special().ive(*args, **kwargs)


def zeta(*args: Any, **kwargs: Any) -> Any:
    """Proxy `scipy.special.zeta` through a lazy import."""
    return _scipy_special().zeta(*args, **kwargs)


D1 = float(digamma(1.0))


def logpdet(x_mat: np.ndarray) -> float:
    """Compute the log-pseudo-determinant of a symmetric dense matrix.

    Args:
        x_mat: Square matrix of shape ``(n, n)``.

    Returns:
        Sum of logarithms of nonzero absolute eigenvalues, or negative
        infinity when all eigenvalues are zero.

    Raises:
        np.linalg.LinAlgError: If eigenvalue computation does not converge.
    """
    eigs = np.abs(np.linalg.eig(x_mat))
    eigs = eigs[eigs != 0]

    if len(eigs) > 0:
        return float(np.sum(np.log(eigs)))
    return -math.inf


def polygamma_loc(
    n: int, y: float, out: Optional[np.ndarray] = None
) -> Union[np.ndarray, float]:
    """Evaluate a polygamma function through Hurwitz zeta.

    The result is ``(-1)**(n + 1) * gamma(n + 1) * zeta(n + 1, y)``.

    Args:
        n: Nonnegative derivative order.
        y: Scalar evaluation point.
        out: Optional output array accepted by SciPy's zeta ufunc.

    Returns:
        ``out`` after in-place evaluation when provided, otherwise a scalar.
    """
    if out is not None:
        fac2 = zeta(n + 1, y, out=out)
        fac2 *= (-1.0) ** (n + 1) * gamma(n + 1.0)
    else:
        fac2 = (-1.0) ** (n + 1) * gamma(n + 1.0) * zeta(n + 1, y)

    return fac2  # type: ignore[no-any-return]


def trigamma(
    y: Union[np.ndarray, int, float, Iterable, List[float]],
    out: Optional[np.ndarray] = None,
) -> Union[np.ndarray, float]:
    """Evaluate the trigamma function elementwise.

    Args:
        y: Scalar or array-like evaluation points.
        out: Optional array receiving results in place.

    Returns:
        A scalar for scalar input or an array broadcast to the shape of ``y``.

    """
    return zeta(2, y, out=out)  # type: ignore[no-any-return]


def digammainv(
    y: Union[np.ndarray, float], out: Optional[np.ndarray] = None
) -> Union[np.ndarray, float]:
    """Approximate the inverse digamma function with five Newton steps.

    Array inputs preserve their shape. Positive infinity maps to positive
    infinity; other non-finite array entries remain zero. The ``out`` argument
    is retained for compatibility but is ignored.

    Args:
        y: Scalar or NumPy array of digamma values.
        out: Deprecated and ignored.

    Returns:
        Approximate inverse values, with the same shape as an array input.

    """
    _ = out

    if isinstance(y, np.ndarray):

        rv = np.zeros(y.shape, dtype=float)
        rv[np.isposinf(y)] = np.inf

        Q = np.isfinite(y)
        z = y[Q]
        M = z >= -2.22
        x = M * (exp(z) + 0.5) + (1.0 - M) * (-1.0 / (z - D1))

        t1 = np.zeros(x.shape, dtype=float)
        t2 = np.zeros(x.shape, dtype=float)

        for _ in range(5):
            digamma(x, out=t1)
            zeta(2, x, out=t2)

            t1 -= z
            t1 /= t2
            x -= t1

        rv[Q] = x
        x = rv

    else:
        m = y >= -2.22
        x = m * (exp(y) + 0.5) + (1.0 - m) * (-1.0 / (y - D1))

        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)

    return x  # type: ignore[no-any-return]


def stirling2(n: int, k: int) -> int:
    """Compute a Stirling number of the second kind recursively.

    ``S(n, k)`` counts partitions of ``n`` labeled elements into ``k`` nonempty
    unlabeled subsets.

    Args:
        n: Positive number of elements.
        k: Positive number of subsets.

    Returns:
        The integer ``S(n, k)``.

    Raises:
        AssertionError: If ``n`` or ``k`` is not positive.
    """
    assert n > 0 and k > 0

    if n == 0 and k == 0:
        return 1
    if n == 0:
        return 0
    if k == 0:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)
