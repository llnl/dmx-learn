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
    """Computes the log-pseudo-determinant for a symmetric dense matrix.

    Args:
        x_mat (np.ndarray): 2-d Numpy array representing a matrix.

    Returns:
        float, log-pseudo-determinant.

    """
    eigs = np.abs(np.linalg.eig(x_mat))
    eigs = eigs[eigs != 0]

    if len(eigs) > 0:
        return float(np.sum(np.log(eigs)))
    return -math.inf


def polygamma_loc(
    n: int, y: float, out: Optional[np.ndarray] = None
) -> Union[np.ndarray, float]:
    """
    Computes the localized polygamma function.

    This function calculates the polygamma function for a given order `n` and input `y`.
    The calculation uses the Riemann zeta function and the Gamma function. If an `out`
    array is provided, the result is stored in the array.

    Args:
        n (int): The order of the polygamma function (non-negative integer).
        y (float): The input value for the polygamma function.
        out (Optional[np.ndarray]): An optional array to store the result.
            If provided, the computation result is stored in this array.
            Defaults to `None`.

    Returns:
        Union[np.ndarray, float]:
            - If `out` is provided, returns the `out` array containing the result.
            - If `out` is not provided, returns the computed result as a float.
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
    """Trigamma function.

    Args:
        y (Array-like): An array-like or float/int.
        out (np.ndarray); Store output in this variable.

    Returns:
        Numpy array of trigamma function evaluated at y.

    """
    return zeta(2, y, out=out)  # type: ignore[no-any-return]


def digammainv(
    y: Union[np.ndarray, float], out: Optional[np.ndarray] = None
) -> Union[np.ndarray, float]:
    """Inverse digamma function evaluated on y.

    Args:
        y (Union[np.ndarray, float]): Numpy array of values to be evaluated
            or single value.
        out (Optional[np.ndarray]): Deprecated. Kept for consistency with other files.

    Returns:
        Numpy array if y is numpy array else float.

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
    """
    Computes the Stirling number of the second kind.

    The Stirling number of the second kind, S(n, k), represents the number of ways
    to partition a set of `n` elements into `k` non-empty subsets. This function
    uses a recursive approach to compute the value.

    Args:
        n (int): The total number of elements in the set (must be positive).
        k (int): The number of non-empty subsets (must be positive).

    Returns:
        int: The Stirling number of the second kind, S(n, k).

    """
    assert n > 0 and k > 0

    if n == 0 and k == 0:
        return 1
    if n == 0:
        return 0
    if k == 0:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)
