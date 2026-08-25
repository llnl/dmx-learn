"""Counting helpers for Python, NumPy, and PyTorch containers."""

from typing import Dict, Sequence, TypeVar, Union

import numpy as np
import torch as tn

T = TypeVar("T")
T1 = TypeVar("T1")


def count_by_value(x: Union[Sequence[T], np.ndarray, tn.Tensor]) -> Dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Values are used as yielded by the input container. Iterating a tensor yields
    scalar tensor keys rather than converting them to Python scalars.

    Args:
        x (Union[Sequence[T], numpy.ndarray, torch.Tensor]): Values to count.

    Returns:
        Dict[T, int]: Dictionary mapping each observed value to its count.
    """
    rv: Dict[T, int] = {}

    for u in x:
        rv[u] = rv.get(u, 0) + 1

    return rv


def int_count_by_value(x: Union[Sequence[T], np.ndarray, tn.Tensor]) -> Dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Each value is converted with `int` before counting. For tensor inputs, this
    converts scalar tensor elements to host Python integers and may synchronize
    the device.

    Args:
        x (Union[Sequence[T], numpy.ndarray, torch.Tensor]): Values to count.

    Returns:
        Dict[int, int]: Dictionary mapping integer values to counts.
    """
    rv: Dict[int, int] = {}

    for u in x:
        rv[int(u)] = rv.get(int(u), 0) + 1  # type: ignore[arg-type]

    return rv  # type: ignore[return-value]


def bincount1(xv: tn.Tensor, w: tn.Tensor, nv: int) -> tn.Tensor:
    """Bincount batched weights by one-dimensional ids.

    Args:
        xv (Tensor): One-dimensional integer tensor with shape `(n,)`.
        w (Tensor): Weight tensor with shape `(s, n)`. It must be on a device
            compatible with `xv`.
        nv (int): Number of bins in the result.

    Returns:
        Tensor: Tensor with shape `(s, nv)`. Dtype and device follow the
        behavior of `torch.bincount` for the supplied weights.
    """
    s, n = w.shape
    idx = tn.arange(s * n)

    col, row = xv[idx % n], tn.divide(idx, n, rounding_mode="floor")
    return tn.bincount(col + n * row, w.flatten(), minlength=nv * s).reshape((s, -1))
