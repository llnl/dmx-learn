from typing import Dict, Sequence, TypeVar, Union

import numpy as np
import torch as tn

T = TypeVar("T")
T1 = TypeVar("T1")


def count_by_value(x: Union[Sequence[T], np.ndarray, tn.Tensor]) -> Dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Args:
        x (Sequence[T]): A sequence of type T or numpy array of type T.

    Returns:
        Dictionary mapping value (type T) to value-count.

    """
    rv: Dict[T, int] = dict()

    for u in x:
        rv[u] = rv.get(u, 0) + 1

    return rv


def int_count_by_value(x: Union[Sequence[T], np.ndarray, tn.Tensor]) -> Dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Args:
        x (Sequence[T]): A sequence of type T or numpy array of type T.

    Returns:
        Dictionary mapping value (type T) to value-count.

    """
    rv: Dict[int, int] = dict()

    for u in x:
        rv[int(u)] = rv.get(int(u), 0) + 1  # type: ignore[arg-type]

    return rv  # type: ignore[return-value]


def bincount1(xv: tn.Tensor, w: tn.Tensor, nv: int) -> tn.Tensor:
    """Take bincount S by N by ids in xv with total number of values nv.

    Args:
        xv (Tensor): Tensor of integers containing the ids for bincount
        w (S by N): Bin count weights
        nv (int): Min length.

    Returns:
        Len nv Tensor.

    """
    s, n = w.shape
    idx = tn.arange(s * n)

    col, row = xv[idx % n], tn.divide(idx, n, rounding_mode="floor")
    return tn.bincount(col + n * row, w.flatten(), minlength=nv * s).reshape((s, -1))
