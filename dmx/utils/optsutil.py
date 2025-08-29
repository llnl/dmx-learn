"""Utility functions for data pre/post processing."""
from pathlib import Path
from collections import defaultdict
from typing import TypeVar, Union, Sequence, Dict, Tuple, List, Callable, Optional
import numpy as np

T = TypeVar('T')
T1 = TypeVar('T1')

def get_parent_directory(filepath: str, levels=0) -> Path:
    """Get the parent directory of a file.
    Args:
        filepath (str): The full path of the file
        levels (int): Number of levels to move up to reach parent.
    Returns:
        Path object containing path to parent directory.
    
    """
    rv = Path(filepath)

    for _ in range(levels):
        rv = rv.parent
    
    return rv

def map_to_integers(x: Sequence[T], val_map: Dict[T, int]) -> List[int]:
    """Map sequence of type T to integers.

    Args:
        x (Sequence[T]): Sequence of type T to be mapped to integers.
        val_map (Dict[T, int]): Dictionary mappings for type T to unique integer values.

    Returns:
        Returns x mapped to a list of integers.

    """
    rv = [None] * len(x)
    for i, u in enumerate(x):
        if u not in val_map:
            val_map[u] = len(val_map)
        rv[i] = val_map[u]
    return rv


def get_inv_map(val_map: Dict[T, T1]) -> Dict[T1, T]:
    """Obtain the inverse dictionary mapping of key/value pairs.

    Args:
        val_map (Dict[T1, T]): Dictionary mapping keys to values.

    Returns:
        Inverse mapping of val_map.

    """
    max_val = max(val_map.values())

    rv = [None] * (max_val + 1)

    for k, v in val_map.items():
        rv[v] = k

    return rv


def text_file(f) -> List[str]:
    """Open a file and split by newline.

    Args
        f: File to be read-in and parsed.

    Returns:
        List of strings split on newline character.

    """
    fin = open(f, 'r')
    rv = fin.read()

    if rv is not None and len(rv) > 0 and rv[-1] == '\n':
        return rv[:-1].split('\n')
    else:
        return rv.split('\n')


def reduce_by_key(f: Callable[[T1, T1], T1], x: Sequence[Tuple[T, T1]]) -> Dict[T, T1]:
    """Reduce sequence of tuple of key value pairs under grouping function f.

    Args:
        f (Callable[[T1, T1], T1]): Function for reducing keys.
        x (Sequence[Tuple[T, T1]): A sequence of key/value pairs.

    Returns:
        Dictionary mapping key types T to value types T1.

    """
    rv = dict()

    for key, val in x:
        if key in rv:
            rv[key] = f(rv[key], val)
        else:
            rv[key] = val

    return rv


def sum_by_key(x: Sequence[Tuple[T, T1]]) -> Dict[T, T1]:
    """Sum values and return dictionary of items with their respective summed values.

    Args:
        x (Sequence[Tuple[T, T1]]): A sequence of tuples of key and value pairs.

    Returns:
        Dictionary of keys with summed values.

    """
    rv = dict()

    for key, val in x:
        if key in rv:
            rv[key] += val
        else:
            rv[key] = val

    return rv


def group_by_key(x: Sequence[Tuple[T, T1]]) -> Dict[T, List[T1]]:
    """Group keys and return dictionary of items with their respective values aggregated as a list.

    Args:
        x (Sequence[Tuple[T, T1]]): A sequence of tuples of key and value pairs.

    Returns:
        Dictionary mapping keys to list of values for respective keys.

    """
    rv = defaultdict(list)

    for key, val in x:
        rv[key].append(val)

    return rv


def group_by(f: Callable[[T], T1], x: Sequence[T]) -> Dict[T1, List[T]]:
    """Maps values in x to key from mapping f. Dictionary mapping keys to list of grouped values in x is returned.

    Args:
        f (Callable[[T], T1]): Function mapping type T to its group 'key' of type T1.
        x (Sequence[T]): Sequence of values to be grouped.

    Returns:
        Dictionary mapping group id 'keys' (type T1) to Lists of values (type T).

    """
    rv = defaultdict(list)

    for val in x:
        key = f(val)
        rv[key].append(val)

    return rv

def count_by_value(x: Union[Sequence[T], np.ndarray]) -> Dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Args:
        x (Sequence[T]): A sequence of type T or numpy array of type T.

    Returns:
        Dictionary mapping value (type T) to value-count.

    """
    rv = dict()

    for u in x:
        rv[u] = rv.get(u, 0) + 1

    return rv


def flat_map(f: Callable[[T], Sequence[T1]], x: Sequence[T]) -> List[T1]:
    """Map values of x under mapping f().

    Args:
        f (Callable[[T], T1]): Maps values of x to sequence of type T1.
        x (Sequence[T]): Seuquence to be mapped under f.

    Returns:
        List of mapped type T1.

    """
    return [u for v in x for u in f(v)]


def least_occurring(
    x: Sequence[T],
    count: Optional[int] = None,
    percent: Optional[float] = None,
    keep_freq: bool = True
) -> Union[List[T], Callable]:
    """
    Identifies the least occurring elements in a sequence.

    This function finds the least frequently occurring elements in a given sequence
    based on the specified `count` or `percent`. Optionally, it can return the filtered
    sequence with only the least occurring elements or just the elements themselves.

    Args:
        x (Sequence[T]): The input sequence to analyze.
        count (Optional[int], optional): The maximum number of least occurring elements to return.
            If specified, the function will return up to `count` least occurring elements. Defaults to `None`.
        percent (Optional[float], optional): The percentage (as a float between 0 and 1) of least occurring
            elements to return. If specified, the function will return the least occurring elements that
            make up this percentage of the total unique elements. Defaults to `None`.
        keep_freq (bool, optional): If `True`, returns the filtered sequence containing only the least
            occurring elements. If `False`, returns a list of the least occurring elements themselves.
            Defaults to `True`.

    Returns:
        Union[List[T], Callable]:
            - If `keep_freq` is `True`, returns a filtered sequence containing only the least occurring elements.
            - If `keep_freq` is `False`, returns a list of the least occurring elements.
            - If neither `count` nor `percent` is specified, the original sequence `x` is returned.
    """
    cnt_map = count_by_value(x).items()
    s_idx = np.argsort([u[1] for u in cnt_map])

    if count is not None:
        n = min(len(s_idx), count)
    elif percent is not None:
        n = max(int(len(s_idx) * percent), 1)
    else:
        return x

    vals = [cnt_map[i][0] for i in s_idx[:n]]

    if keep_freq:
        vals = set(vals)
        return filter(lambda u: u in vals, x)
    else:
        return vals
