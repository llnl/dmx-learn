"""Resolve seeds and random states for stochastic entry points.

Fitting, embedding, and sampling helpers accept either an integer ``seed`` or a
``numpy.random.RandomState`` (``rng``). They resolve them with
:func:`resolve_rng` so that:

- a given ``seed`` always produces the same results;
- a given ``rng`` is used as is, so existing reproducible calls are unchanged;
- with neither, a fresh seed is drawn from operating-system entropy, used to
  build a private ``RandomState``, and reported, so the run can be repeated by
  passing that seed back. NumPy's global random state is never used.

See the "Seeds and reproducibility" page of the documentation for the seed
parameters of every entry point.
"""

from __future__ import annotations

import secrets
from typing import Any, Optional, Union

import numpy as np
from numpy.random import RandomState

__all__ = [
    "MAX_SEED",
    "new_seed",
    "resolve_seed",
    "resolve_sampler_seed",
    "resolve_rng",
]

#: Largest seed accepted by :func:`resolve_seed` (``numpy.random.RandomState``
#: accepts seeds in ``[0, 2**32 - 1]``).
MAX_SEED = 2**32 - 1


def new_seed() -> int:
    """Draw a fresh seed from operating-system entropy.

    Returns:
        Integer seed in ``[0, 2**31 - 1]``, independent of NumPy's global
        random state.
    """
    return secrets.randbelow(2**31)


def resolve_seed(seed: Optional[Union[int, np.integer[Any]]]) -> int:
    """Return ``seed`` after validation, or a fresh seed when it is ``None``.

    Args:
        seed: Integer seed in ``[0, 2**32 - 1]``, or ``None``.

    Returns:
        The seed to use.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is outside ``[0, 2**32 - 1]``.
    """
    if seed is None:
        return new_seed()
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, not {type(seed).__name__}.")
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], not {seed}.")
    return seed


def resolve_sampler_seed(seed: Any) -> Any:
    """Resolve a sampler seed while accepting every seed NumPy accepts.

    ``None`` draws a fresh seed and integers are validated as in
    :func:`resolve_seed`. Any other value, such as a sequence of integers, is
    returned unchanged for ``numpy.random.RandomState`` to interpret, so
    samplers keep accepting the seeds they accepted before.

    Args:
        seed: ``None``, an integer, or another ``RandomState`` seed.

    Returns:
        The seed to build the sampler's ``RandomState`` from.
    """
    if seed is None or (
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
    ):
        return resolve_seed(seed)
    return seed


def resolve_rng(
    seed: Optional[Union[int, np.integer[Any]]] = None,
    rng: Optional[RandomState] = None,
) -> tuple[RandomState, Optional[int]]:
    """Return the random state to use and the seed it was built from.

    Args:
        seed: Integer seed, or ``None``.
        rng: Caller-owned random state, or ``None``.

    Returns:
        ``(rng, seed)``. When ``rng`` is given it is returned unchanged with
        ``seed=None``, since its seed is unknown. Otherwise a new
        ``RandomState`` is built from ``seed``, or from a fresh seed when
        ``seed`` is ``None``, and that seed is returned.

    Raises:
        ValueError: If both ``seed`` and ``rng`` are given, or ``seed`` is
            out of range.
        TypeError: If ``seed`` is not an integer.
    """
    if rng is not None:
        if seed is not None:
            raise ValueError("Pass seed or rng, not both.")
        return rng, None
    used = resolve_seed(seed)
    return RandomState(used), used
