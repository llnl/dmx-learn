"""Shared infrastructure for MPI-enabled modeling workflows.

This package separates orchestration utilities from the low-level collectives
in :mod:`dmx.mpi4py.stats` and :mod:`dmx.mpi4py.bstats`. Import optimization,
automatic-modeling, embedding, and option helpers from their defining
``dmx.mpi4py.utils`` submodules. The package root intentionally exposes only
:func:`get_runtime_attr`, which supports lazy access to optional MPI-dependent
objects and avoids importing those workflow modules eagerly, and
:func:`resolve_rng_mpi`, which gives every rank the same random state.

The submodule functions are collective operations unless documented otherwise;
all MPI ranks must enter them consistently, and callers must install the
optional MPI dependencies before invoking them.
"""

from importlib import import_module
from typing import Any, Optional

from numpy.random import RandomState

from dmx.utils.seeding import resolve_rng, resolve_seed


def get_runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load an attribute lazily from a module."""
    return getattr(import_module(module_name), attr_name)


def resolve_rng_mpi(
    comm: Any, seed: Optional[int] = None, rng: Optional[RandomState] = None
) -> tuple[RandomState, Optional[int]]:
    """Return the random state to use on this rank and the seed behind it.

    This is collective over ``comm`` when neither ``seed`` nor ``rng`` is
    given: rank 0 draws a fresh seed from OS entropy and broadcasts it, so
    every rank builds the same random state. A given ``seed`` is used on every
    rank, and a given ``rng`` is returned unchanged with seed ``None``.

    Args:
        comm: MPI communicator, such as ``MPI.COMM_WORLD``.
        seed: Integer seed, or ``None``.
        rng: Caller-owned random state, or ``None``.

    Returns:
        ``(rng, seed)`` as returned by :func:`dmx.utils.seeding.resolve_rng`.

    Raises:
        ValueError: If both ``seed`` and ``rng`` are given.
    """
    if seed is None and rng is None:
        seed = resolve_seed(None) if comm.Get_rank() == 0 else None
        seed = comm.bcast(seed, root=0)
    return resolve_rng(seed, rng)
