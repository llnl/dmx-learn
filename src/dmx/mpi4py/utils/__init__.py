"""Shared infrastructure for MPI-enabled modeling workflows.

This package separates orchestration utilities from the low-level collectives
in :mod:`dmx.mpi4py.stats` and :mod:`dmx.mpi4py.bstats`. Import optimization,
automatic-modeling, embedding, and option helpers from their defining
``dmx.mpi4py.utils`` submodules. The package root intentionally exposes only
:func:`get_runtime_attr`, which supports lazy access to optional MPI-dependent
objects and avoids importing those workflow modules eagerly.

The submodule functions are collective operations unless documented otherwise;
all MPI ranks must enter them consistently, and callers must install the
optional MPI dependencies before invoking them.
"""

from importlib import import_module
from typing import Any


def get_runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load an attribute lazily from a module."""
    return getattr(import_module(module_name), attr_name)
