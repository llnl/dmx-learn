"""Shared helpers for `dmx.mpi4py.utils`."""

from importlib import import_module
from typing import Any


def get_runtime_attr(module_name: str, attr_name: str) -> Any:
    """Load an attribute lazily from a module."""
    return getattr(import_module(module_name), attr_name)
