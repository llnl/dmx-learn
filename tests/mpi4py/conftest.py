"""Size-1 MPI stand-ins for running the MPI helpers under plain pytest.

The MPI helpers initialize MPI when ``mpi4py``, ``dmx.mpi4py.stats``, or
``dmx.mpi4py.bstats`` is imported. The ``mpi_utils`` fixture replaces those
modules and the MPI runtime with size-1 fakes that delegate to the non-MPI
``dmx.stats`` and ``dmx.bstats`` functions, imports the helper modules fresh,
and restores ``sys.modules`` afterwards. The answer-key tests in this
directory do not use it and run with ``mpiexec -n 4``.
"""

import importlib
import sys
import types
from typing import Any, Iterator

import pytest

import dmx.bstats
import dmx.mpi4py.utils
import dmx.stats

FAKE_TARGETS = (
    "dmx.mpi4py.utils.estimation",
    "dmx.mpi4py.utils.bestimation",
    "dmx.mpi4py.utils.automatic",
    "dmx.mpi4py.utils.humap",
)


class _FakeComm:
    """Size-1 communicator: collectives return their input."""

    def Get_rank(self) -> int:  # pylint: disable=invalid-name
        """Return rank 0."""
        return 0

    def allreduce(self, value: Any, op: Any = None) -> Any:
        """Return ``value`` unchanged."""
        del op
        return value

    def bcast(self, value: Any, root: int = 0) -> Any:
        """Return ``value`` unchanged."""
        del root
        return value


FAKE_MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm(), LAND="land")


def _fake_module(name: str, **attrs: Any) -> types.ModuleType:
    """Return a module named ``name`` holding ``attrs``."""
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture(name="mpi_utils")
def fixture_mpi_utils() -> Iterator[dict[str, Any]]:
    """Import the MPI helpers against size-1 fakes, then restore sys.modules."""
    fakes = {
        "mpi4py": _fake_module("mpi4py", MPI=FAKE_MPI),
        "dmx.mpi4py.stats": _fake_module(
            "dmx.mpi4py.stats",
            seq_encode_mpi=dmx.stats.seq_encode,
            seq_estimate_mpi=dmx.stats.seq_estimate,
            seq_initialize_mpi=dmx.stats.seq_initialize,
            seq_log_density_sum_mpi=dmx.stats.seq_log_density_sum,
        ),
        "dmx.mpi4py.bstats": _fake_module(
            "dmx.mpi4py.bstats",
            initialize_mpi=dmx.bstats.initialize,
            seq_encode_mpi=dmx.bstats.seq_encode,
            seq_estimate_mpi=dmx.bstats.seq_estimate,
            seq_log_density_sum_mpi=dmx.bstats.seq_log_density_sum,
        ),
    }
    names = (*fakes, *FAKE_TARGETS)
    saved_modules = {name: sys.modules.pop(name, None) for name in names}
    saved_attrs = {
        name: getattr(dmx.mpi4py.utils, name.rsplit(".", 1)[1], None)
        for name in FAKE_TARGETS
    }
    try:
        sys.modules.update(fakes)
        modules = {
            name.rsplit(".", 1)[1]: importlib.import_module(name)
            for name in FAKE_TARGETS
        }

        def runtime_attr(module_name: str, attr_name: str) -> Any:
            if module_name == "mpi4py":
                return FAKE_MPI
            return getattr(modules[module_name.rsplit(".", 1)[1]], attr_name)

        for module in modules.values():
            module.__dict__["get_runtime_attr"] = runtime_attr
        yield modules
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        for name, attr in saved_attrs.items():
            short = name.rsplit(".", 1)[1]
            if attr is None:
                if hasattr(dmx.mpi4py.utils, short):
                    delattr(dmx.mpi4py.utils, short)
            else:
                setattr(dmx.mpi4py.utils, short, attr)
