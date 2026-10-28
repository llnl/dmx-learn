"""Single-process tests for verbose and progress_callback on the MPI helpers.

The MPI helpers initialize MPI when ``dmx.mpi4py.stats`` or
``dmx.mpi4py.bstats`` is imported. These tests replace those modules and the
MPI runtime with size-1 fakes that delegate to the non-MPI ``dmx.stats`` and
``dmx.bstats`` functions, so they run under plain ``pytest`` without
``mpiexec``. Collective behavior across ranks is covered by the other tests in
this directory, which are run with ``mpiexec -n 4``.
"""

import importlib
import sys
import types
from typing import Any, Iterator

import numpy as np
import pytest

import dmx.bstats
import dmx.mpi4py.utils
import dmx.stats
from dmx.bstats.gaussian import GaussianEstimator as BayesianGaussianEstimator
from dmx.stats import GaussianEstimator, MixtureEstimator
from dmx.utils.progress import ProgressEvent

DATA = list(np.random.RandomState(3).normal(0.0, 1.0, 60))
FAKE_TARGETS = (
    "dmx.mpi4py.utils.estimation",
    "dmx.mpi4py.utils.bestimation",
    "dmx.mpi4py.utils.automatic",
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


def _mixture_estimator() -> MixtureEstimator:
    """Return a two-component Gaussian mixture estimator."""
    return MixtureEstimator([GaussianEstimator()] * 2)


def test_optimize_mpi_default_output(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    optimize_mpi = mpi_utils["estimation"].optimize_mpi
    optimize_mpi(
        DATA,
        _mixture_estimator(),
        max_its=3,
        delta=None,
        rng=np.random.RandomState(1),
        out=sys.stdout,
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert all(line.startswith("Iteration ") for line in lines)
    assert "ln[p_mat(Data|Model)]=" in lines[0]


def test_optimize_mpi_quiet_with_one_event_per_iteration(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[ProgressEvent] = []
    mpi_utils["estimation"].optimize_mpi(
        DATA,
        _mixture_estimator(),
        max_its=4,
        delta=None,
        rng=np.random.RandomState(1),
        out=sys.stdout,
        print_iter=3,
        verbose=False,
        progress_callback=events.append,
    )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert [e.iteration for e in events] == [1, 2, 3, 4]
    assert all(e.source == "optimize_mpi" and e.event == "iteration" for e in events)
    assert all("validation_log_likelihood" in e.metrics for e in events)


def test_best_of_mpi_quiet_with_iteration_and_trial_events(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[ProgressEvent] = []
    mpi_utils["estimation"].best_of_mpi(
        DATA,
        None,
        _mixture_estimator(),
        trials=2,
        max_its=3,
        max_its_cnt=2,
        init_p=0.5,
        delta=None,
        rng=np.random.RandomState(1),
        out=sys.stdout,
        verbose=False,
        progress_callback=events.append,
    )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    kinds = [(e.source, e.event) for e in events]
    # 2 trials of 3 iterations and a trial event each, then 2 refinement steps.
    assert kinds == (
        [("best_of_mpi", "iteration")] * 3
        + [("best_of_mpi", "trial")]
        + [("best_of_mpi", "iteration")] * 3
        + [("best_of_mpi", "trial")]
        + [("optimize_mpi", "iteration")] * 2
    )


def test_bstats_optimize_mpi_quiet_with_one_event_per_iteration(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[ProgressEvent] = []
    mpi_utils["bestimation"].optimize_mpi(
        DATA,
        BayesianGaussianEstimator(),
        max_its=3,
        delta=None,
        rng=np.random.RandomState(1),
        out=sys.stdout,
        verbose=False,
        progress_callback=events.append,
    )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert [e.iteration for e in events] == [1, 2, 3]


def test_get_dpm_mixture_mpi_quiet_and_result_event(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[ProgressEvent] = []
    mixture = mpi_utils["automatic"].get_dpm_mixture_mpi(
        DATA,
        estimator=BayesianGaussianEstimator(),
        max_comp=3,
        rng=np.random.RandomState(2),
        max_its=3,
        verbose=False,
        progress_callback=events.append,
    )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert [e.event for e in events].count("iteration") == 3
    result = events[-1]
    assert (result.source, result.event) == ("get_dpm_mixture_mpi", "result")
    assert result.metrics["num_components"] == mixture.num_components


def test_get_dpm_mixture_mpi_default_prints_summary(
    mpi_utils: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    mixture = mpi_utils["automatic"].get_dpm_mixture_mpi(
        DATA,
        estimator=BayesianGaussianEstimator(),
        max_comp=3,
        rng=np.random.RandomState(2),
        max_its=3,
        print_iter=100,
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == f"# Components = {mixture.num_components}"
    assert lines[-2] == str(mixture.w)
