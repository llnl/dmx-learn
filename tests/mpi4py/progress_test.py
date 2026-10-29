"""Single-process tests for verbose and progress_callback on the MPI helpers.

The ``mpi_utils`` fixture in ``conftest.py`` imports the MPI helpers against
size-1 fakes of MPI, ``dmx.mpi4py.stats``, and ``dmx.mpi4py.bstats``, so these
tests run under plain ``pytest`` without ``mpiexec``. Collective behavior
across ranks is covered by the other tests in this directory, which are run
with ``mpiexec -n 4``.
"""

import sys
from typing import Any

import numpy as np
import pytest

from dmx.bstats.gaussian import GaussianEstimator as BayesianGaussianEstimator
from dmx.stats import GaussianEstimator, MixtureEstimator
from dmx.utils.progress import ProgressEvent

DATA = list(np.random.RandomState(3).normal(0.0, 1.0, 60))


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
    # A caller-supplied rng has no known seed, so the start event reports None.
    assert (events[0].event, events[0].metrics) == ("start", {"seed": None})
    iterations = events[1:]
    assert [e.iteration for e in iterations] == [1, 2, 3, 4]
    assert all(e.source == "optimize_mpi" for e in events)
    assert all(e.event == "iteration" for e in iterations)
    assert all("validation_log_likelihood" in e.metrics for e in iterations)


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
    # A start event, 2 trials of 3 iterations and a trial event each, then the
    # refinement call's start event and 2 refinement steps.
    assert kinds == (
        [("best_of_mpi", "start")]
        + [("best_of_mpi", "iteration")] * 3
        + [("best_of_mpi", "trial")]
        + [("best_of_mpi", "iteration")] * 3
        + [("best_of_mpi", "trial")]
        + [("optimize_mpi", "start")]
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
    assert [e.event for e in events] == ["start"] + ["iteration"] * 3
    assert [e.iteration for e in events[1:]] == [1, 2, 3]


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
