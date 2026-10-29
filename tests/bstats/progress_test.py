"""Tests for verbose and progress_callback in dmx.bstats.bestimation."""

import io
from typing import Any, Callable

import numpy as np
import pytest

from dmx.bstats.bestimation import best_of, hill_climb, iterate, optimize
from dmx.bstats.dpm import DirichletProcessMixtureEstimator
from dmx.bstats.gaussian import GaussianEstimator
from dmx.utils.progress import ProgressEvent

DATA = (-2.0, -1.8, -1.6, 1.7, 1.9, 2.1)
MAX_ITS = 3


def _estimator() -> DirichletProcessMixtureEstimator:
    """Create a small homogeneous Gaussian DPM estimator."""
    return DirichletProcessMixtureEstimator(
        tuple(GaussianEstimator() for _ in range(3))
    )


def _run_best_of(**kwargs: Any) -> Any:
    return best_of(
        DATA,
        DATA,
        _estimator(),
        2,
        MAX_ITS,
        1.0,
        None,
        np.random.RandomState(1),
        **kwargs,
    )


def _run_optimize(**kwargs: Any) -> Any:
    return optimize(
        DATA,
        _estimator(),
        max_its=MAX_ITS,
        delta=None,
        init_p=1.0,
        rng=np.random.RandomState(2),
        **kwargs,
    )


def _run_iterate(**kwargs: Any) -> Any:
    return iterate(
        DATA, _estimator(), MAX_ITS, init_p=1.0, rng=np.random.RandomState(3), **kwargs
    )


def _run_hill_climb(**kwargs: Any) -> Any:
    start = optimize(
        DATA,
        _estimator(),
        max_its=1,
        init_p=1.0,
        rng=np.random.RandomState(4),
        out=io.StringIO(),
    )
    return hill_climb(
        DATA, DATA, _estimator(), start, MAX_ITS, lambda v, m: 0.0, **kwargs
    )


RUNNERS: dict[str, Callable[..., Any]] = {
    "best_of": _run_best_of,
    "optimize": _run_optimize,
    "iterate": _run_iterate,
    "hill_climb": _run_hill_climb,
}


@pytest.mark.parametrize("name", list(RUNNERS))
def test_verbose_false_writes_nothing(
    name: str, capfd: pytest.CaptureFixture[str]
) -> None:
    # The default ``out`` is bound at import, so capture at the descriptor level.
    RUNNERS[name](verbose=False)
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("name", list(RUNNERS))
def test_callback_receives_one_event_per_iteration(name: str) -> None:
    events: list[ProgressEvent] = []
    RUNNERS[name](verbose=False, progress_callback=events.append)

    iterations = [e for e in events if e.event == "iteration"]
    assert all(e.source == name for e in events)
    trials = 2 if name == "best_of" else 1
    assert len(iterations) == MAX_ITS * trials
    assert [e.iteration for e in iterations] == list(range(1, MAX_ITS + 1)) * trials


def _drop_start(events: list[ProgressEvent]) -> list[ProgressEvent]:
    """Check for exactly one leading ``"start"`` event and return the rest."""
    assert [e.event for e in events].count("start") == 1
    assert events[0].event == "start"
    return events[1:]


def test_best_of_text_matches_events() -> None:
    out = io.StringIO()
    events: list[ProgressEvent] = []
    _run_best_of(out=out, progress_callback=events.append)
    events = _drop_start(events)

    expected = []
    for e in events:
        if e.event == "iteration":
            expected.append(
                f"Iteration {e.iteration}. LL={e.log_likelihood:f}, "
                f"delta LL={e.delta:e}"
            )
        else:
            vll = e.metrics["validation_log_likelihood"]
            expected.append(f"Trial {e.trial}. VLL={vll:f}")
    assert out.getvalue().splitlines() == expected
    assert [e.trial for e in events if e.event == "trial"] == [1, 2]


def test_optimize_text_matches_events() -> None:
    out = io.StringIO()
    events: list[ProgressEvent] = []
    _run_optimize(out=out, progress_callback=events.append)
    events = _drop_start(events)

    expected = [
        f"Iteration {e.iteration}. LL={e.log_likelihood:f}, dLL={e.delta:e}, "
        f"MLL={e.metrics['model_log_likelihood']:f}, "
        f"dMLL={e.metrics['model_delta']:e}, "
        f"VLL={e.metrics['validation_log_likelihood']:f}"
        for e in events
    ]
    assert out.getvalue().splitlines() == expected


def test_iterate_and_hill_climb_default_text_format() -> None:
    out = io.StringIO()
    _run_iterate(out=out)
    lines = out.getvalue().splitlines()
    assert [line.split("\t")[0] for line in lines] == [
        f"Iteration {i}" for i in range(1, MAX_ITS + 1)
    ]
    assert all(" E[dT]=" in line for line in lines)

    out = io.StringIO()
    events: list[ProgressEvent] = []
    _run_hill_climb(out=out, progress_callback=events.append)
    expected = [
        f"Iteration {e.iteration}. "
        f"LL={e.metrics['validation_log_likelihood']:f}, "
        f"Best LL={e.metrics['best_validation_log_likelihood']:f}, "
        f"Best Score={e.metrics['best_score']:f}"
        for e in events
    ]
    assert out.getvalue().splitlines() == expected


def test_print_iter_throttles_text_but_not_events() -> None:
    out = io.StringIO()
    events: list[ProgressEvent] = []
    _run_optimize(out=out, print_iter=2, progress_callback=events.append)
    events = _drop_start(events)

    assert len(out.getvalue().splitlines()) == 1
    assert len(events) == MAX_ITS
