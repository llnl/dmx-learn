"""Tests for quiet progress reporting: verbose, logging, and progress callbacks."""

import io
import json
import logging
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pytest
from numpy.random import RandomState

from dmx.stats import GaussianEstimator, MixtureEstimator
from dmx.utils.automatic import get_dpm_mixture
from dmx.utils.estimation import best_of, hill_climb, iterate, optimize
from dmx.utils.htsne import dpmsne, htsne
from dmx.utils.progress import ProgressEvent, ProgressReporter

DATA = [float(x) for x in RandomState(0).normal(0.0, 1.0, 40)]


def _estimator() -> MixtureEstimator:
    """Return a two-component Gaussian mixture estimator."""
    return MixtureEstimator([GaussianEstimator()] * 2)


def _fit() -> Any:
    """Return a fitted model to start hill_climb and iterate from."""
    return optimize(DATA, _estimator(), max_its=2, rng=RandomState(1), verbose=False)


# Each runner takes (out, verbose, callback) and calls one estimation helper.
RUNNERS: dict[str, Callable[..., Any]] = {
    "optimize": lambda out, verbose, cb: optimize(
        DATA,
        _estimator(),
        max_its=4,
        delta=None,
        rng=RandomState(1),
        out=out,
        verbose=verbose,
        progress_callback=cb,
    ),
    "best_of": lambda out, verbose, cb: best_of(
        DATA,
        None,
        _estimator(),
        2,
        3,
        0.5,
        float("-inf"),  # never stop early, so every iteration runs
        RandomState(1),
        out=out,
        verbose=verbose,
        progress_callback=cb,
    ),
    "iterate": lambda out, verbose, cb: iterate(
        DATA,
        _estimator(),
        3,
        rng=RandomState(1),
        out=out,
        verbose=verbose,
        progress_callback=cb,
    ),
    "hill_climb": lambda out, verbose, cb: hill_climb(
        DATA,
        DATA,
        _estimator(),
        _fit(),
        3,
        lambda data, model: 0.0,
        out=out,
        verbose=verbose,
        progress_callback=cb,
    ),
}

# Expected one-based iteration numbers for the runners above.
EXPECTED_ITERATIONS = {
    "optimize": [1, 2, 3, 4],
    "best_of": [1, 2, 3, 1, 2, 3],
    "iterate": [1, 2, 3],
    "hill_climb": [1, 2, 3],
}

# First characters of each helper's existing progress text.
LEGACY_PREFIX = {
    "optimize": "Iteration 1: ln[p_mat(Data|Model)]=",
    "best_of": "Iteration 1. LL=",
    "iterate": "Iteration 1\t E[dT]=",
    "hill_climb": "Iteration 1. LL=",
}


def _drop_start(events: list[ProgressEvent]) -> list[ProgressEvent]:
    """Check there is exactly one leading ``"start"`` event and drop it."""
    assert [e.event for e in events].count("start") == 1
    assert events[0].event == "start"
    return events[1:]


def test_event_to_dict_is_json_ready() -> None:
    event = ProgressEvent(
        "optimize",
        "iteration",
        iteration=np.int64(2),  # type: ignore[arg-type]
        log_likelihood=np.float64(-1.5),
        metrics={"weights": np.array([0.25, 0.75])},
    )
    payload = json.loads(json.dumps(event.to_dict()))

    assert payload["iteration"] == 2
    assert payload["log_likelihood"] == -1.5
    assert payload["metrics"] == {"weights": [0.25, 0.75]}


def test_reporter_logs_at_debug_even_when_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out = io.StringIO()
    reporter = ProgressReporter("demo", logging.getLogger("dmx.test"), out, False)
    with caplog.at_level(logging.DEBUG, logger="dmx.test"):
        reporter.message("Iteration 1. LL=0.0\n")

    assert out.getvalue() == ""
    assert [r.getMessage() for r in caplog.records] == ["Iteration 1. LL=0.0"]


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_default_output_is_unchanged(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    RUNNERS[name](sys.stdout, True, None)

    captured = capsys.readouterr()
    assert captured.out.startswith(LEGACY_PREFIX[name])
    assert captured.err == ""


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_quiet_writes_nothing(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    RUNNERS[name](sys.stdout, False, None)

    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_callback_gets_one_event_per_iteration(name: str) -> None:
    events: list[ProgressEvent] = []
    RUNNERS[name](io.StringIO(), False, events.append)

    iterations = [e.iteration for e in events if e.event == "iteration"]
    assert iterations == EXPECTED_ITERATIONS[name]
    assert all(e.source == name for e in events)


def test_callback_events_carry_likelihoods() -> None:
    events: list[ProgressEvent] = []
    RUNNERS["optimize"](io.StringIO(), False, events.append)
    events = _drop_start(events)

    lls = [float(e.log_likelihood or "nan") for e in events]
    deltas = [float(e.delta or "nan") for e in events]
    assert np.all(np.isfinite(lls)) and np.all(np.isfinite(deltas))
    for i in range(1, len(events)):
        assert deltas[i] == pytest.approx(lls[i] - lls[i - 1])


def test_best_of_reports_trials() -> None:
    events: list[ProgressEvent] = []
    RUNNERS["best_of"](io.StringIO(), False, events.append)

    trials = [e for e in events if e.event == "trial"]
    assert [e.trial for e in trials] == [1, 2]
    assert all("validation_log_likelihood" in e.metrics for e in trials)


def test_quiet_still_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="dmx.utils.estimation"):
        RUNNERS["optimize"](sys.stdout, False, None)

    messages = [r.getMessage() for r in caplog.records]
    # One seed record, then one record per iteration.
    assert len(messages) == 5
    assert messages[0].startswith("optimize seed=")
    assert messages[1].startswith(LEGACY_PREFIX["optimize"])


def _dpm(verbose: bool, callback: Any = None) -> Any:
    """Fit a small DPM; its optimizer writes to the stream given as ``out``."""
    return get_dpm_mixture(
        DATA,
        max_comp=3,
        rng=RandomState(3),
        max_its=5,
        print_iter=1,
        verbose=verbose,
        progress_callback=callback,
    )


def test_get_dpm_mixture_default_prints_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _dpm(True)

    assert f"# Components = {len(model.components)}\n" in capsys.readouterr().out


def test_get_dpm_mixture_quiet_and_events(capsys: pytest.CaptureFixture[str]) -> None:
    events: list[ProgressEvent] = []
    model = _dpm(False, events.append)

    assert capsys.readouterr() == ("", "")
    iterations = [e.iteration for e in events if e.event == "iteration"]
    assert iterations == list(range(1, len(iterations) + 1))
    assert 1 <= len(iterations) <= 5
    result = events[-1]
    assert (result.source, result.event) == ("get_dpm_mixture", "result")
    assert result.metrics["num_components"] == len(model.components)


def test_htsne_quiet_and_events(capsys: pytest.CaptureFixture[str]) -> None:
    events: list[ProgressEvent] = []
    htsne(
        DATA,
        max_components=3,
        max_its=6,
        print_iter=2,
        seed=4,
        verbose=False,
        progress_callback=events.append,
    )

    assert capsys.readouterr() == ("", "")
    embed = _drop_start([e for e in events if e.source == "htsne"])
    assert [e.iteration for e in embed] == [1, 2, 3, 4, 5, 6]
    assert all(np.isfinite(e.metrics["kl_divergence"]) for e in embed)


def test_dpmsne_default_output_and_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    rng = RandomState(5)
    affinities = rng.rand(12, 12)
    affinities = affinities + affinities.T

    dpmsne(affinities.copy(), max_its=4, print_iter=2, seed=6)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == ["Iteration 2", "Iteration 4"]

    events: list[ProgressEvent] = []
    dpmsne(
        affinities.copy(),
        max_its=4,
        print_iter=2,
        seed=6,
        verbose=False,
        progress_callback=events.append,
    )
    assert capsys.readouterr() == ("", "")
    assert [e.iteration for e in _drop_start(events)] == [1, 2, 3, 4]


def test_humap_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    humap = pytest.importorskip("dmx.utils.humap").humap
    events: list[ProgressEvent] = []
    humap(
        DATA,
        max_components=3,
        max_its=5,
        seed=7,
        umap_kwargs={"n_neighbors": 5, "random_state": 0, "n_jobs": 1},
        verbose=False,
        progress_callback=events.append,
    )

    assert capsys.readouterr() == ("", "")
    assert events[-1].source == "get_dpm_mixture"


_SUBPROCESS_SCRIPT = """
import numpy as np
from numpy.random import RandomState
from dmx.stats import GaussianEstimator, MixtureEstimator
from dmx.utils.automatic import get_dpm_mixture
from dmx.utils.estimation import optimize
data = [float(x) for x in RandomState(0).normal(0.0, 1.0, 40)]
est = MixtureEstimator([GaussianEstimator()] * 2)
optimize(data, est, max_its=3, rng=RandomState(1), verbose={verbose})
get_dpm_mixture(data, max_comp=3, rng=RandomState(3), max_its=3, print_iter=1,
                verbose={verbose})
"""


@pytest.mark.parametrize("verbose", [False, True])
def test_default_streams_in_fresh_interpreter(verbose: bool) -> None:
    # The ``out=sys.stdout`` defaults bind at import time, so capsys cannot see
    # them; a fresh interpreter checks the real standard output and error.
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT.format(verbose=verbose)],
        capture_output=True,
        text=True,
        check=True,
    )

    if verbose:
        assert "Iteration 1: ln[p_mat(Data|Model)]=" in result.stdout
        assert "# Components = " in result.stdout
    else:
        assert result.stdout == ""
        assert result.stderr == ""
