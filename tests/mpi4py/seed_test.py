"""Single-process tests for seed discipline in the MPI helpers.

The ``mpi_utils`` fixture in ``conftest.py`` imports the helpers against size-1
MPI fakes, so these run under plain ``pytest``. The broadcast of a drawn seed
from rank 0 is checked directly on :func:`dmx.mpi4py.utils.resolve_rng_mpi`
with a fake worker-rank communicator.
"""

from typing import Any

import numpy as np
import pytest
from numpy.random import RandomState

from dmx.bstats.gaussian import GaussianEstimator as BayesianGaussianEstimator
from dmx.mpi4py.utils import resolve_rng_mpi
from dmx.stats import GaussianEstimator, MixtureEstimator
from dmx.utils.progress import ProgressEvent

DATA = list(np.random.RandomState(3).normal(0.0, 1.0, 60))


def _mixture_estimator() -> MixtureEstimator:
    """Return a two-component Gaussian mixture estimator."""
    return MixtureEstimator([GaussianEstimator()] * 2)


def _optimize(mpi_utils: dict[str, Any], **kwargs: Any) -> Any:
    """Run the dmx.stats ``optimize_mpi`` quietly on ``DATA``."""
    return mpi_utils["estimation"].optimize_mpi(
        DATA, _mixture_estimator(), max_its=5, verbose=False, **kwargs
    )


def _bstats_optimize(mpi_utils: dict[str, Any], **kwargs: Any) -> Any:
    """Run the dmx.bstats ``optimize_mpi`` quietly on ``DATA``."""
    return mpi_utils["bestimation"].optimize_mpi(
        DATA, BayesianGaussianEstimator(), max_its=3, verbose=False, **kwargs
    )


def _best_of(mpi_utils: dict[str, Any], rng: Any = None, **kwargs: Any) -> Any:
    """Run ``best_of_mpi`` quietly on ``DATA``."""
    return mpi_utils["estimation"].best_of_mpi(
        DATA,
        None,
        _mixture_estimator(),
        trials=2,
        max_its=3,
        max_its_cnt=2,
        init_p=0.5,
        delta=1.0e-9,
        rng=rng,
        verbose=False,
        **kwargs,
    )


def _dpm(mpi_utils: dict[str, Any], **kwargs: Any) -> Any:
    """Run ``get_dpm_mixture_mpi`` quietly on ``DATA``."""
    return mpi_utils["automatic"].get_dpm_mixture_mpi(
        DATA,
        estimator=BayesianGaussianEstimator(),
        max_comp=3,
        max_its=3,
        verbose=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "runner", [_optimize, _bstats_optimize, _best_of, _dpm], ids=lambda f: f.__name__
)
def test_same_seed_gives_identical_results(
    mpi_utils: dict[str, Any], runner: Any
) -> None:
    assert str(runner(mpi_utils, seed=11)) == str(runner(mpi_utils, seed=11))


@pytest.mark.parametrize(
    "runner", [_optimize, _bstats_optimize, _best_of, _dpm], ids=lambda f: f.__name__
)
def test_reported_seed_reproduces_run(mpi_utils: dict[str, Any], runner: Any) -> None:
    events: list[ProgressEvent] = []
    first = runner(mpi_utils, progress_callback=events.append)

    seed = events[0].metrics["seed"]
    assert events[0].event == "start"
    assert isinstance(seed, int)
    assert str(runner(mpi_utils, seed=seed)) == str(first)


def test_given_rng_matches_equal_seed(mpi_utils: dict[str, Any]) -> None:
    # A caller-owned rng is used unchanged, so it reproduces seed-based runs.
    assert str(_optimize(mpi_utils, rng=RandomState(7))) == str(
        _optimize(mpi_utils, seed=7)
    )


@pytest.mark.parametrize(
    "runner", [_optimize, _bstats_optimize, _best_of, _dpm], ids=lambda f: f.__name__
)
def test_seed_and_rng_together_raise(mpi_utils: dict[str, Any], runner: Any) -> None:
    with pytest.raises(ValueError, match="seed or rng"):
        runner(mpi_utils, seed=1, rng=RandomState(1))


def _humap(mpi_utils: dict[str, Any], **kwargs: Any) -> Any:
    """Run ``humap_mpi`` quietly with a fixed UMAP random state."""
    return mpi_utils["humap"].humap_mpi(
        DATA,
        max_components=3,
        max_its=3,
        umap_kwargs={"n_neighbors": 5, "random_state": 0, "n_jobs": 1},
        verbose=False,
        **kwargs,
    )


def test_humap_mpi_seed_reproduces(mpi_utils: dict[str, Any]) -> None:
    events: list[ProgressEvent] = []
    first = _humap(mpi_utils, progress_callback=events.append)

    start = events[0]
    assert (start.source, start.event) == ("humap_mpi", "start")
    second = _humap(mpi_utils, seed=start.metrics["seed"])
    assert str(first[1]) == str(second[1])
    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[0], second[0])


class _WorkerComm:
    """Fake rank-1 communicator whose broadcast delivers rank 0's seed."""

    def __init__(self, root_seed: int) -> None:
        self.root_seed = root_seed
        self.received: list[Any] = []

    def Get_rank(self) -> int:  # pylint: disable=invalid-name
        """Return rank 1."""
        return 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        """Record the worker's contribution and return rank 0's seed."""
        assert root == 0
        self.received.append(value)
        return self.root_seed


def test_worker_rank_uses_broadcast_seed() -> None:
    comm = _WorkerComm(root_seed=1234)
    rng, seed = resolve_rng_mpi(comm)

    assert comm.received == [None]
    assert seed == 1234
    assert rng.randint(1000) == RandomState(1234).randint(1000)


def test_given_seed_or_rng_skip_broadcast() -> None:
    comm = _WorkerComm(root_seed=1234)
    _, seed = resolve_rng_mpi(comm, seed=9)
    own = RandomState(3)
    rng, no_seed = resolve_rng_mpi(comm, rng=own)

    assert (seed, no_seed, comm.received) == (9, None, [])
    assert rng is own
