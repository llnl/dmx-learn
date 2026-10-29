"""Tests for seed discipline: utils/seeding.py and the dmx.utils entry points."""

import io
from typing import Any, Callable

import numpy as np
import pytest
from numpy.random import RandomState

from dmx.stats import GaussianEstimator, MixtureDistribution, MixtureEstimator
from dmx.stats.gaussian import GaussianDistribution
from dmx.utils.automatic import get_dpm_mixture, prepare_mixture_model
from dmx.utils.estimation import best_of, iterate, optimize
from dmx.utils.htsne import dpmsne, htsne
from dmx.utils.progress import ProgressEvent
from dmx.utils.seeding import MAX_SEED, new_seed, resolve_rng, resolve_seed
from dmx.utils.vector import row_choice

DATA = [float(x) for x in RandomState(0).normal(0.0, 1.0, 30)] + [
    float(x) for x in RandomState(1).normal(6.0, 1.0, 30)
]


def _estimator() -> MixtureEstimator:
    """Return a two-component Gaussian mixture estimator."""
    return MixtureEstimator([GaussianEstimator()] * 2)


def _assert_global_state_untouched(action: Callable[[], Any]) -> None:
    """Run ``action`` and check NumPy's global random stream did not advance."""
    np.random.seed(123)
    action()
    assert np.random.random_sample() == RandomState(123).random_sample()


def _start_seed(events: list[ProgressEvent]) -> Any:
    """Return the seed from the first ``"start"`` event."""
    return next(e for e in events if e.event == "start").metrics["seed"]


def test_new_seed_range_and_global_state_untouched() -> None:
    seeds: set[int] = set()
    _assert_global_state_untouched(lambda: seeds.update(new_seed() for _ in range(20)))

    assert all(0 <= s < 2**31 for s in seeds)
    assert len(seeds) > 1


def test_resolve_seed_validation() -> None:
    assert resolve_seed(7) == 7
    assert resolve_seed(np.int64(MAX_SEED)) == MAX_SEED
    assert isinstance(resolve_seed(None), int)
    with pytest.raises(TypeError, match="integer"):
        resolve_seed(True)
    with pytest.raises(TypeError, match="integer"):
        resolve_seed(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be in"):
        resolve_seed(-1)
    with pytest.raises(ValueError, match="seed must be in"):
        resolve_seed(MAX_SEED + 1)


def test_resolve_rng() -> None:
    rng = RandomState(3)
    assert resolve_rng(None, rng) == (rng, None)

    built, seed = resolve_rng(11, None)
    assert seed == 11
    assert built.randint(1000) == RandomState(11).randint(1000)

    _, drawn = resolve_rng(None, None)
    assert isinstance(drawn, int)

    with pytest.raises(ValueError, match="not both"):
        resolve_rng(1, rng)


# Each fitter takes keyword arguments (seed, rng, progress_callback) and returns
# something whose ``str`` identifies the fitted result.
FITTERS: dict[str, Callable[..., Any]] = {
    "optimize": lambda **kw: optimize(
        DATA, _estimator(), max_its=10, verbose=False, **kw
    ),
    "best_of": lambda **kw: best_of(
        DATA,
        None,
        _estimator(),
        2,
        5,
        0.5,
        1.0e-9,
        kw.pop("rng", None),
        verbose=False,
        **kw,
    ),
    "iterate": lambda **kw: iterate(DATA, _estimator(), 5, verbose=False, **kw),
    "get_dpm_mixture": lambda **kw: get_dpm_mixture(
        DATA, max_comp=3, max_its=5, verbose=False, **kw
    ),
    "prepare_mixture_model": lambda **kw: prepare_mixture_model(
        DATA, kw.pop("rng", None), max_components=3, max_its=5, verbose=False, **kw
    )[0],
}


@pytest.mark.parametrize("name", sorted(FITTERS))
def test_same_seed_gives_identical_results(name: str) -> None:
    first = FITTERS[name](seed=17)
    second = FITTERS[name](seed=17)

    assert str(first) == str(second)


@pytest.mark.parametrize("name", sorted(FITTERS))
def test_reported_seed_reproduces_unseeded_run(name: str) -> None:
    events: list[ProgressEvent] = []
    first = FITTERS[name](progress_callback=events.append)
    seed = _start_seed(events)

    assert isinstance(seed, int)
    assert str(FITTERS[name](seed=seed)) == str(first)


@pytest.mark.parametrize("name", sorted(FITTERS))
def test_seed_matches_equivalent_rng(name: str) -> None:
    # Backward compatibility: seed=s is the same stream as rng=RandomState(s).
    assert str(FITTERS[name](seed=23)) == str(FITTERS[name](rng=RandomState(23)))


@pytest.mark.parametrize("name", sorted(FITTERS))
def test_seed_and_rng_together_raise(name: str) -> None:
    with pytest.raises(ValueError, match="not both"):
        FITTERS[name](seed=1, rng=RandomState(1))


def test_caller_rng_is_reported_as_none() -> None:
    events: list[ProgressEvent] = []
    FITTERS["optimize"](rng=RandomState(4), progress_callback=events.append)

    assert _start_seed(events) is None


def test_unseeded_fit_leaves_global_state_untouched() -> None:
    _assert_global_state_untouched(FITTERS["optimize"])
    _assert_global_state_untouched(FITTERS["get_dpm_mixture"])


def test_htsne_seed_reproduces() -> None:
    kwargs: dict[str, Any] = {"max_components": 3, "max_its": 5, "verbose": False}
    events: list[ProgressEvent] = []
    first = htsne(DATA, progress_callback=events.append, **kwargs)
    seed = _start_seed(events)

    np.testing.assert_array_equal(htsne(DATA, seed=seed, **kwargs), first)
    np.testing.assert_array_equal(
        htsne(DATA, seed=8, **kwargs), htsne(DATA, seed=8, **kwargs)
    )


def test_dpmsne_seed_reproduces() -> None:
    affinities = RandomState(5).rand(10, 10)
    affinities = affinities + affinities.T
    events: list[ProgressEvent] = []
    first = dpmsne(
        affinities.copy(), max_its=5, verbose=False, progress_callback=events.append
    )
    seed = _start_seed(events)

    again = dpmsne(affinities.copy(), max_its=5, verbose=False, seed=seed)
    np.testing.assert_array_equal(again, first)


def test_humap_seed_reproduces() -> None:
    humap = pytest.importorskip("dmx.utils.humap").humap
    umap_kwargs = {"n_neighbors": 5, "random_state": 0, "n_jobs": 1}
    events: list[ProgressEvent] = []
    first = humap(
        DATA,
        max_components=3,
        max_its=5,
        umap_kwargs=dict(umap_kwargs),
        verbose=False,
        progress_callback=events.append,
    )
    seed = _start_seed(events)
    again = humap(
        DATA,
        max_components=3,
        max_its=5,
        seed=seed,
        umap_kwargs=dict(umap_kwargs),
        verbose=False,
    )

    np.testing.assert_array_equal(again[3], first[3])
    np.testing.assert_array_equal(again[0], first[0])


def test_unseeded_sampler_records_reproducible_seed() -> None:
    dist = MixtureDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0)], [0.5, 0.5]
    )
    sampler = dist.sampler()

    assert isinstance(sampler.seed, int)
    np.testing.assert_array_equal(
        sampler.sample(25), dist.sampler(seed=sampler.seed).sample(25)
    )
    assert dist.sampler(seed=4).seed == 4


def test_row_choice_without_rng_leaves_global_state_untouched() -> None:
    picks: list[Any] = []
    probs = np.array([[0.2, 0.8], [1.0, 0.0]])
    _assert_global_state_untouched(lambda: picks.append(row_choice(probs, None)))

    assert picks[0].shape == (2,)
    assert picks[0][1] == 0


def test_fitters_write_nothing_extra_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The start event must not add text to the default output.
    out = io.StringIO()
    optimize(DATA, _estimator(), max_its=2, seed=1, out=out)

    assert "seed" not in out.getvalue()
    assert capsys.readouterr() == ("", "")
