"""Tests for seed discipline in torch_stats samplers and torch_utils optimize."""

import importlib
import inspect
import logging
import pkgutil
import re
from typing import Any

import numpy as np
import pytest

import tests.torch_stats as torch_tests
from dmx.torch_stats import (
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    NullDistribution,
    PoissonDistribution,
)
from dmx.torch_utils.estimation import optimize
from tests.torch_stats.torch_stats_tests import (
    TorchStatsTestClass,
    get_test_torch_device,
)


def _dists() -> list[Any]:
    """Return torch distributions covering several sampler implementations."""
    return [
        GaussianDistribution(mu=0.0, sigma2=1.0),
        ExponentialDistribution(beta=2.0),
        PoissonDistribution(lam=3.0),
        GeometricDistribution(p=0.3),
        IntegerCategoricalDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5]),
        MixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [0.4, 0.6],
        ),
        CompositeDistribution(
            [GaussianDistribution(0.0, 1.0), PoissonDistribution(lam=2.0)]
        ),
    ]


def _as_array(x: Any) -> np.ndarray:
    """Convert sampler output (arrays, lists, tuples, tensors) for comparison."""
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x, dtype=object)


@pytest.mark.parametrize("index", range(len(_dists())))
def test_same_seed_gives_identical_samples(index: int) -> None:
    dist = _dists()[index]
    first = dist.sampler(seed=11).sample(25)
    second = dist.sampler(seed=11).sample(25)

    np.testing.assert_array_equal(_as_array(first), _as_array(second))


@pytest.mark.parametrize("index", range(len(_dists())))
def test_sampler_seed_attribute_reproduces_unseeded_draws(index: int) -> None:
    dist = _dists()[index]
    sampler = dist.sampler()
    draws = sampler.sample(25)

    assert isinstance(sampler.seed, int)
    replay = dist.sampler(seed=sampler.seed).sample(25)
    np.testing.assert_array_equal(_as_array(draws), _as_array(replay))


def test_explicit_seed_is_recorded() -> None:
    assert GaussianDistribution(0.0, 1.0).sampler(seed=42).seed == 42


def _fit_data() -> list[float]:
    """Return two well-separated Gaussian clusters."""
    rng = np.random.RandomState(0)
    return [float(x) for x in np.concatenate([rng.normal(m, 1.0, 60) for m in (-4, 4)])]


def _estimator() -> Any:
    """Return a two-component Gaussian mixture estimator."""
    return MixtureDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5]
    ).estimator()


def _fit(seed: Any) -> Any:
    """Fit the mixture with torch optimize and no progress output."""
    return optimize(
        _fit_data(),
        _estimator(),
        seed=seed,
        max_its=5,
        device=get_test_torch_device(),
        print_iter=1000,
    )


def test_optimize_same_seed_identical() -> None:
    assert str(_fit(7)) == str(_fit(7))


def test_optimize_logs_drawn_seed_that_reproduces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="dmx.torch_utils.estimation"):
        model = _fit(None)

    matches = [
        re.fullmatch(r"optimize seed=(\d+)", record.getMessage())
        for record in caplog.records
    ]
    seeds = [int(m.group(1)) for m in matches if m is not None]
    assert len(seeds) == 1
    assert str(_fit(seeds[0])) == str(model)


def test_optimize_never_touches_global_numpy_state() -> None:
    saved = np.random.get_state()
    try:
        np.random.seed(123)
        _fit(None)
        next_value = np.random.random_sample()
    finally:
        np.random.set_state(saved)

    # The global stream is exactly where seeding left it.
    assert next_value == np.random.RandomState(123).random_sample()


def _harness_sampler_dists() -> list[tuple[str, Any]]:
    """Return ``sampler_dist`` from every torch_stats harness test case."""
    rv = []
    for info in pkgutil.iter_modules(torch_tests.__path__):
        if not info.name.endswith("_test") or info.name == "seed_test":
            continue
        module = importlib.import_module(f"tests.torch_stats.{info.name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, TorchStatsTestClass)
                and cls is not TorchStatsTestClass
                and cls.__module__ == module.__name__
            ):
                case = cls()
                case.setUp()
                rv.append((name, case.sampler_dist))
    return rv


HARNESS_DISTS = _harness_sampler_dists()


@pytest.mark.parametrize(
    "dist", [d for _, d in HARNESS_DISTS], ids=[n for n, _ in HARNESS_DISTS]
)
def test_every_harness_sampler_replays_from_its_seed(dist: Any) -> None:
    sampler = dist.sampler()
    draws = sampler.sample(20)

    assert isinstance(sampler.seed, int)
    assert repr(dist.sampler(seed=sampler.seed).sample(20)) == repr(draws)


def test_null_sampler_records_resolved_seed() -> None:
    assert NullDistribution().sampler(seed=5).seed == 5
    assert isinstance(NullDistribution().sampler().seed, int)
