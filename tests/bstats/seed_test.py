"""Tests for seed discipline in dmx.bstats estimation helpers and samplers."""

import importlib
import inspect
import pkgutil
from typing import Any, Callable, Optional

import numpy as np
import pytest

import dmx.bstats
from dmx.bstats.bernoulli import BernoulliDistribution
from dmx.bstats.bestimation import best_of, iterate, optimize
from dmx.bstats.beta import BetaDistribution
from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.categorical import CategoricalDistribution
from dmx.bstats.composite import CompositeDistribution
from dmx.bstats.conditional import ConditionalDistribution
from dmx.bstats.dirac import DiracDistribution
from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.dmvn import DiagonalGaussianDistribution
from dmx.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
)
from dmx.bstats.exponential import ExponentialDistribution
from dmx.bstats.gamma import GammaDistribution
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.bstats.geometric import GeometricDistribution
from dmx.bstats.ignored import IgnoredDistribution
from dmx.bstats.intrange import IntegerCategoricalDistribution
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
from dmx.bstats.normgamma import NormalGammaDistribution
from dmx.bstats.nulldist import NullDistribution
from dmx.bstats.optional import OptionalDistribution
from dmx.bstats.pdist import ProbabilityDistribution
from dmx.bstats.poisson import PoissonDistribution
from dmx.bstats.sequence import SequenceDistribution
from dmx.bstats.setdist import BernoulliSetDistribution
from dmx.bstats.symdirichlet import SymmetricDirichletDistribution
from dmx.utils.progress import ProgressEvent

DATA = (-2.0, -1.8, -1.6, -0.2, 0.1, 1.7, 1.9, 2.1, 2.4, 3.0)


def _estimator() -> DirichletProcessMixtureEstimator:
    """Create a small homogeneous Gaussian DPM estimator."""
    return DirichletProcessMixtureEstimator(
        tuple(GaussianEstimator() for _ in range(3))
    )


# Each runner fits with the given seed/rng and returns a comparable result.
RUNNERS: dict[str, Callable[..., Any]] = {
    "optimize": lambda **kw: str(
        optimize(DATA, _estimator(), max_its=3, init_p=0.5, verbose=False, **kw)
    ),
    "best_of": lambda rng=None, **kw: best_of(
        DATA, DATA, _estimator(), 2, 3, 0.5, None, rng, verbose=False, **kw
    )[0],
    "iterate": lambda **kw: str(
        iterate(DATA, _estimator(), 3, init_p=0.5, verbose=False, **kw)
    ),
}


@pytest.mark.parametrize("name", list(RUNNERS))
def test_same_seed_gives_identical_results(name: str) -> None:
    assert RUNNERS[name](seed=11) == RUNNERS[name](seed=11)


@pytest.mark.parametrize("name", list(RUNNERS))
def test_seed_matches_equivalent_rng(name: str) -> None:
    # A given rng keeps today's behavior, so seed=s equals RandomState(s).
    assert RUNNERS[name](seed=12) == RUNNERS[name](rng=np.random.RandomState(12))


@pytest.mark.parametrize("name", list(RUNNERS))
def test_start_event_seed_reproduces_unseeded_run(name: str) -> None:
    events: list[ProgressEvent] = []
    first = RUNNERS[name](progress_callback=events.append)

    starts = [e for e in events if e.event == "start"]
    assert len(starts) == 1
    seed = starts[0].metrics["seed"]
    assert isinstance(seed, int)
    assert RUNNERS[name](seed=seed) == first


@pytest.mark.parametrize("name", list(RUNNERS))
def test_caller_rng_reports_no_seed(name: str) -> None:
    events: list[ProgressEvent] = []
    RUNNERS[name](rng=np.random.RandomState(3), progress_callback=events.append)

    assert [e.metrics["seed"] for e in events if e.event == "start"] == [None]


@pytest.mark.parametrize("name", list(RUNNERS))
def test_seed_and_rng_together_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="seed or rng"):
        RUNNERS[name](seed=1, rng=np.random.RandomState(1))


def _dpm() -> DirichletProcessMixtureDistribution:
    """Return a three-component Gaussian DPM."""
    k = 3
    return DirichletProcessMixtureDistribution(
        [GaussianDistribution(float(i), 1.0) for i in range(k)],
        [0.5, 0.3, 0.2],
        1.0,
        np.ones((k, 2)),
        [NormalGammaDistribution(0.0, 1.0, 2.0, 1.0) for _ in range(k)],
    )


DISTRIBUTIONS: dict[str, Callable[[], Any]] = {
    "gaussian": lambda: GaussianDistribution(0.5, 2.0),
    "categorical": lambda: CategoricalDistribution({"a": 0.2, "b": 0.8}),
    "poisson": lambda: PoissonDistribution(3.0),
    "beta": lambda: BetaDistribution(2.0, 3.0),
    "optional": lambda: OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.3),
    "sequence": lambda: SequenceDistribution(
        GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0)
    ),
    "mixture": lambda: MixtureDistribution(
        [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)],
        [0.4, 0.6],
    ),
    "dpm": _dpm,
    "bernoulli": lambda: BernoulliDistribution(0.3),
    "dict_dirichlet": lambda: DictDirichletDistribution({"a": 1.0, "b": 2.0}),
    "composite": lambda: CompositeDistribution(
        (GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0))
    ),
    "conditional": lambda: ConditionalDistribution(
        {"a": GaussianDistribution(0.0, 1.0), "b": ExponentialDistribution(1.0)},
        CategoricalDistribution({"a": 0.5, "b": 0.5}),
    ),
    "dirac": lambda: DiracDistribution(2.5),
    "dirichlet": lambda: DirichletDistribution([1.0, 2.0, 3.0]),
    "diagonal_gaussian": lambda: DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
    "exponential": lambda: ExponentialDistribution(2.0),
    "gamma": lambda: GammaDistribution(2.0, 1.5),
    "geometric": lambda: GeometricDistribution(0.4),
    "ignored": lambda: IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
    "integer_categorical": lambda: IntegerCategoricalDistribution(
        [0.2, 0.5, 0.3], min_index=-1
    ),
    "mvn_gamma": lambda: MultivariateNormalGammaDistribution(
        np.zeros(2), np.ones(2), np.ones(2) * 2.0, np.ones(2)
    ),
    "normal_gamma": lambda: NormalGammaDistribution(0.0, 1.0, 2.0, 1.0),
    "null": NullDistribution,
    "bernoulli_set": lambda: BernoulliSetDistribution({"x": 0.2, "y": 0.7}),
    "symmetric_dirichlet": lambda: SymmetricDirichletDistribution(2.0, ndim=3),
}


def _draw(dist: Any, seed: Optional[int]) -> tuple[Any, list[Any]]:
    """Return a sampler and 20 samples drawn from it."""
    sampler = dist.sampler(seed=seed)
    return sampler, [repr(x) for x in sampler.sample(20)]


@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_sampler_same_seed_gives_identical_samples(name: str) -> None:
    dist = DISTRIBUTIONS[name]()
    sampler, first = _draw(dist, 21)

    assert sampler.seed == 21
    assert _draw(dist, 21)[1] == first


@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_sampler_seed_reproduces_unseeded_samples(name: str) -> None:
    dist = DISTRIBUTIONS[name]()
    sampler, first = _draw(dist, None)

    assert isinstance(sampler.seed, int)
    assert _draw(dist, sampler.seed)[1] == first


def test_every_bstats_sampler_is_covered() -> None:
    covered = {type(factory()) for factory in DISTRIBUTIONS.values()}
    with_sampler = set()
    for module in pkgutil.iter_modules(dmx.bstats.__path__):
        mod = importlib.import_module(f"dmx.bstats.{module.name}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                cls.__module__ == mod.__name__
                and issubclass(cls, ProbabilityDistribution)
                and cls is not ProbabilityDistribution
                and "sampler" in cls.__dict__
            ):
                with_sampler.add(cls)

    assert with_sampler - covered == set()
