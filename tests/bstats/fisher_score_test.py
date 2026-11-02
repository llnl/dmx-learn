"""Fisher score contract for the dmx.bstats families used by the automatic DPM.

``humap`` fits a bstats Dirichlet-process mixture by default, so these
families need scores for the Fisher representation to work on its default
path (llnl/dmx-learn#287). Scores use plug-in parameters. Cases are skipped
until implemented and share the checks in ``tests/fisher_contract.py`` with
``tests/stats/fisher_score_test.py``.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from dmx import bstats, stats
from dmx.bstats import (
    BernoulliDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IgnoredDistribution,
    NullDistribution,
    OptionalDistribution,
    PoissonDistribution,
    SequenceDistribution,
)
from dmx.utils.automatic import get_dpm_mixture
from dmx.utils.fisher import FisherScoreOptions, has_fisher_score
from tests.fisher_contract import (
    FisherCase,
    case_seed,
    dense_scores,
    encode,
    run_fisher_contract,
    sample_encoded,
)

BSTATS = 287

ABC = {"a": 0.5, "b": 0.3, "c": 0.2}
# Lengths 1..3 keep every sequence non-empty.
LENGTHS = {1: 0.3, 2: 0.4, 3: 0.3}

BSTATS_CASES = [
    FisherCase("gaussian", lambda: GaussianDistribution(1.0, 4.0), BSTATS),
    FisherCase("exponential", lambda: ExponentialDistribution(2.0), BSTATS),
    FisherCase("gamma", lambda: GammaDistribution(2.0, 1.5), BSTATS),
    FisherCase("poisson", lambda: PoissonDistribution(3.5), BSTATS),
    FisherCase("geometric", lambda: GeometricDistribution(0.3), BSTATS),
    FisherCase("bernoulli", lambda: BernoulliDistribution(0.3), BSTATS),
    FisherCase(
        "categorical",
        lambda: CategoricalDistribution(dict(ABC)),
        BSTATS,
        rank_deficit=1,
    ),
    FisherCase(
        "optional",
        lambda: OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25),
        BSTATS,
    ),
    FisherCase(
        "ignored",
        lambda: IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
        BSTATS,
    ),
    FisherCase(
        "composite_mixed",
        lambda: CompositeDistribution(
            [
                GaussianDistribution(0.0, 1.0, name="height"),
                CategoricalDistribution(dict(ABC), name="color"),
                OptionalDistribution(PoissonDistribution(2.0), p=0.25, name="visits"),
                IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
            ]
        ),
        BSTATS,
        rank_deficit=1,
    ),
    FisherCase(
        "sequence_conditional",
        lambda: SequenceDistribution(
            PoissonDistribution(2.0), len_dist=CategoricalDistribution(dict(LENGTHS))
        ),
        BSTATS,
        rank_deficit=1,
    ),
    FisherCase(
        "sequence_marginal",
        lambda: SequenceDistribution(
            PoissonDistribution(2.0), len_dist=CategoricalDistribution(dict(LENGTHS))
        ),
        BSTATS,
        options=FisherScoreOptions(sequence_scaling="marginal"),
        rank_deficit=1,
    ),
    FisherCase("null", NullDistribution, BSTATS),
]


@dataclass(frozen=True)
class ParityCase:
    """Equivalent bstats and stats models with shared observations."""

    case_id: str
    make_bstats: Callable[[], Any]
    make_stats: Callable[[], Any]
    data: list[Any]
    options: Optional[FisherScoreOptions] = None


PARITY_CASES = [
    ParityCase(
        "gaussian",
        lambda: bstats.GaussianDistribution(1.0, 4.0),
        lambda: stats.GaussianDistribution(1.0, 4.0),
        [-1.0, 1.0, 3.0],
    ),
    ParityCase(
        "categorical_unseen",
        lambda: bstats.CategoricalDistribution(dict(ABC), default_value=0.1),
        lambda: stats.CategoricalDistribution(dict(ABC), default_value=0.1),
        ["a", "unseen", "c", "b"],
    ),
    ParityCase(
        "categorical_sparse",
        lambda: bstats.CategoricalDistribution(dict(ABC)),
        lambda: stats.CategoricalDistribution(dict(ABC)),
        ["a", "unseen", "c", "b"],
        FisherScoreOptions(dense_category_limit=2),
    ),
    ParityCase(
        "poisson",
        lambda: bstats.PoissonDistribution(3.5),
        lambda: stats.PoissonDistribution(3.5),
        [0, 2, 5],
    ),
    ParityCase(
        "bernoulli",
        lambda: bstats.BernoulliDistribution(0.3),
        lambda: stats.BinomialDistribution(0.3, 1),
        [False, True, True, False],
    ),
    ParityCase(
        "exponential",
        lambda: bstats.ExponentialDistribution(0.5),
        lambda: stats.ExponentialDistribution(2.0),
        [0.5, 1.0, 3.0],
    ),
    ParityCase(
        "geometric",
        lambda: bstats.GeometricDistribution(0.3),
        lambda: stats.GeometricDistribution(0.3),
        [1, 2, 5],
    ),
    ParityCase(
        "gamma",
        lambda: bstats.GammaDistribution(2.0, 1.5),
        lambda: stats.GammaDistribution(2.0, 1.5),
        [0.5, 2.0, 5.0],
    ),
    ParityCase(
        "optional",
        lambda: bstats.OptionalDistribution(
            bstats.GaussianDistribution(0.0, 1.0), p=0.25
        ),
        lambda: stats.OptionalDistribution(
            stats.GaussianDistribution(0.0, 1.0), p=0.25
        ),
        [None, -1.0, 1.0, None],
    ),
    ParityCase(
        "ignored",
        lambda: bstats.IgnoredDistribution(bstats.GaussianDistribution(0.0, 1.0)),
        lambda: stats.IgnoredDistribution(stats.GaussianDistribution(0.0, 1.0)),
        [-1.0, 0.0, 1.0],
    ),
    ParityCase(
        "composite",
        lambda: bstats.CompositeDistribution(
            (
                bstats.GaussianDistribution(0.0, 1.0),
                bstats.CategoricalDistribution(dict(ABC)),
            )
        ),
        lambda: stats.CompositeDistribution(
            (
                stats.GaussianDistribution(0.0, 1.0),
                stats.CategoricalDistribution(dict(ABC)),
            )
        ),
        [(-1.0, "a"), (0.0, "b"), (1.0, "c")],
    ),
    ParityCase(
        "sequence",
        lambda: bstats.SequenceDistribution(
            bstats.PoissonDistribution(2.0),
            len_dist=bstats.CategoricalDistribution(dict(LENGTHS)),
        ),
        lambda: stats.SequenceDistribution(
            stats.PoissonDistribution(2.0),
            len_dist=stats.CategoricalDistribution(dict(LENGTHS)),
        ),
        [[0], [1, 2], [0, 2, 4]],
    ),
    ParityCase("null", bstats.NullDistribution, stats.NullDistribution, []),
]


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.case_id)
def test_stats_bstats_score_parity(case: ParityCase) -> None:
    """Return identical scores for equivalent plug-in model parameters."""
    bstats_dist = case.make_bstats()
    stats_dist = case.make_stats()

    bstats_scores = dense_scores(
        bstats_dist, encode(bstats_dist, case.data), case.options
    )
    stats_scores = dense_scores(stats_dist, encode(stats_dist, case.data), case.options)

    np.testing.assert_allclose(bstats_scores, stats_scores, rtol=0.0, atol=1.0e-12)


def test_automatic_dpm_components_have_fisher_scores() -> None:
    """Support every retained component fitted to heterogeneous example data."""
    path = Path("tests/data/testInput_automatic1.pkl")
    data = pickle.loads(path.read_bytes())[:10]
    mixture = get_dpm_mixture(
        data,
        max_comp=2,
        rng=np.random.RandomState(1),
        max_its=1,
        print_iter=2,
        mix_threshold_count=0.0,
        verbose=False,
    )

    assert has_fisher_score(mixture)


def test_bstats_mixture_fisher_score_is_finite() -> None:
    """Exercise the shared nested-mixture score implementation."""
    mixture = bstats.MixtureDistribution(
        (
            bstats.GaussianDistribution(-1.0, 1.0),
            bstats.GaussianDistribution(1.0, 1.0),
        ),
        [0.4, 0.6],
    )
    options = FisherScoreOptions(whitening_samples=128)
    encoded = mixture.seq_encode([-2.0, 0.0, 2.0])

    scores = np.asarray(mixture.seq_fisher_score(encoded, options))

    assert scores.shape == (3, mixture.fisher_dim(options))
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("case", BSTATS_CASES, ids=lambda case: case.case_id)
def test_fisher_score_contract(case: FisherCase) -> None:
    run_fisher_contract(case)


@pytest.mark.parametrize("case", BSTATS_CASES, ids=lambda case: case.case_id)
def test_case_samples_and_encodes(case: FisherCase) -> None:
    """Keep every registry entry valid before its family is implemented."""
    dist = case.make()
    encoded = sample_encoded(dist, case.n, case_seed(case.case_id))
    log_density = np.asarray(dist.seq_log_density(encoded))
    assert log_density.shape == (case.n,)
    assert np.all(np.isfinite(log_density))


def test_case_ids_are_unique() -> None:
    ids = [case.case_id for case in BSTATS_CASES]
    assert len(ids) == len(set(ids))
