"""Fisher score contract for the dmx.bstats families used by the automatic DPM.

``humap`` fits a bstats Dirichlet-process mixture by default, so these
families need scores for the Fisher representation to work on its default
path (llnl/dmx-learn#287). Scores use plug-in parameters. Cases are skipped
until implemented and share the checks in ``tests/fisher_contract.py`` with
``tests/stats/fisher_score_test.py``.
"""

import numpy as np
import pytest

from dmx.bstats import (
    BernoulliDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IgnoredDistribution,
    OptionalDistribution,
    PoissonDistribution,
    SequenceDistribution,
)
from dmx.utils.fisher import FisherScoreOptions
from tests.fisher_contract import (
    FisherCase,
    case_seed,
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
]


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
