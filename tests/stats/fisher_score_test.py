"""Fisher score contract for every dmx.stats family in the humap-fisher plan.

Each :class:`~tests.fisher_contract.FisherCase` names the GitHub issue that
implements its family. Unimplemented families are skipped with that issue, so
this file tracks progress through roadmap phases B (leaves), C (observed
combinators), and D (latent-variable components). Implementations make their
cases pass without editing the shared checks in ``tests/fisher_contract.py``.

Per-family hand-computed value tests live with each family's implementation,
not here.
"""

import numpy as np
import pytest

from dmx.stats import (
    BernoulliSetDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    DiagonalGaussianDistribution,
    DirichletDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    HeterogeneousMixtureDistribution,
    HiddenMarkovModelDistribution,
    IgnoredDistribution,
    IntegerCategoricalDistribution,
    IntegerMultinomialDistribution,
    LogGaussianDistribution,
    MixtureDistribution,
    MultinomialDistribution,
    MultivariateGaussianDistribution,
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

# GitHub issues (llnl/dmx-learn) implementing each group of families.
CONTINUOUS = 279
COUNT = 280
CATEGORICAL = 281
MULTIVARIATE = 282
COMBINATORS = 283
VARIABLE_LENGTH = 284
NESTED_MIXTURES = 285
HMM = 286

LOCATION_ONLY = FisherScoreOptions(location_only=True)
MARGINAL = FisherScoreOptions(sequence_scaling="marginal")
# Forces the sparse categorical and set forms on small supports.
SPARSE_CATEGORIES = FisherScoreOptions(dense_category_limit=2)

ABC = {"a": 0.5, "b": 0.3, "c": 0.2}


def _lengths() -> IntegerCategoricalDistribution:
    """Return a length distribution on 1..3, so no sequence is empty."""
    return IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3])


STATS_CASES = [
    # Phase B: continuous scalar leaves.
    FisherCase("gaussian", lambda: GaussianDistribution(1.0, 4.0), CONTINUOUS),
    FisherCase("log_gaussian", lambda: LogGaussianDistribution(0.5, 0.25), CONTINUOUS),
    FisherCase("exponential", lambda: ExponentialDistribution(2.0), CONTINUOUS),
    FisherCase("gamma_small_shape", lambda: GammaDistribution(0.7, 2.0), CONTINUOUS),
    FisherCase("gamma_large_shape", lambda: GammaDistribution(20.0, 0.5), CONTINUOUS),
    # Phase B: count leaves.
    FisherCase("poisson", lambda: PoissonDistribution(3.5), COUNT),
    FisherCase("poisson_small_rate", lambda: PoissonDistribution(0.8), COUNT),
    FisherCase("geometric", lambda: GeometricDistribution(0.3), COUNT),
    FisherCase(
        "binomial_shifted", lambda: BinomialDistribution(0.3, 10, min_val=2), COUNT
    ),
    # Phase B: categorical leaves. A one-hot score has one linear constraint.
    FisherCase(
        "categorical",
        lambda: CategoricalDistribution(dict(ABC)),
        CATEGORICAL,
        rank_deficit=1,
    ),
    FisherCase(
        "categorical_sparse",
        lambda: CategoricalDistribution({"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}),
        CATEGORICAL,
        options=SPARSE_CATEGORIES,
        rank_deficit=1,
        centered=False,
    ),
    FisherCase(
        "integer_categorical",
        lambda: IntegerCategoricalDistribution(3, [0.2, 0.5, 0.3]),
        CATEGORICAL,
        rank_deficit=1,
    ),
    # Phase B: multivariate leaves.
    FisherCase(
        "mvn",
        lambda: MultivariateGaussianDistribution([0.0, 1.0], [[2.0, 0.5], [0.5, 1.0]]),
        MULTIVARIATE,
    ),
    FisherCase(
        "mvn_location_only",
        lambda: MultivariateGaussianDistribution([0.0, 1.0], [[2.0, 0.5], [0.5, 1.0]]),
        MULTIVARIATE,
        options=LOCATION_ONLY,
    ),
    FisherCase(
        "diagonal_gaussian",
        lambda: DiagonalGaussianDistribution([0.0, 1.0, -1.0], [1.0, 2.0, 0.5]),
        MULTIVARIATE,
    ),
    FisherCase(
        "diagonal_gaussian_location_only",
        lambda: DiagonalGaussianDistribution([0.0, 1.0, -1.0], [1.0, 2.0, 0.5]),
        MULTIVARIATE,
        options=LOCATION_ONLY,
    ),
    FisherCase(
        "dirichlet", lambda: DirichletDistribution([2.0, 3.0, 5.0]), MULTIVARIATE
    ),
    # Phase C: observed combinators.
    FisherCase(
        "optional",
        lambda: OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25),
        COMBINATORS,
    ),
    FisherCase(
        "ignored",
        lambda: IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
        COMBINATORS,
    ),
    FisherCase(
        "composite_mixed",
        lambda: CompositeDistribution(
            (
                GaussianDistribution(0.0, 1.0, name="height"),
                CategoricalDistribution(dict(ABC), name="color"),
                OptionalDistribution(PoissonDistribution(2.0), p=0.25, name="visits"),
                IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
            )
        ),
        COMBINATORS,
        rank_deficit=1,
    ),
    # Phase C: variable-length observed data.
    FisherCase(
        "sequence_conditional",
        lambda: SequenceDistribution(
            GaussianDistribution(0.0, 1.0), len_dist=_lengths()
        ),
        VARIABLE_LENGTH,
        rank_deficit=1,
    ),
    FisherCase(
        "sequence_marginal",
        lambda: SequenceDistribution(
            GaussianDistribution(0.0, 1.0), len_dist=_lengths()
        ),
        VARIABLE_LENGTH,
        options=MARGINAL,
        rank_deficit=1,
    ),
    FisherCase(
        "integer_multinomial",
        lambda: IntegerMultinomialDistribution(
            0, [0.2, 0.3, 0.5], len_dist=IntegerCategoricalDistribution(2, [0.5, 0.5])
        ),
        VARIABLE_LENGTH,
        rank_deficit=2,
    ),
    FisherCase(
        "multinomial",
        lambda: MultinomialDistribution(
            CategoricalDistribution(dict(ABC)),
            len_dist=IntegerCategoricalDistribution(2, [0.5, 0.5]),
        ),
        VARIABLE_LENGTH,
        rank_deficit=2,
    ),
    FisherCase(
        "bernoulli_set",
        lambda: BernoulliSetDistribution({"a": 0.3, "b": 0.6, "c": 0.15}),
        VARIABLE_LENGTH,
    ),
    FisherCase(
        "bernoulli_set_sparse",
        lambda: BernoulliSetDistribution({"a": 0.3, "b": 0.6, "c": 0.15}),
        VARIABLE_LENGTH,
        options=SPARSE_CATEGORIES,
        centered=False,
    ),
    # Phase D: latent-variable components. Their scores are whitened
    # empirically, so moments hold by construction and are not checked.
    FisherCase(
        "nested_mixture",
        lambda: MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.4, 0.6],
        ),
        NESTED_MIXTURES,
        check_moments=False,
    ),
    FisherCase(
        "heterogeneous_mixture",
        lambda: HeterogeneousMixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 2.0)],
            [0.5, 0.5],
        ),
        NESTED_MIXTURES,
        check_moments=False,
    ),
    FisherCase(
        "hmm",
        lambda: HiddenMarkovModelDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.9, 0.1], [0.2, 0.8]],
            len_dist=IntegerCategoricalDistribution(2, [0.5, 0.5]),
        ),
        HMM,
        check_moments=False,
    ),
]


@pytest.mark.parametrize("case", STATS_CASES, ids=lambda case: case.case_id)
def test_fisher_score_contract(case: FisherCase) -> None:
    run_fisher_contract(case)


@pytest.mark.parametrize("case", STATS_CASES, ids=lambda case: case.case_id)
def test_case_samples_and_encodes(case: FisherCase) -> None:
    """Keep every registry entry valid before its family is implemented."""
    dist = case.make()
    encoded = sample_encoded(dist, case.n, case_seed(case.case_id))
    log_density = np.asarray(dist.seq_log_density(encoded))
    assert log_density.shape == (case.n,)
    assert np.all(np.isfinite(log_density))


def test_case_ids_are_unique() -> None:
    ids = [case.case_id for case in STATS_CASES]
    assert len(ids) == len(set(ids))
