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


# A 5-dimensional mean and a well-conditioned covariance with correlations.
_MVN_5D = (
    [0.0, 1.0, -1.0, 2.0, 0.5],
    (0.5 * np.eye(5) + 0.5 * np.full((5, 5), 0.3) + np.diag([0.5, 1.0, 1.5, 2.0, 2.5]))
    .round(6)
    .tolist(),
)


def _lengths() -> IntegerCategoricalDistribution:
    """Return a length distribution on 1..3, so no sequence is empty."""
    return IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3])


STATS_CASES = [
    # Phase B: continuous scalar leaves.
    FisherCase("gaussian", lambda: GaussianDistribution(1.0, 4.0), CONTINUOUS),
    FisherCase(
        "gaussian_small_variance",
        lambda: GaussianDistribution(-3.0, 1.0e-4),
        CONTINUOUS,
    ),
    FisherCase("log_gaussian", lambda: LogGaussianDistribution(0.5, 0.25), CONTINUOUS),
    FisherCase("exponential", lambda: ExponentialDistribution(2.0), CONTINUOUS),
    FisherCase(
        "exponential_small_mean", lambda: ExponentialDistribution(0.05), CONTINUOUS
    ),
    FisherCase("gamma_small_shape", lambda: GammaDistribution(0.7, 2.0), CONTINUOUS),
    FisherCase("gamma_large_shape", lambda: GammaDistribution(20.0, 0.5), CONTINUOUS),
    FisherCase("gamma_unit_shape", lambda: GammaDistribution(1.0, 3.0), CONTINUOUS),
    # Large enough to use the series for trigamma(k) - 1/k.
    FisherCase(
        "gamma_huge_shape", lambda: GammaDistribution(1.0e4, 1.0e-2), CONTINUOUS
    ),
    # Phase B: count leaves.
    FisherCase("poisson", lambda: PoissonDistribution(3.5), COUNT),
    FisherCase("poisson_small_rate", lambda: PoissonDistribution(0.8), COUNT),
    # Near-degenerate: about 50 nonzero counts in 1000 draws.
    FisherCase("poisson_tiny_rate", lambda: PoissonDistribution(0.05), COUNT),
    FisherCase("geometric", lambda: GeometricDistribution(0.3), COUNT),
    FisherCase("geometric_near_one", lambda: GeometricDistribution(0.95), COUNT),
    FisherCase(
        "binomial_shifted", lambda: BinomialDistribution(0.3, 10, min_val=2), COUNT
    ),
    FisherCase("binomial_rare", lambda: BinomialDistribution(0.02, 20), COUNT),
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
    FisherCase(
        "integer_categorical_sparse",
        lambda: IntegerCategoricalDistribution(3, [0.2, 0.5, 0.3]),
        CATEGORICAL,
        options=SPARSE_CATEGORIES,
        rank_deficit=1,
        centered=False,
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
    # Dimensions 1 and 5 for each multivariate family.
    FisherCase(
        "mvn_1d",
        lambda: MultivariateGaussianDistribution([2.0], [[0.5]]),
        MULTIVARIATE,
    ),
    FisherCase(
        "mvn_5d", lambda: MultivariateGaussianDistribution(*_MVN_5D), MULTIVARIATE
    ),
    FisherCase(
        "mvn_5d_location_only",
        lambda: MultivariateGaussianDistribution(*_MVN_5D),
        MULTIVARIATE,
        options=LOCATION_ONLY,
    ),
    FisherCase(
        "diagonal_gaussian_1d",
        lambda: DiagonalGaussianDistribution([-1.0], [3.0]),
        MULTIVARIATE,
    ),
    FisherCase(
        "diagonal_gaussian_5d",
        lambda: DiagonalGaussianDistribution(
            [0.0, 1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 2.0, 4.0, 8.0]
        ),
        MULTIVARIATE,
    ),
    FisherCase("dirichlet_2d", lambda: DirichletDistribution([1.5, 4.0]), MULTIVARIATE),
    FisherCase(
        "dirichlet_5d",
        lambda: DirichletDistribution([1.0, 2.0, 3.0, 4.0, 5.0]),
        MULTIVARIATE,
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
    FisherCase(
        "composite_nested",
        lambda: CompositeDistribution(
            (
                ExponentialDistribution(1.5, name="wait"),
                CompositeDistribution(
                    (
                        PoissonDistribution(3.0, name="count"),
                        IntegerCategoricalDistribution(
                            0, [0.3, 0.3, 0.4], name="level"
                        ),
                    ),
                    name="inner",
                ),
            )
        ),
        COMBINATORS,
        rank_deficit=1,
    ),
    FisherCase(
        "composite_sparse_field",
        lambda: CompositeDistribution(
            (
                GaussianDistribution(0.0, 2.0, name="x"),
                CategoricalDistribution(dict(ABC), name="color"),
            )
        ),
        COMBINATORS,
        options=SPARSE_CATEGORIES,
        rank_deficit=1,
        centered=False,
    ),
    FisherCase(
        "optional_unmodeled_missing",
        lambda: OptionalDistribution(PoissonDistribution(2.5)),
        COMBINATORS,
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
    # Poisson lengths include empty sequences: under conditional scaling their
    # zero element block makes that block's covariance (1 - P(n = 0)) I, here
    # within tolerance; marginal scaling is exactly whitened.
    FisherCase(
        "sequence_poisson_conditional",
        lambda: SequenceDistribution(
            PoissonDistribution(1.5), len_dist=PoissonDistribution(4.0)
        ),
        VARIABLE_LENGTH,
    ),
    FisherCase(
        "sequence_poisson_marginal",
        lambda: SequenceDistribution(
            PoissonDistribution(1.5), len_dist=PoissonDistribution(4.0)
        ),
        VARIABLE_LENGTH,
        options=MARGINAL,
    ),
    FisherCase(
        "integer_multinomial_marginal",
        lambda: IntegerMultinomialDistribution(
            0, [0.2, 0.3, 0.5], len_dist=IntegerCategoricalDistribution(2, [0.5, 0.5])
        ),
        VARIABLE_LENGTH,
        options=MARGINAL,
        rank_deficit=2,
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
