"""Hand-computed Fisher score values for dmx.stats scalar leaves.

The shared contract (``fisher_score_test.py``) checks moments on samples;
these tests pin exact rows computed independently from closed forms, so
errors too small for the moment checks still fail.
"""

import math
from typing import Any

import mpmath
import numpy as np
import pytest
import scipy.sparse

from dmx.stats import (
    BinomialDistribution,
    CategoricalDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    LogGaussianDistribution,
    PoissonDistribution,
)
from dmx.utils._fisher_kernels import bernoulli_score
from dmx.utils.fisher import FisherScoreOptions

SQRT_HALF = math.sqrt(0.5)
EULER_GAMMA = 0.5772156649015329


def _scores(dist: Any, data: list) -> np.ndarray:
    """Encode ``data`` and return the distribution's Fisher scores."""
    return np.asarray(dist.seq_fisher_score(dist.dist_to_encoder().seq_encode(data)))


def test_gaussian_values() -> None:
    dist = GaussianDistribution(1.0, 4.0)
    expected = [[0.0, -SQRT_HALF], [1.0, 0.0], [-1.0, 0.0]]
    np.testing.assert_allclose(_scores(dist, [1.0, 3.0, -1.0]), expected, atol=1e-15)


def test_log_gaussian_values_score_log_x() -> None:
    dist = LogGaussianDistribution(0.0, 1.0)
    data = [1.0, math.e, math.exp(-2.0)]
    expected = [[0.0, -SQRT_HALF], [1.0, 0.0], [-2.0, 3.0 * SQRT_HALF]]
    np.testing.assert_allclose(_scores(dist, data), expected, atol=1e-12)


def test_exponential_values() -> None:
    dist = ExponentialDistribution(2.0)
    np.testing.assert_allclose(
        _scores(dist, [0.5, 2.0, 5.0]), [[-0.75], [0.0], [1.5]], atol=1e-15
    )


def test_gamma_values() -> None:
    # k = 2: digamma(2) = 1 - Euler's gamma, trigamma(2) = pi^2 / 6 - 1.
    k, theta = 2.0, 3.0
    dist = GammaDistribution(k, theta)
    data = [6.0, 3.0, 10.0]
    x = np.asarray(data)
    s1 = (x - k * theta) / (math.sqrt(k) * theta)
    centered_log = np.log(x) - (1.0 - EULER_GAMMA) - math.log(theta)
    s2 = (centered_log - s1 / math.sqrt(k)) / math.sqrt(math.pi**2 / 6.0 - 1.5)
    np.testing.assert_allclose(
        _scores(dist, data), np.column_stack((s1, s2)), rtol=1e-12, atol=1e-14
    )


def test_gamma_unit_shape_first_column_matches_exponential() -> None:
    data = [0.1, 1.0, 2.5, 7.0]
    gamma = _scores(GammaDistribution(1.0, 2.5), data)
    exponential = _scores(ExponentialDistribution(2.5), data)
    np.testing.assert_allclose(gamma[:, :1], exponential, rtol=1e-14)


def test_gamma_huge_shape_matches_high_precision() -> None:
    # At x = k theta the first column is zero and the second is
    # (log k - digamma(k)) / sqrt(trigamma(k) - 1/k), close to sqrt(1/2).
    k, theta = 1.0e8, 1.0e-8
    scores = _scores(GammaDistribution(k, theta), [k * theta])
    with mpmath.workdps(40):
        k_mp = mpmath.mpf(k)
        second = (mpmath.log(k_mp) - mpmath.digamma(k_mp)) / mpmath.sqrt(
            mpmath.polygamma(1, k_mp) - 1 / k_mp
        )
    np.testing.assert_allclose(scores[0, 0], 0.0, atol=1e-12)
    # log(x) - digamma(k) - log(theta) cancels to ~1e-8 in double precision.
    np.testing.assert_allclose(scores[0, 1], float(second), rtol=1e-5)


def test_poisson_values() -> None:
    dist = PoissonDistribution(4.0)
    np.testing.assert_allclose(
        _scores(dist, [4, 6, 0]), [[0.0], [1.0], [-2.0]], atol=1e-15
    )


def test_geometric_values() -> None:
    dist = GeometricDistribution(0.5)
    expected = [[0.0], [-SQRT_HALF], [math.sqrt(2.0)]]
    np.testing.assert_allclose(_scores(dist, [2, 1, 4]), expected, atol=1e-15)


def test_binomial_values_shift_by_min_val() -> None:
    # n = 4, p = 1/2: mean 2 successes, standard deviation 1. Repeated values
    # check the mapping back from unique values to observations.
    dist = BinomialDistribution(0.5, 4, min_val=2)
    np.testing.assert_allclose(
        _scores(dist, [4, 2, 6, 4]), [[0.0], [-2.0], [2.0], [0.0]], atol=1e-15
    )


def test_bernoulli_kernel_values() -> None:
    np.testing.assert_allclose(
        bernoulli_score([0, 1, 1], 0.2), [[-0.5], [2.0], [2.0]], atol=1e-15
    )


def test_degenerate_count_parameters_are_clipped() -> None:
    floor = 1.0e-3
    options = FisherScoreOptions(prob_floor=floor)

    def scores(dist: Any, data: list) -> np.ndarray:
        encoded = dist.dist_to_encoder().seq_encode(data)
        return np.asarray(dist.seq_fisher_score(encoded, options))

    # Constructing these distributions takes log(0); only the scores below
    # must be warning-free.
    with np.errstate(divide="ignore"):
        tiny_rate = PoissonDistribution(1.0e-12)
        certain = GeometricDistribution(1.0)
        no_trials = BinomialDistribution(0.5, 0)

    # Rate below the floor: lam is raised to the floor.
    np.testing.assert_allclose(
        scores(tiny_rate, [0, 1]),
        [[-math.sqrt(floor)], [(1.0 - floor) / math.sqrt(floor)]],
        rtol=1e-12,
    )
    # p = 1: every count is 1 and p is clipped to 1 - floor.
    np.testing.assert_allclose(
        scores(certain, [1]), [[-floor / math.sqrt(floor)]], rtol=1e-9
    )
    # The binomial rejects p = 0 and p = 1, but p within the floor of either
    # end is clipped the same way.
    for p, clipped in ((1.0e-12, floor), (1.0 - 1.0e-12, 1.0 - floor)):
        expected = (np.asarray([0.0, 1.0, 3.0]) - 3.0 * clipped) / math.sqrt(
            3.0 * clipped * (1.0 - clipped)
        )
        np.testing.assert_allclose(
            scores(BinomialDistribution(p, 3), [0, 1, 3]),
            expected[:, None],
            rtol=1e-9,
        )
    # n = 0 carries no information and scores zero.
    np.testing.assert_array_equal(scores(no_trials, [0, 0]), [[0.0], [0.0]])


ABC = {"a": 0.5, "b": 0.25, "c": 0.25}


def test_categorical_values_and_pairwise_distances() -> None:
    dist = CategoricalDistribution(dict(ABC))
    scores = _scores(dist, ["a", "b", "c", "a"])
    expected_a = [1.0 / math.sqrt(0.5) - math.sqrt(0.5), -0.5, -0.5]
    expected_b = [-math.sqrt(0.5), 2.0 - 0.5, -0.5]
    np.testing.assert_allclose(scores[0], expected_a, atol=1e-12)
    np.testing.assert_allclose(scores[1], expected_b, atol=1e-12)
    np.testing.assert_array_equal(scores[0], scores[3])
    # Different levels a, b are 1/p_a + 1/p_b apart in squared distance.
    for i, j, expected in ((0, 1, 2.0 + 4.0), (0, 2, 2.0 + 4.0), (1, 2, 8.0)):
        squared = float(np.sum((scores[i] - scores[j]) ** 2))
        assert squared == pytest.approx(expected, rel=1e-9)


def test_categorical_default_value_adds_other_column() -> None:
    # Seen levels have mass 0.5 / 1.25 = 0.4; unseen values 0.25 / 1.25 = 0.2.
    dist = CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.25)
    assert dist.fisher_dim() == 3
    q = np.asarray([0.4, 0.4, 0.2])
    expected = np.tile(-np.sqrt(q), (3, 1))
    expected[0, 0] += 1.0 / math.sqrt(0.4)
    expected[1, 2] += 1.0 / math.sqrt(0.2)
    expected[2, 2] += 1.0 / math.sqrt(0.2)
    np.testing.assert_allclose(_scores(dist, ["a", "z", "y"]), expected, atol=1e-12)


def test_categorical_unseen_value_without_default_is_finite() -> None:
    # Zero probability under the distribution: the "no level" row -sqrt(q).
    dist = CategoricalDistribution(dict(ABC))
    np.testing.assert_allclose(
        _scores(dist, ["z"]), [[-math.sqrt(0.5), -0.5, -0.5]], atol=1e-12
    )


def test_categorical_zero_weight_level_is_floored() -> None:
    dist = CategoricalDistribution({"a": 1.0, "b": 0.0})
    scores = _scores(dist, ["a", "b"])
    assert np.all(np.isfinite(scores))
    # q_b is the floor (normalized), so the rare level is ~1/sqrt(1e-6) away.
    assert scores[1, 1] == pytest.approx(1.0e3, rel=1e-3)


def test_sparse_categorical_is_dense_without_offset() -> None:
    dist = CategoricalDistribution(dict(ABC))
    encoded = dist.dist_to_encoder().seq_encode(["a", "c", "z"])
    dense = np.asarray(dist.seq_fisher_score(encoded))
    sparse: Any = dist.seq_fisher_score(
        encoded, FisherScoreOptions(dense_category_limit=2)
    )
    assert scipy.sparse.isspmatrix_csr(sparse)
    assert sparse.dtype == np.float64
    assert sparse.shape == (3, 3)
    np.testing.assert_allclose(
        sparse.toarray() - dense, np.tile(np.sqrt([0.5, 0.25, 0.25]), (3, 1))
    )
    # The unseen value has no nonzero entry.
    assert sparse[2].nnz == 0


def test_integer_categorical_values_and_out_of_range() -> None:
    dist = IntegerCategoricalDistribution(3, [0.2, 0.5, 0.3])
    sqrt_p = np.sqrt([0.2, 0.5, 0.3])
    expected = np.tile(-sqrt_p, (4, 1))
    expected[0, 1] += 1.0 / math.sqrt(0.5)
    expected[1, 0] += 1.0 / math.sqrt(0.2)
    # Rows 2 and 3 (values 10 and 0) are outside [3, 5].
    np.testing.assert_allclose(_scores(dist, [4, 3, 10, 0]), expected, atol=1e-12)


DISTRIBUTIONS = [
    (GaussianDistribution(1.0, 4.0), [1.0, 3.0], 2),
    (LogGaussianDistribution(0.0, 1.0), [1.0, 2.0], 2),
    (ExponentialDistribution(2.0), [0.5, 2.0], 1),
    (GammaDistribution(2.0, 3.0), [6.0, 3.0], 2),
    (PoissonDistribution(4.0), [4, 6], 1),
    (GeometricDistribution(0.5), [2, 1], 1),
    (BinomialDistribution(0.5, 4, min_val=2), [4, 2], 1),
    (CategoricalDistribution({"a": 0.5, "b": 0.5}), ["a", "b"], 2),
    (IntegerCategoricalDistribution(3, [0.2, 0.5, 0.3]), [3, 4], 3),
]


@pytest.mark.parametrize(
    "dist, data, dim", DISTRIBUTIONS, ids=lambda v: type(v).__name__
)
def test_fisher_dim_and_location_only_is_ignored(
    dist: Any, data: list, dim: int
) -> None:
    location = FisherScoreOptions(location_only=True)
    encoded = dist.dist_to_encoder().seq_encode(data)
    assert dist.fisher_dim() == dim
    assert dist.fisher_dim(location) == dim
    np.testing.assert_array_equal(
        dist.seq_fisher_score(encoded), dist.seq_fisher_score(encoded, location)
    )
    assert dist.fisher_field_slices() == {type(dist).__name__: slice(0, dim)}


@pytest.mark.parametrize(
    "dist", [case[0] for case in DISTRIBUTIONS], ids=lambda v: type(v).__name__
)
def test_wrong_encoding_raises(dist: Any) -> None:
    wrong = (
        GaussianDistribution(0.0, 1.0)
        if not isinstance(dist, GaussianDistribution)
        else ExponentialDistribution(1.0)
    )
    encoded = wrong.dist_to_encoder().seq_encode([1.0, 2.0])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        dist.seq_fisher_score(encoded)
