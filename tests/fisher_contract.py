"""Shared contract checks for whitened Fisher scores.

Every distribution that implements ``seq_fisher_score`` must satisfy the same
contract (see :mod:`dmx.utils.fisher`): scores of shape
``(N, fisher_dim(options))`` in float64, finite, deterministic, field slices
that partition the columns, and, under the distribution itself, mean zero
and identity covariance on the score's support.

Package registries (``tests/stats/fisher_score_test.py`` and
``tests/bstats/fisher_score_test.py``) list one :class:`FisherCase` per
family and parameter setting and call :func:`run_fisher_contract`. A case
whose distribution has no score yet is skipped with the GitHub issue that
will implement it, so the registries track progress.

The moment checks compare sample moments with their expectations using
standard errors estimated from the same sample, so one tolerance rule works
for every family and sample size. With the default ``n = 1000`` they catch
mean offsets of about 0.2 and covariance errors of about 0.3 in whitened
units (for example a missing factor of ``sqrt(2)``). Smaller errors are the
job of the hand-computed value tests each family adds.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pytest
import scipy.sparse

from dmx.bstats.pdist import ProbabilityDistribution as BstatsProbabilityDistribution
from dmx.utils.fisher import FisherScoreOptions, has_fisher_score

N_SAMPLES = 1000
"""Default sample size per case; keeps the whole contract suite fast."""

Z_TOLERANCE = 5.0
"""Allowed deviation of a sample moment from its expectation, in standard errors."""

ABS_TOLERANCE = 1.0e-9
"""Absolute slack for moments that are exactly determined, such as constraints."""


@dataclass(frozen=True)
class FisherCase:
    """One distribution and setting checked against the Fisher score contract.

    Attributes:
        case_id: Readable pytest id, for example ``"gaussian"``.
        make: Factory returning a fresh distribution.
        issue: GitHub issue that implements the family's score, reported when
            the case is skipped.
        options: Options passed to every score call.
        rank_deficit: Expected ``fisher_dim - rank`` of the score covariance.
            Zero means identity covariance. Each over-complete block, such as
            a categorical one-hot whose whitened coordinates satisfy one
            linear constraint, adds one; the covariance is then a projector.
        centered: Whether the score has mean zero. Set False for encodings
            defined only up to a constant offset (for example a sparse
            offset-free categorical form); the covariance is then checked
            after centring and the mean is not checked.
        check_moments: Whether to check mean and covariance. Set False for
            families whose scores are whitened empirically after the fact,
            where the check would hold by construction.
        n: Number of samples to draw. Choose parameters so every level or
            event of the distribution is observed many times in ``n`` draws;
            the standard errors are estimated from the sample and are
            unreliable for events seen only a few times.
    """

    case_id: str
    make: Callable[[], Any]
    issue: int
    options: Optional[FisherScoreOptions] = None
    rank_deficit: int = 0
    centered: bool = True
    check_moments: bool = True
    n: int = N_SAMPLES


def case_seed(case_id: str) -> int:
    """Return a fixed sampling seed derived from a case id."""
    return zlib.crc32(case_id.encode("utf-8"))


def encode(dist: Any, data: Any) -> Any:
    """Encode samples the way the distribution's package expects."""
    if isinstance(dist, BstatsProbabilityDistribution):
        return dist.seq_encode(data)
    return dist.dist_to_encoder().seq_encode(data)


def sample_encoded(dist: Any, n: int, seed: int) -> Any:
    """Draw ``n`` samples from ``dist`` with a fixed seed and encode them."""
    return encode(dist, dist.sampler(seed=seed).sample(n))


def dense_scores(
    dist: Any, encoded: Any, options: Optional[FisherScoreOptions]
) -> np.ndarray:
    """Return ``dist.seq_fisher_score`` as a dense array.

    Sparse results are densified so every check sees the same type; their
    dtype is checked separately.
    """
    scores = dist.seq_fisher_score(encoded, options)
    if scipy.sparse.issparse(scores):
        assert scores.dtype == np.float64, f"sparse scores have dtype {scores.dtype}"
        return np.asarray(scores.toarray())
    return np.asarray(scores)


def check_score_array(scores: Any, n: int, dim: int) -> None:
    """Assert shape ``(n, dim)``, float64 dtype, and finite values."""
    assert isinstance(scores, np.ndarray), f"scores are {type(scores).__name__}"
    assert scores.shape == (n, dim), f"shape {scores.shape}, expected {(n, dim)}"
    assert scores.dtype == np.float64, f"dtype {scores.dtype}, expected float64"
    assert np.all(np.isfinite(scores)), "scores contain non-finite values"


def check_determinism(
    dist: Any, encoded: Any, options: Optional[FisherScoreOptions]
) -> None:
    """Assert two calls on the same input return identical scores."""
    first = dense_scores(dist, encoded, options)
    second = dense_scores(dist, encoded, options)
    np.testing.assert_array_equal(first, second, err_msg="scores are not deterministic")


def check_field_slices(
    dist: Any, options: Optional[FisherScoreOptions], dim: int
) -> None:
    """Assert ``fisher_field_slices`` covers each score column exactly once."""
    slices = dist.fisher_field_slices(options)
    assert isinstance(slices, dict) and slices, "field slices must be a non-empty dict"
    counts = np.zeros(dim, dtype=int)
    for name, columns in slices.items():
        assert isinstance(name, str) and name, f"bad field name {name!r}"
        assert isinstance(columns, slice), f"field {name!r} maps to {columns!r}"
        assert columns.step in (None, 1), f"field {name!r} has step {columns.step}"
        counts[columns] += 1
    assert np.all(counts == 1), f"columns covered {counts.tolist()} times"


def _within_tolerance(
    observed: np.ndarray, expected: np.ndarray, standard_error: np.ndarray
) -> np.ndarray:
    """Return a mask of entries within ``Z_TOLERANCE`` standard errors."""
    allowed = Z_TOLERANCE * standard_error + ABS_TOLERANCE
    return np.asarray(np.abs(observed - expected) <= allowed)


def check_moments(scores: np.ndarray, rank_deficit: int, centered: bool) -> None:
    """Assert mean zero and projector covariance of the requested rank.

    The second moment ``s^T s / n`` is compared with a projector: the
    identity when ``rank_deficit`` is zero, otherwise the projector onto the
    top ``dim - rank_deficit`` eigenvectors of the sample moment, which also
    requires the remaining eigenvalues to be close to zero. Each entry may deviate by
    ``Z_TOLERANCE`` standard errors estimated from the sample.

    Args:
        scores: Dense ``(n, dim)`` scores sampled from the distribution.
        rank_deficit: Expected ``dim - rank`` of the covariance.
        centered: Whether to require mean zero. When False the scores are
            centred first and only the covariance is checked.
    """
    n, dim = scores.shape
    if dim == 0:
        return
    sqrt_n = math.sqrt(n)

    mean = scores.mean(axis=0)
    if centered:
        mean_se = scores.std(axis=0, ddof=1) / sqrt_n
        ok = _within_tolerance(mean, np.zeros(dim), mean_se)
        assert np.all(ok), (
            f"score mean is not zero: columns {np.flatnonzero(~ok).tolist()} "
            f"have mean {mean[~ok].round(4).tolist()}"
        )
        values = scores
    else:
        values = scores - mean

    # Standard error of each entry of s^T s / n from E[(s_a s_b)^2], computed
    # with matrix products to avoid an (n, dim, dim) array.
    moment = values.T @ values / n
    squares = values * values
    fourth = squares.T @ squares / n
    variance = np.maximum(fourth - moment * moment, 0.0) * n / (n - 1)
    moment_se = np.sqrt(variance) / sqrt_n

    full_rank = dim - rank_deficit
    assert 0 <= full_rank <= dim, f"rank_deficit {rank_deficit} is outside [0, {dim}]"
    if full_rank == dim:
        target = np.eye(dim)
    else:
        _, vectors = np.linalg.eigh(moment)
        top = vectors[:, dim - full_rank :]
        target = top @ top.T

    ok = _within_tolerance(moment, target, moment_se)
    assert np.all(ok), (
        f"score covariance is not a rank-{full_rank} projector; worst entries "
        f"deviate by {np.abs(moment - target)[~ok].max():.4f}"
    )


def skip_if_unsupported(dist: Any, case: FisherCase) -> None:
    """Skip the current test if ``dist`` has no Fisher score yet."""
    if not has_fisher_score(dist, case.options):
        pytest.skip(
            f"{type(dist).__name__} has no Fisher score yet "
            f"(llnl/dmx-learn#{case.issue})"
        )


def run_fisher_contract(case: FisherCase) -> None:
    """Run every contract check for one case, skipping unsupported families."""
    dist = case.make()
    skip_if_unsupported(dist, case)

    dim = dist.fisher_dim(case.options)
    assert isinstance(dim, int) and dim >= 0, f"fisher_dim returned {dim!r}"

    encoded = sample_encoded(dist, case.n, case_seed(case.case_id))
    scores = dense_scores(dist, encoded, case.options)
    check_score_array(scores, case.n, dim)
    check_determinism(dist, encoded, case.options)
    check_field_slices(dist, case.options, dim)
    if case.check_moments:
        check_moments(scores, case.rank_deficit, case.centered)
