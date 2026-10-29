"""Self-tests for the shared Fisher score contract in ``tests/fisher_contract.py``.

Every real family is skipped until it is implemented, so these tests run the
contract on small toy scores: correct ones must pass, and each kind of defect
the contract is meant to catch must fail.
"""

from typing import Dict, Optional

import numpy as np
import pytest

from dmx.stats import CategoricalDistribution, GaussianDistribution
from dmx.utils.fisher import FisherScoreOptions
from tests.fisher_contract import (
    N_SAMPLES,
    FisherCase,
    check_moments,
    run_fisher_contract,
)


class _WhitenedGaussian(GaussianDistribution):
    """Gaussian with a correct whitened score, plus switchable defects."""

    defect: Optional[str] = None

    def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int:
        """Return the location and scale score dimension."""
        return 2

    def seq_fisher_score(
        self, x: object, options: Optional[FisherScoreOptions] = None
    ) -> np.ndarray:
        """Return ``[u, (u^2 - 1) / sqrt(2)]`` with an optional defect."""
        u = (np.asarray(getattr(x, "data")) - self.mu) / np.sqrt(self.sigma2)
        scores = np.column_stack((u, (u * u - 1.0) / np.sqrt(2.0)))
        if self.defect == "scale":
            scores[:, 0] *= 1.5
        elif self.defect == "offset":
            scores[:, 0] += 0.3
        elif self.defect == "correlated":
            scores[:, 1] = (scores[:, 1] + scores[:, 0]) / np.sqrt(2.0)
        elif self.defect == "noise":
            scores = scores + np.random.default_rng().normal(0.0, 1e-3, scores.shape)
        elif self.defect == "nan":
            scores[0, 0] = np.nan
        elif self.defect == "float32":
            scores = scores.astype(np.float32)
        elif self.defect == "extra_column":
            scores = np.column_stack((scores, np.zeros(len(u))))
        return scores

    def fisher_field_slices(
        self, options: Optional[FisherScoreOptions] = None
    ) -> Dict[str, slice]:
        """Return one field, or overlapping fields for the slices defect."""
        if self.defect == "slices":
            return {"location": slice(0, 2), "scale": slice(1, 2)}
        return super().fisher_field_slices(options)


def _gaussian(defect: Optional[str] = None) -> _WhitenedGaussian:
    """Return a toy Gaussian with the given defect."""
    dist = _WhitenedGaussian(1.0, 4.0)
    dist.defect = defect
    return dist


class _WhitenedCategorical(CategoricalDistribution):
    """Categorical with the dense one-hot score, or its offset-free form."""

    offset_free: bool = False

    def _levels(self) -> list:
        """Return the levels in score order."""
        return sorted(self.pmap)

    def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int:
        """Return the number of levels."""
        return len(self.pmap)

    def seq_fisher_score(
        self, x: object, options: Optional[FisherScoreOptions] = None
    ) -> np.ndarray:
        """Return ``e_x / sqrt(p_x) - sqrt(p)``, without the offset if set."""
        xs, values = getattr(x, "data")
        levels = self._levels()
        p = np.asarray([self.pmap[level] for level in levels])
        index = np.asarray([levels.index(value) for value in values])[xs]
        scores = np.zeros((len(index), len(levels)))
        scores[np.arange(len(index)), index] = 1.0 / np.sqrt(p[index])
        if not self.offset_free:
            scores -= np.sqrt(p)
        return scores


def _categorical(offset_free: bool = False) -> _WhitenedCategorical:
    """Return a toy categorical."""
    dist = _WhitenedCategorical({"a": 0.5, "b": 0.3, "c": 0.2})
    dist.offset_free = offset_free
    return dist


def test_correct_gaussian_score_passes() -> None:
    run_fisher_contract(FisherCase("toy_gaussian", _gaussian, issue=0))


def test_correct_categorical_score_passes_as_projector() -> None:
    run_fisher_contract(
        FisherCase("toy_categorical", _categorical, issue=0, rank_deficit=1)
    )


def test_offset_free_categorical_passes_when_not_centered() -> None:
    run_fisher_contract(
        FisherCase(
            "toy_categorical_offset_free",
            lambda: _categorical(offset_free=True),
            issue=0,
            rank_deficit=1,
            centered=False,
        )
    )


@pytest.mark.parametrize(
    "defect",
    [
        "scale",
        "offset",
        "correlated",
        "noise",
        "nan",
        "float32",
        "extra_column",
        "slices",
    ],
)
def test_gaussian_defects_fail(defect: str) -> None:
    with pytest.raises(AssertionError):
        run_fisher_contract(
            FisherCase(f"toy_gaussian_{defect}", lambda: _gaussian(defect), issue=0)
        )


def test_categorical_fails_without_its_rank_deficit() -> None:
    with pytest.raises(AssertionError, match="projector"):
        run_fisher_contract(FisherCase("toy_categorical_full", _categorical, issue=0))


def test_offset_free_categorical_fails_when_centered() -> None:
    with pytest.raises(AssertionError, match="mean"):
        run_fisher_contract(
            FisherCase(
                "toy_categorical_offset_mean",
                lambda: _categorical(offset_free=True),
                issue=0,
                rank_deficit=1,
            )
        )


def test_unsupported_family_is_skipped_with_issue() -> None:
    case = FisherCase("plain_gaussian", lambda: GaussianDistribution(0.0, 1.0), 279)
    with pytest.raises(pytest.skip.Exception, match="llnl/dmx-learn#279"):
        run_fisher_contract(case)


def test_moments_allow_empty_scores() -> None:
    check_moments(np.zeros((N_SAMPLES, 0)), rank_deficit=0, centered=True)


def test_moments_accept_exact_constraints() -> None:
    # Two identical whitened columns scaled by 1/sqrt(2) span a rank-1
    # projector with an exact null direction.
    u = np.random.default_rng(0).standard_normal(N_SAMPLES)
    scores = np.column_stack((u, u)) / np.sqrt(2.0)
    check_moments(scores, rank_deficit=1, centered=True)
    with pytest.raises(AssertionError):
        check_moments(scores, rank_deficit=0, centered=True)
