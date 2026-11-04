"""Private helpers for Fisher scores of latent-variable dmx.stats families.

A mixture used as a component (nested inside another model) has no
closed-form Fisher information. Its score is built from the observed-data
score and whitened with a covariance estimated from draws of the mixture
itself, so the result depends only on the model, not on the records scored.
"""

from typing import Any, Dict, NamedTuple, Optional, Sequence

import numpy as np

from dmx.utils._fisher_kernels import length_weights, scale_rows, sum_rows
from dmx.utils._fisher_mixture import (
    _raw_mixture_score,
    dense_scores,
    mixture_fisher_embedding,
    model_whitener,
    nested_mixture_dim,
    nested_mixture_score,
    nested_mixture_slices,
)
from dmx.utils.fisher import FisherScoreArray, FisherScoreOptions

__all__ = [
    "_raw_mixture_score",
    "hmm_raw_score",
    "hmm_slices",
    "mixture_fisher_embedding",
    "model_whitener",
    "nested_mixture_dim",
    "nested_mixture_score",
    "nested_mixture_slices",
]


class HmmPosteriors(NamedTuple):
    """Per-sequence forward-backward statistics of a hidden Markov model.

    Attributes:
        initial: Posterior of the first state, shape ``(N, K)``.
        transitions: Expected transition counts, shape ``(N, K, K)``.
        gamma: Posterior state of each element, shape ``(M, K)``, in the
            order of the encoded element data.
        owner: Sequence index of each element, shape ``(M,)``.
        lengths: Length of each sequence, shape ``(N,)``.
    """

    initial: np.ndarray
    transitions: np.ndarray
    gamma: np.ndarray
    owner: np.ndarray
    lengths: np.ndarray


def hmm_raw_score(
    posteriors: HmmPosteriors,
    w: Any,
    transitions: Any,
    emission_scores: Sequence[Optional[FisherScoreArray]],
    length_block: Optional[FisherScoreArray],
    options: FisherScoreOptions,
) -> np.ndarray:
    """Return the unwhitened score of hidden Markov sequences.

    Blocks, each an expected complete-data statistic minus its model mean so
    that the mean is exactly zero:

    - initial: ``gamma_0 - w`` (zero for an empty sequence);
    - transitions: ``xi_ij - (sum_j xi_ij) A_ij`` flattened, from the expected
      transition counts ``xi``;
    - emissions: ``sum_t gamma_t(k) s_k(x_t)`` for each state ``k`` whose
      emission score is given;
    - length: ``length_block`` if given.

    With ``sequence_scaling == "conditional"`` the transition block is divided
    by ``sqrt(T - 1)`` and the emission blocks by ``sqrt(T)``, per sequence.
    ``"marginal"`` leaves them unscaled: the scores are whitened afterwards,
    which makes any constant per-block factor irrelevant.

    Args:
        posteriors: Per-sequence forward-backward statistics.
        w: Initial state probabilities, shape ``(K,)``.
        transitions: Transition matrix, shape ``(K, K)``.
        emission_scores: Per-state element scores ``(M, d_k)``, or None to
            skip a state's emission block.
        length_block: Length distribution scores ``(N, d)``, or None.
        options: Resolved score options.

    Returns:
        Array of shape ``(N, K + K^2 + sum_k d_k + d)``.
    """
    lengths = np.asarray(posteriors.lengths, dtype=np.float64)
    n, k = posteriors.initial.shape
    conditional = options.sequence_scaling == "conditional"
    initial = np.where((lengths > 0)[:, None], posteriors.initial - np.asarray(w), 0.0)
    xi = posteriors.transitions
    trans = (
        xi - xi.sum(axis=2, keepdims=True) * np.asarray(transitions)[None]
    ).reshape(n, k * k)
    if conditional:
        trans = (
            trans * length_weights(np.maximum(lengths - 1, 0), "conditional")[:, None]
        )
    blocks = [initial, trans]
    emission_weights = length_weights(lengths, "conditional") if conditional else None
    for state, scores in enumerate(emission_scores):
        if scores is None:
            continue
        summed = sum_rows(scores, posteriors.owner, n, posteriors.gamma[:, state])
        if emission_weights is not None:
            summed = scale_rows(summed, emission_weights)
        blocks.append(dense_scores(summed))
    if length_block is not None:
        blocks.append(dense_scores(length_block))
    return np.ascontiguousarray(np.hstack(blocks))


def hmm_slices(
    num_states: int, emission_dims: Sequence[Optional[int]], length_dim: int
) -> Dict[str, slice]:
    """Map the HMM score parts to their columns."""
    parts = [("initial", num_states), ("transitions", num_states * num_states)]
    parts += [
        (f"emission_{state}", dim) for state, dim in enumerate(emission_dims) if dim
    ]
    parts.append(("length", length_dim))
    slices: Dict[str, slice] = {}
    offset = 0
    for name, dim in parts:
        if dim:
            slices[name] = slice(offset, offset + dim)
            offset += dim
    return slices
