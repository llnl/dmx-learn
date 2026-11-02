"""Private helpers for Fisher scores of latent-variable dmx.stats families.

A mixture used as a component (nested inside another model) has no
closed-form Fisher information. Its score is built from the observed-data
score and whitened with a covariance estimated from draws of the mixture
itself, so the result depends only on the model, not on the records scored.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from dmx.utils._fisher_kernels import (
    clamp_probability,
    length_weights,
    scale_rows,
    sum_rows,
)
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherScoreArray,
    FisherScoreOptions,
    component_score_dims,
    warn_unsupported,
)


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


ComponentData = Callable[[Any, int], Any]


def _dense(scores: Any) -> np.ndarray:
    """Return scores as a dense float64 array."""
    if hasattr(scores, "toarray"):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float64)


def _raw_mixture_score(
    mixture: Any,
    encoded: Any,
    component_data: ComponentData,
    dims: List[Optional[int]],
    options: FisherScoreOptions,
) -> np.ndarray:
    """Return the unwhitened score ``[(z - w)/sqrt(w), z_j s_j/sqrt(w_j)]``."""
    posteriors = np.asarray(mixture.seq_posterior(encoded), dtype=np.float64)
    weights = np.asarray(
        [clamp_probability(w, options.prob_floor) for w in mixture.w], dtype=np.float64
    )
    sqrt_w = np.sqrt(weights)
    blocks = [(posteriors - weights) / sqrt_w]
    for j, dim in enumerate(dims):
        if not dim:
            continue
        scores = _dense(
            mixture.components[j].seq_fisher_score(component_data(encoded, j), options)
        )
        blocks.append(scores * (posteriors[:, j] / sqrt_w[j])[:, None])
    return np.ascontiguousarray(np.hstack(blocks))


def nested_mixture_score(
    mixture: Any,
    encoded: Any,
    component_data: ComponentData,
    options: FisherScoreOptions,
) -> np.ndarray:
    """Return the whitened score of a mixture used as a component.

    The raw score stacks the weight block ``(z_j - w_j) / sqrt(w_j)`` and the
    component blocks ``z_j s_j(x) / sqrt(w_j)``, where ``z`` are posteriors,
    ``w`` weights (clipped with ``options.prob_floor``), and ``s_j`` the
    component's whitened score. This is the observed-data score scaled by the
    complete-data Fisher information: its mean is exactly zero, but the
    complete-data information over-states the observed one, so its covariance
    is below the identity. It is therefore whitened (ZCA) with the covariance
    of the raw score over ``options.whitening_samples`` draws from the mixture,
    seeded with ``options.whitening_seed``. The weight block's sum-to-one
    constraint leaves one null direction, so the result has a projector
    covariance of rank ``fisher_dim - 1``.

    Under ``options.on_missing == "posterior_only"`` components without a
    score contribute only through the weight block, and one warning lists
    their classes.

    Args:
        mixture: Mixture with ``w``, ``components``, ``seq_posterior``,
            ``sampler``, and ``dist_to_encoder``.
        encoded: The mixture's encoded data.
        component_data: Returns the encoded data for component ``j``.
        options: Resolved score options.

    Returns:
        Array of shape ``(N, J + sum of supported component dimensions)``.
    """
    dims, unsupported = component_score_dims(mixture.components, options)
    warn_unsupported(unsupported)

    def raw_score(enc: Any) -> np.ndarray:
        return _raw_mixture_score(mixture, enc, component_data, dims, options)

    return model_whitener(mixture, raw_score, options).transform(raw_score(encoded))


def model_whitener(
    dist: Any, raw_score: Callable[[Any], np.ndarray], options: FisherScoreOptions
) -> EmpiricalFisherWhitener:
    """Fit a ZCA whitener on raw scores of draws from ``dist`` itself.

    Using draws from the model, rather than the records being scored, makes
    the whitened score a deterministic function of the model: chunked and
    out-of-sample scoring agree. The raw score must have mean zero under the
    model, so no centering is applied.

    Args:
        dist: Distribution with ``sampler`` and ``dist_to_encoder``.
        raw_score: Maps the distribution's encoded data to raw scores.
        options: Resolved score options; ``whitening_samples`` draws are taken
            with seed ``whitening_seed``.

    Returns:
        The fitted whitener.

    Raises:
        RuntimeError: If the model cannot draw samples, for example a hidden
            Markov model without a length distribution or terminal values.
    """
    try:
        draws = dist.sampler(seed=options.whitening_seed).sample(
            options.whitening_samples
        )
    except (RuntimeError, ValueError) as err:
        raise RuntimeError(
            f"Fisher scores of {type(dist).__name__} are whitened with draws from "
            f"the model, which failed: {err}"
        ) from err
    sample = dist.dist_to_encoder().seq_encode(draws)
    return EmpiricalFisherWhitener("full", center=False).fit(raw_score(sample))


def nested_mixture_dim(mixture: Any, options: FisherScoreOptions) -> int:
    """Return ``J`` plus the dimensions of the components that have scores."""
    dims, _ = component_score_dims(mixture.components, options)
    return len(mixture.components) + sum(dim or 0 for dim in dims)


def nested_mixture_slices(
    mixture: Any, options: FisherScoreOptions
) -> Dict[str, slice]:
    """Map ``"weights"`` and ``"component_<j>"`` to their score columns.

    Whitening mixes columns, so after ZCA whitening these slices attribute
    distance approximately rather than exactly.
    """
    dims, _ = component_score_dims(mixture.components, options)
    num = len(mixture.components)
    slices = {"weights": slice(0, num)}
    offset = num
    for j, dim in enumerate(dims):
        if dim:
            slices[f"component_{j}"] = slice(offset, offset + dim)
            offset += dim
    return slices


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
        blocks.append(_dense(summed))
    if length_block is not None:
        blocks.append(_dense(length_block))
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
