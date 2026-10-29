"""Private helpers for Fisher scores of latent-variable dmx.stats families.

A mixture used as a component (nested inside another model) has no
closed-form Fisher information. Its score is built from the observed-data
score and whitened with a covariance estimated from draws of the mixture
itself, so the result depends only on the model, not on the records scored.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from dmx.utils._fisher_kernels import clamp_probability
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherScoreOptions,
    component_score_dims,
    warn_unsupported,
)

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
    draws = mixture.sampler(seed=options.whitening_seed).sample(
        options.whitening_samples
    )
    sample = mixture.dist_to_encoder().seq_encode(draws)
    whitener = EmpiricalFisherWhitener("full", center=False).fit(
        _raw_mixture_score(mixture, sample, component_data, dims, options)
    )
    raw = _raw_mixture_score(mixture, encoded, component_data, dims, options)
    return whitener.transform(raw)


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
