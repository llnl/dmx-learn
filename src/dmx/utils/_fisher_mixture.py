"""Private Fisher-score helpers shared by stats and bstats mixtures."""

from __future__ import annotations

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


def dense_scores(scores: Any) -> np.ndarray:
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
    """Return ``[(z - w)/sqrt(w), z_j s_j/sqrt(w_j)]`` before whitening."""
    posteriors = np.asarray(mixture.seq_posterior(encoded), dtype=np.float64)
    weights = np.asarray(
        [clamp_probability(w, options.prob_floor) for w in mixture.w],
        dtype=np.float64,
    )
    sqrt_weights = np.sqrt(weights)
    blocks = [(posteriors - weights) / sqrt_weights]
    for index, dim in enumerate(dims):
        if not dim:
            continue
        scores = dense_scores(
            mixture.components[index].seq_fisher_score(
                component_data(encoded, index), options
            )
        )
        blocks.append(scores * (posteriors[:, index] / sqrt_weights[index])[:, None])
    return np.ascontiguousarray(np.hstack(blocks))


def model_whitener(
    dist: Any, raw_score: Callable[[Any], np.ndarray], options: FisherScoreOptions
) -> EmpiricalFisherWhitener:
    """Fit a deterministic ZCA whitener to raw scores drawn from ``dist``."""
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


def nested_mixture_score(
    mixture: Any,
    encoded: Any,
    component_data: ComponentData,
    options: FisherScoreOptions,
) -> np.ndarray:
    """Return the empirically whitened score of a nested mixture.

    The raw score uses plug-in mixture weights and component parameters.
    Components without scores follow ``options.on_missing``. Whitening uses
    deterministic draws from the model, so results do not depend on the data
    batch being scored.
    """
    dims, unsupported = component_score_dims(mixture.components, options)
    warn_unsupported(unsupported)

    def raw_score(enc: Any) -> np.ndarray:
        return _raw_mixture_score(mixture, enc, component_data, dims, options)

    return model_whitener(mixture, raw_score, options).transform(raw_score(encoded))


def nested_mixture_dim(mixture: Any, options: FisherScoreOptions) -> int:
    """Return mixture-weight columns plus supported component dimensions."""
    dims, _ = component_score_dims(mixture.components, options)
    return len(mixture.components) + sum(dim or 0 for dim in dims)


def nested_mixture_slices(
    mixture: Any, options: FisherScoreOptions
) -> Dict[str, slice]:
    """Map mixture weights and component score blocks to column slices."""
    dims, _ = component_score_dims(mixture.components, options)
    component_count = len(mixture.components)
    slices = {"weights": slice(0, component_count)}
    offset = component_count
    for index, dim in enumerate(dims):
        if dim:
            slices[f"component_{index}"] = slice(offset, offset + dim)
            offset += dim
    return slices
