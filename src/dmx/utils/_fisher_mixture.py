"""Private Fisher-score helpers shared by stats and bstats mixtures."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from dmx.utils._fisher_kernels import clamp_probability
from dmx.utils.fisher import (
    EmpiricalFisherWhitener,
    FisherEmbedding,
    FisherScoreOptions,
    component_score_dims,
    resolve_fisher_options,
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


class EmbeddingLayout(NamedTuple):
    """Columns of a mixture Fisher embedding.

    Attributes:
        dims: Width of each component's score block, or None for components
            without one (zero weight, or no score under ``posterior_only``).
        component_slices: Columns of each component block.
        field_slices: Columns of each field inside a component block.
        width: Total number of columns.
    """

    dims: List[Optional[int]]
    component_slices: Dict[int, slice]
    field_slices: Dict[int, Dict[str, slice]]
    width: int


def check_embedding_args(scale: Optional[float], top_m: Optional[int]) -> None:
    """Raise ValueError for a negative or non-finite scale or a bad ``top_m``."""
    if scale is not None and not (math.isfinite(scale) and scale >= 0.0):
        raise ValueError(f"scale must be finite and nonnegative, got {scale!r}.")
    if top_m is not None and (
        isinstance(top_m, bool) or not isinstance(top_m, (int, np.integer)) or top_m < 1
    ):
        raise ValueError(f"top_m must be a positive integer or None, got {top_m!r}.")


def mixture_component_data(encoded: Any) -> Any:
    """Return the component encoding inside a mixture's encoded sequence."""
    return encoded.data


def embedding_layout(
    mixture: Any,
    options: FisherScoreOptions,
    whiteners: Optional[Sequence[Optional[EmpiricalFisherWhitener]]],
) -> EmbeddingLayout:
    """Return the column layout of ``mixture``'s Fisher embedding.

    Warns once if components are skipped under ``on_missing="posterior_only"``.

    Raises:
        ValueError: If ``whiteners`` does not have one entry per component, or
            a whitener does not match its component's score dimension.
        NotImplementedError: If a component has no score and
            ``options.on_missing == "raise"``.
    """
    count = len(mixture.components)
    if whiteners is not None and len(whiteners) != count:
        raise ValueError(
            f"Expected {count} whiteners (one per component), got {len(whiteners)}."
        )
    zero_weight = np.asarray(mixture.zw, dtype=bool)
    scored = [
        component
        for index, component in enumerate(mixture.components)
        if not zero_weight[index]
    ]
    scored_dims, unsupported = component_score_dims(scored, options)
    warn_unsupported(unsupported)
    next_dim = iter(scored_dims)
    dims: List[Optional[int]] = []
    component_slices: Dict[int, slice] = {}
    field_slices: Dict[int, Dict[str, slice]] = {}
    offset = count
    for index, component in enumerate(mixture.components):
        dim = None if zero_weight[index] else next(next_dim)
        if not dim:
            dims.append(None)
            continue
        whitener = None if whiteners is None else whiteners[index]
        width = dim
        if whitener is not None:
            if whitener.matrix_ is None or whitener.matrix_.shape[0] != dim:
                raise ValueError(
                    f"Whitener for component {index} must be fitted on {dim} "
                    "score columns."
                )
            width = int(whitener.matrix_.shape[1])
        dims.append(width)
        component_slices[index] = slice(offset, offset + width)
        field_slices[index] = (
            {
                name: slice(offset + part.start, offset + part.stop)
                for name, part in component.fisher_field_slices(options).items()
            }
            if width == dim
            else {}
        )
        offset += width
    return EmbeddingLayout(dims, component_slices, field_slices, offset)


def default_scale(
    posteriors: Sequence[np.ndarray], dims: Sequence[Optional[int]]
) -> float:
    """Return ``1 / p_eff`` with ``p_eff = mean_i sum_k z_ik dim_k``.

    ``posteriors`` is a list of row blocks; the result does not depend on how
    the rows are split. Returns 0 when no record has a score block.
    """
    widths = np.asarray([dim or 0 for dim in dims], dtype=np.float64)
    per_row = np.concatenate([(block * widths).sum(axis=1) for block in posteriors])
    effective = float(per_row.mean()) if per_row.size else 0.0
    return 1.0 / effective if effective > 0.0 else 0.0


def top_components(posteriors: np.ndarray, top_m: Optional[int]) -> np.ndarray:
    """Mark each row's ``top_m`` most probable components with nonzero posterior.

    Ties are broken toward the lower component index.
    """
    keep = posteriors > 0.0
    if top_m is not None and top_m < posteriors.shape[1]:
        order = np.argsort(-posteriors, axis=1, kind="stable")[:, :top_m]
        mask = np.zeros_like(keep)
        np.put_along_axis(mask, order, True, axis=1)
        keep &= mask
    return keep


def mixture_fisher_embedding(
    mixture: Any,
    encoded: Any,
    component_encoded: ComponentData,
    *,
    scale: Optional[float],
    top_m: Optional[int],
    options: Optional[FisherScoreOptions],
    dtype: Any,
    whiteners: Optional[Sequence[Optional[EmpiricalFisherWhitener]]],
) -> FisherEmbedding:
    """Return ``[sqrt(z_i), sqrt(c z_ik) s_ik for k in top_m(i)]`` for each row.

    Shared by the stats and bstats ``MixtureDistribution.seq_fisher_embedding``.
    Scores are computed for all rows of a component that any row keeps;
    values are assembled in float64 and cast to ``dtype`` at the end.
    """
    options = resolve_fisher_options(options)
    check_embedding_args(scale, top_m)
    layout = embedding_layout(mixture, options, whiteners)
    posteriors = np.asarray(mixture.seq_posterior(encoded), dtype=np.float64)
    if scale is None:
        scale = default_scale([posteriors], layout.dims)
    count = posteriors.shape[0]
    keep = top_components(posteriors, top_m)

    block = coo_matrix(np.sqrt(posteriors))
    rows, cols, values = [block.row], [block.col], [block.data]
    for index, columns in layout.component_slices.items():
        selected = np.flatnonzero(keep[:, index])
        if scale == 0.0 or selected.size == 0:
            continue
        scores = mixture.components[index].seq_fisher_score(
            component_encoded(encoded, index), options
        )
        weights = np.sqrt(scale * posteriors[selected, index])
        whitener = None if whiteners is None else whiteners[index]
        if whitener is not None:
            weighted: Any = whitener.transform(_select_rows(scores, selected))
            weighted *= weights[:, None]
        elif hasattr(scores, "tocsr"):
            weighted = diags(weights) @ scores.tocsr()[selected]
        else:
            weighted = np.asarray(scores, dtype=np.float64)[selected] * weights[:, None]
        block = coo_matrix(weighted)
        rows.append(selected[block.row])
        cols.append(block.col + columns.start)
        values.append(block.data)

    matrix = csr_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(count, layout.width),
        dtype=np.float64,
    ).astype(dtype)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return FisherEmbedding(
        matrix=matrix,
        scale=float(scale),
        top_m=top_m,
        posterior_slice=slice(0, posteriors.shape[1]),
        component_slices=layout.component_slices,
        field_slices=layout.field_slices,
        options=options,
        whiteners=None if whiteners is None else list(whiteners),
    )


def _select_rows(scores: Any, rows: np.ndarray) -> Any:
    """Return the given rows of dense or sparse scores."""
    if hasattr(scores, "tocsr"):
        return scores.tocsr()[rows]
    return np.asarray(scores, dtype=np.float64)[rows]


ScoreMoments = Tuple[float, np.ndarray, np.ndarray]


def fit_component_whiteners(
    template: EmpiricalFisherWhitener,
    moments: Optional[Sequence[Optional[ScoreMoments]]],
) -> List[Optional[EmpiricalFisherWhitener]]:
    """Fit one whitener per component from posterior-weighted score moments.

    Each entry of ``moments`` is ``(sum_i z_i, sum_i z_i s_i, sum_i z_i s_i
    s_i^T)`` or None. Components without moments or with zero total weight
    get None.
    """
    whiteners: List[Optional[EmpiricalFisherWhitener]] = []
    for entry in moments or []:
        if entry is None or entry[0] <= 0.0:
            whiteners.append(None)
            continue
        total, first, second = entry
        mean = first / total if template.center else np.zeros(len(first))
        covariance = second / total - np.outer(mean, mean)
        whitener = dataclasses.replace(template, mean_=None, matrix_=None)
        whiteners.append(whitener.fit_covariance(covariance, mean))
    return whiteners


def add_score_moments(
    mixture: Any,
    encoded: Any,
    posteriors: np.ndarray,
    dims: Sequence[Optional[int]],
    options: FisherScoreOptions,
    moments: Optional[List[Optional[ScoreMoments]]],
) -> List[Optional[ScoreMoments]]:
    """Add one chunk's posterior-weighted score moments for each component.

    Args:
        mixture: Mixture whose components are scored.
        encoded: The chunk's component encoding.
        posteriors: The chunk's posteriors, shape ``(n, K)``.
        dims: Score dimension per component, None to skip it.
        options: Score options.
        moments: Moments of earlier chunks, or None for the first chunk.

    Returns:
        Updated moments, one entry per component.
    """
    if moments is None:
        moments = [None] * len(dims)
    for index, dim in enumerate(dims):
        if not dim:
            continue
        scores = dense_scores(
            mixture.components[index].seq_fisher_score(encoded, options)
        )
        weights = posteriors[:, index]
        total, first, second = (
            float(weights.sum()),
            weights @ scores,
            (scores * weights[:, None]).T @ scores,
        )
        previous = moments[index]
        if previous is not None:
            total, first, second = (
                previous[0] + total,
                previous[1] + first,
                previous[2] + second,
            )
        moments[index] = (total, first, second)
    return moments
