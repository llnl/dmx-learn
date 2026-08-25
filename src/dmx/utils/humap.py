"""Embed heterogeneous observations using mixture posteriors and UMAP."""

from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np
import umap  # type: ignore[import-untyped]
from numpy.random import RandomState
from umap import UMAP

from dmx.utils.automatic import MixtureModel, prepare_mixture_model

DATUM_TYPE = TypeVar("DATUM_TYPE")


# --- UMAP with fixed seed ---
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="hellinger",
    random_state=42,  # << seed
)

DEFAULT_UMAP = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.10,
    "metric": "hellinger",
}


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def humap(
    data: Sequence[DATUM_TYPE],
    max_components: int = 30,
    mix_threshold_count: float = 0.5,
    max_its: int = 1000,
    print_iter: int = 100,
    seed: Optional[int] = None,
    comp_estimator: Optional[Any] = None,
    mix_model: Optional[MixtureModel] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Any,
    MixtureModel,
    UMAP,
    np.ndarray,
]:
    """Fit UMAP to mixture-component posterior probabilities.

    A supplied mixture is reused; otherwise a pruned Dirichlet-process mixture
    is fitted. ``seed`` controls only mixture fitting. UMAP randomness is
    controlled independently through ``umap_kwargs`` (for example,
    ``random_state``). Missing default UMAP options are inserted into a
    caller-supplied dictionary in place.

    Args:
        data: Nonempty heterogeneous observations.
        max_components: Maximum components used when fitting a mixture.
        mix_threshold_count: Minimum effective component count retained after
            mixture fitting.
        max_its: Maximum mixture-fitting iterations.
        print_iter: Interval for mixture-fitting progress output.
        seed: Seed for the local legacy NumPy random state used in mixture
            fitting. ``None`` uses OS entropy.
        comp_estimator: Optional estimator for one mixture component.
        mix_model: Optional pre-fitted mixture distribution.
        umap_kwargs: Options passed to :class:`umap.UMAP`. Defaults are added
            for missing keys.

    Returns:
        The embedding with shape ``(len(data), n_components)``, fitted mixture
        model, fitted UMAP object, and posterior matrix with shape
        ``(len(data), n_mixture_components)``.

    Raises:
        ValueError: If ``max_components`` is not an integer greater than one,
            or UMAP rejects its inputs or options.
        RuntimeError: If mixture fitting produces no components.
    """
    rng = RandomState(seed) if seed is not None else RandomState()
    mix_model, _, posteriors = prepare_mixture_model(
        data,
        rng,
        max_components,
        mix_threshold_count,
        max_its,
        print_iter,
        comp_estimator,
        mix_model,
    )

    if umap_kwargs is not None:
        for k, v in DEFAULT_UMAP.items():
            if k not in umap_kwargs:
                umap_kwargs[k] = v
    else:
        umap_kwargs = DEFAULT_UMAP

    fit = umap.UMAP(**umap_kwargs)
    embeddings = fit.fit_transform(posteriors)

    return embeddings, mix_model, fit, posteriors
