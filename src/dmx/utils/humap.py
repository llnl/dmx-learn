"""Heterogenous UMAP for embedding tuples of heterogenous data in lower-dimensions."""
from typing import Sequence, Optional, Tuple, TypeVar, Any, Dict
import numpy as np
from numpy.random import RandomState
from dmx.utils.automatic import get_dpm_mixture
from dmx.bstats import *
from dmx.bstats import MixtureDistribution
from dmx.bstats.pdist import ParameterEstimator
import umap 
from umap import UMAP

DATUM_TYPE = TypeVar('DATUM_TYPE')


# --- UMAP with fixed seed ---
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='hellinger',
    random_state=42      # << seed
)

DEFAULT_UMAP = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.10,
    "metric": "hellinger"
}


def humap(data: Sequence[DATUM_TYPE], 
          max_components: int = 30, 
          mix_threshold_count: float = 0.5,
          max_its: int = 1000, 
          print_iter: int = 100,
          seed: Optional[int] = None, 
          comp_estimator: Optional[ParameterEstimator] = None, 
          mix_model: Optional[MixtureDistribution] = None,
          umap_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, MixtureDistribution, UMAP, np.ndarray]:
    """Performs UMAP fit on posteriors of DPM mixture model. 

    Args:
        data (Sequence[DATUM_TYPE]): Input data sequence.
        max_components (int): Maximum number of components for the mixture model.
        mix_threshold_count (float): Threshold for mixture component selection.
        seed (Optional[int]): Random seed for reproducibility.
        comp_estimator (Optional[ParameterEstimator]): Component estimator for mixture model.
        mix_model (Optional[MixtureDistribution]): Precomputed mixture model.
        umap_kwargs (Optional[Dict[str, Any]]): Kwargs for UMAP fit. 
    Returns:
        embeddings, dpm mixture model, umap fit, posteriors 

    """

    rng = RandomState(seed) if seed is not None else RandomState()
    if max_components <= 1 or not isinstance(max_components, (int, np.integer)):
        raise Exception('max_components must be and integer greater than 1.')
    # Fit DPM to data using comp_estimator if passed.
    if mix_model is None:
        mix_model = get_dpm_mixture(
                data=data,
                estimator=comp_estimator,
                max_comp=max_components,
                rng=rng,
                max_its=max_its,
                print_iter=print_iter,
                mix_threshold_count=mix_threshold_count)
    # Mixture must have at least one comp!
    if mix_model.num_components == 0:
        raise Exception('Something is broken. Mixture model has zero components.')
    # This is until all bstats is updated!
    try:
        enc_data = mix_model.dist_to_encoder().seq_encode(data)
    except Exception:
        enc_data = mix_model.seq_encode(data)
    # Posterior and log comp density for each point [z | x] and [x | z]
    posteriors = mix_model.seq_posterior(enc_data)

    if umap_kwargs is not None:
        for k, v in DEFAULT_UMAP.items():
            if k not in umap_kwargs:
                umap_kwargs[k] = v
    else:
        umap_kwargs = DEFAULT_UMAP


    fit = umap.UMAP(**umap_kwargs)
    embeddings = fit.fit_transform(posteriors)

    return embeddings, mix_model, fit, posteriors
