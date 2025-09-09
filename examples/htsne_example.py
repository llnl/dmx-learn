"""A detailed example of heterogenous SNE embeddings."""
import numpy as np
from dmx.stats import *
from dmx.utils.htsne import htsne

def sample_with_labels(size, mixture_comps, mixture_weights, rng):
    seeds = rng.randint(low=0, high=2**32, size=len(mixture_comps))

    samplers = [comp.sampler(seed=s) for s, comp in zip(seeds, mixture_comps)]

    labels = rng.choice(len(mixture_comps), p=mixture_weights, replace=True, size=size)
    labels = np.bincount(labels, minlength=len(mixture_comps))

    cnt = 0
    rv0 = np.zeros(size, dtype=int)
    rv1 = []
    for i, c in enumerate(labels):
        if c > 0:
            rv0[cnt:(cnt+c)] += i
            rv1.extend(samplers[i].sample(c))
            cnt += c

    return rv0, rv1


if __name__ == '__main__':
    rng = np.random.RandomState(1)
    # define composite mixture
    ncomps = 5
    p = 0.75
    p_vec = np.ones((ncomps, ncomps))*(1.0-p)/(ncomps-1)
    np.fill_diagonal(p_vec, p)

    s2 = 1.0
    mu = np.linspace(-10, 10, ncomps)

    comps = []
    for i in range(ncomps):

        d0 = IntegerCategoricalDistribution(min_val=0, p_vec=p_vec[i])
        d1 = GaussianDistribution(mu=float(mu[i]), sigma2=s2)

        comps.append(CompositeDistribution([d0, d1]))

    dist = MixtureDistribution(comps, w=np.ones(ncomps) / ncomps)

    # simulate data from mixture
    N = int(1e3)
    labels, data = sample_with_labels(
            size=N,
            mixture_comps=dist.components,
            mixture_weights=dist.w,
            rng=rng)
    embs = htsne(data, mix_model=dist)

    # make plot 

   

