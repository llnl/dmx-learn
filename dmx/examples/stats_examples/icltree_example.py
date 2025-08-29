"""Example for ICLTreeDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.

ICLTree (Chow-Liu Tree) are great for unknown first order
dependencies. In contrast to Markov chains, the feature order
can be learned. 

""" 
import numpy as np
from numpy.random import RandomState
from dmx.stats import *
from dmx.utils.estimation import optimize


if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    dep_list = [(0, None), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
    cond_list = [np.log(rng.dirichlet(np.ones(20)))]
    cond_list.extend([np.log(rng.dirichlet(np.ones(20), size=20)) for _ in
                      range(7)])
    dist = ICLTreeDistribution(dependency_list=dep_list,
                               conditional_log_densities=cond_list)
    # Generate data
    sampler = dist.sampler(1)
    data = sampler.sample(n)
    # Define the estimator
    est = ICLTreeEstimator(num_features=8, num_states=20, pseudo_count=1.0e-4)
    # Estimate model
    model = optimize(data, est, max_its=100, rng=rng, print_iter=1)
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs: {x}, Likelihood: {y}.')
