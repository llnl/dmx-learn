"""Example for GaussianMixtureDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.

This is a fast implementation of a mixture of univariate Gaussians.
This model allows for the variance components to be key'd which can 
not be easily done with MixtureEstimator([GaussianEstimator()]*K).

This also differs from GaussianMixtureDistribution(), as it is
univariate, and much faster for univariate case. 

""" 
import os

from numpy.random import RandomState
from dmx.stats import *
from dmx.utils.estimation import optimize


if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    mu = [0.0, 5.0, 10.0]
    sigma2 = [1.0, 1.0, 1.0]
    w = [0.50, 0.40, 0.10]
    dist = GaussianMixtureDistribution(mu=mu, sigma2=sigma2, w=w)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est = GaussianMixtureEstimator(
            num_components=3,
            tied=True)
    # Estimate model
    model = optimize(data, est, max_its=100, rng=rng, print_iter=1)
    print(str(model))
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data: {ll}')
    # Fast evaluation of the posterior
    post = model.seq_posterior(enc_data)
    print(f'Posterior of each observed point: {post}')
