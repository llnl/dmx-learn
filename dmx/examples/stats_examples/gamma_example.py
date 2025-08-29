"""Example for GammaDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.
""" 
from numpy.random import RandomState
from dmx.stats import *
from dmx.utils.estimation import optimize


if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    dist = GammaDistribution(k=2.0, theta=5.0) 
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est = GammaEstimator() 
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
