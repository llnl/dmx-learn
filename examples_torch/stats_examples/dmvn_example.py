"""Example for DiagonalGaussianDistribution. Define distribution,
generate data, estimate, and evaluate likelihoods."""
import numpy as np
from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == '__main__':
    n = int(1e4)
    # Define the model
    dist = DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, 1.0])
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est = DiagonalGaussianEstimator(dim=2)
    # Estimate model
    model = optimize(data=data, estimator=est, max_its=100, seed=1, print_iter=1, device=device)
    print(str(model))
    # Eval likelihood on a an observation
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data: {ll}')
