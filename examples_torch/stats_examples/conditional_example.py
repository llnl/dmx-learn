"""Example for ConditionalDistribution. Define distribution, generate data,
estimate, and evaluate likelihoods.

Note: The torch version uses IntegerCategoricalDistribution as the given
distribution (integer keys), in place of a string-keyed CategoricalDistribution
which is not available in torch_stats.
"""
import torch
import numpy as np
from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == '__main__':
    n = int(1e4)
    # Define the model
    # Conditional Gaussian distributions given an integer value
    d1 = GaussianDistribution(1.0, 1.0)
    d2 = GaussianDistribution(5.0, 1.0)
    d3 = GaussianDistribution(3.0, 1.0)
    # Given distribution: integer categorical with support {0, 1, 2, 3}
    d0 = IntegerCategoricalDistribution(min_val=0, p_vec=np.array([0.5, 0.2, 0.2, 0.1]))
    dist = ConditionalDistribution(
            dmap={0: d1, 1: d2},
            default_dist=d3,
            given_dist=d0)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = GaussianEstimator()
    est1 = IntegerCategoricalEstimator()
    emap = {0: est0, 1: est0}
    est = ConditionalDistributionEstimator(
            estimator_map=emap,
            default_estimator=est0,
            given_estimator=est1)
    # Estimate model
    model = optimize(data=data, estimator=est, max_its=100, seed=1, print_iter=1, device=device)
    print(str(model))
    # Eval likelihood on an observation
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data: {ll}')

    # Check model device and move it to cpu (or some other device if preferred)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
