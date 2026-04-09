"""Example for MultivariateGaussianDistribution. Define distribution,
generate data, estimate, and evaluate likelihoods.
"""

import numpy as np
import torch

from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    n = int(1e4)
    rng = np.random.RandomState(1)
    # Define the model
    m = [0.0, 5.0, 10.0]
    cov = np.reshape(rng.rand(9), (3, 3))
    cov = np.matmul(cov, cov.T).tolist()
    dist = MultivariateGaussianDistribution(mu=m, covar=cov)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est = MultivariateGaussianEstimator(dim=3)
    # Estimate model
    model = optimize(
        data=data, estimator=est, max_its=100, seed=1, print_iter=1, device=device
    )
    print(str(model))
    # Eval likelihood on an observation
    ll0 = model.log_density(data[0])
    print(f"Likelihood of estimated model eval at {data[0]}: {ll0}")
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f"Likelihood of estimated model on data: {ll}")

    # Check model device and move it to cpu (or some other device if preferred)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
