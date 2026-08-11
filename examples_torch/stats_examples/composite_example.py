"""Composite distribution example.

Generate paired synthetic data from independent component distributions, fit a
composite model, and evaluate log densities.
"""

# pylint: disable=duplicate-code

import torch

from dmx.torch_stats import (
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    PoissonDistribution,
    PoissonEstimator,
    seq_encode,
)
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    n = int(1e4)
    # Define the model: CompositeDistribution of two independent distributions
    dist0 = GaussianDistribution(mu=3.0, sigma2=1.0)
    dist1 = PoissonDistribution(lam=5.0)
    dist = CompositeDistribution([dist0, dist1])
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = GaussianEstimator()
    est1 = PoissonEstimator()
    est = CompositeEstimator([est0, est1])
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
