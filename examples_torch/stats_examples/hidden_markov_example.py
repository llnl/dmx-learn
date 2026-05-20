"""Hidden Markov model example.

Generate synthetic sequences from a Gaussian-emission HMM, fit the model, and
evaluate sequence log densities.
"""

# pylint: disable=duplicate-code

import torch

from dmx.torch_stats import (
    GaussianDistribution,
    GaussianEstimator,
    HiddenMarkovEstimator,
    HiddenMarkovModelDistribution,
    PoissonDistribution,
    PoissonEstimator,
    seq_encode,
)
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    n = int(1e4)
    # Define the model: 3-state HMM with Gaussian emission distributions
    d1 = GaussianDistribution(mu=-5.0, sigma2=1.0)
    d2 = GaussianDistribution(mu=0.0, sigma2=1.0)
    d3 = GaussianDistribution(mu=5.0, sigma2=1.0)
    init_weights = [0.4, 0.4, 0.2]
    transitions = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    len_dist = PoissonDistribution(lam=8.0)
    dist = HiddenMarkovModelDistribution(
        topics=[d1, d2, d3], w=init_weights, transitions=transitions, len_dist=len_dist
    )
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est_emission = GaussianEstimator()
    est = HiddenMarkovEstimator(
        estimators=[est_emission] * 3, len_estimator=PoissonEstimator()
    )
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
    print(f"Likelihood of estimated model on data: {ll[:5]}")

    # Check model device and move it to cpu (or some other device if preferred)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
