"""Example for JointMixtureDistribution. Define distribution,
generate data, estimate, and evaluate likelihoods.

JointMixtureDistribution: f(x1, x2) = sum_k pi_k * f_k(x1) * sum_j tau_{kj} * g_j(x2).

Note: The torch version uses GaussianDistribution for both component sets
in place of the more complex CompositeDistribution and SequenceDistribution
components from the stats reference, for simplicity.
"""

import torch

from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    n = int(1e4)
    # Define the model: 3 components for X1 and 3 components for X2
    d11 = GaussianDistribution(mu=-6.0, sigma2=1.0)
    d12 = GaussianDistribution(mu=0.0, sigma2=1.0)
    d13 = GaussianDistribution(mu=6.0, sigma2=1.0)

    d21 = GaussianDistribution(mu=-4.0, sigma2=1.0)
    d22 = GaussianDistribution(mu=0.0, sigma2=1.0)
    d23 = GaussianDistribution(mu=4.0, sigma2=1.0)

    taus12 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    taus21 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w1 = [0.6, 0.3, 0.1]
    w2 = [0.7, 0.2, 0.1]

    dist = JointMixtureDistribution(
        components1=[d11, d12, d13],
        components2=[d21, d22, d23],
        w1=w1,
        w2=w2,
        taus12=taus12,
        taus21=taus21,
    )
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est1 = GaussianEstimator()
    est2 = GaussianEstimator()
    est = JointMixtureEstimator(
        [est1] * 3, [est2] * 3, pseudo_count=(0.001, 0.001, 0.001)
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
    for x, y in zip(data[:5], ll[:5]):
        print(f"Obs {x}, Likelihood {y}")

    # Check model device and move it to cpu (or some other device if preferred)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
