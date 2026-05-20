"""Integer categorical example.

Generate synthetic categorical integers, fit an integer categorical model, and
evaluate log densities.
"""

# pylint: disable=duplicate-code

import numpy as np
import torch

from dmx.torch_stats import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    seq_encode,
)
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    n = int(1e4)
    # Define the model
    p = np.ones(10) / 10.0
    dist = IntegerCategoricalDistribution(min_val=0, p_vec=p)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Define estimator
    est = IntegerCategoricalEstimator()
    # Estimate model
    model = optimize(
        data=data, estimator=est, max_its=1000, seed=1, print_iter=100, device=device
    )
    print(model)
    # Eval likelihood on a an observation
    ll0 = model.log_density(data[0])
    print(f"Likelihood of estimated model eval at {data[0]}: {ll0}")
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f"Obs: {str(x)}, Likelihood: {y}.")

    # Check model device and move it to cpu (or some other device if prefered)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
