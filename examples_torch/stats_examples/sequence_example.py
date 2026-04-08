"""Example for SequenceDistribution. Define distribution,
generate data, estimate, and evaluate likelihoods.

Note: The torch version uses IntegerCategoricalDistribution as the length
distribution in place of CategoricalDistribution (not available in torch_stats).
"""
import torch
import numpy as np
from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == '__main__':
    n = int(1e4)
    # Define the model, len_normalized corrects for length of observed sequences
    d = ExponentialDistribution(beta=1.0)
    len_pvec = np.array([0.5, 0.0, 0.5])  # support {3, 5} (min_val=3)
    len_dist = IntegerCategoricalDistribution(min_val=3, p_vec=len_pvec)
    dist = SequenceDistribution(
            dist=d,
            len_dist=len_dist,
            len_normalized=True)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = ExponentialEstimator()
    len_est = IntegerCategoricalEstimator()
    est = SequenceEstimator(
            estimator=est0,
            len_estimator=len_est,
            len_normalized=True)
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
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs {x}, LL {float(y)}')

    # Check model device and move it to cpu (or some other device if preferred)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
