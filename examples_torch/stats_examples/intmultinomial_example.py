"""IntegerMultinomialDistribution example on generated data."""
import numpy as np
from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == '__main__':
    n = int(1e4)
    # Define the model
    len_pvec = np.array([0.5] + [0.0] * 9 + [0.5])  # support {10, 20}
    len_dist = IntegerCategoricalDistribution(min_val=10, p_vec=len_pvec)
    p = np.ones(10) / 10.0
    dist = IntegerMultinomialDistribution(
            min_val=0,
            p_vec=p,
            len_dist=len_dist)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Define estimator
    len_est = IntegerCategoricalEstimator()
    est = IntegerMultinomialEstimator(len_estimator=len_est)
    # Estimate model
    model = optimize(
            data=data,
            estimator=est,
            max_its=1000,
            seed=1,
            print_iter=100,
            device=device)
    print(model)
    # Eval likelihood on a an observation
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs: {str(x)}, Likelihood: {y}.')
