"""Example for HiddenAssociationDistribution. Define distribution,
generate data, estimate, and evaluate likelihoods.
""" 
from numpy.random import RandomState
import numpy as np
from dmx.stats import *
from dmx.utils.estimation import optimize


if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    aa = 0.90
    bb = (1.0 - aa) / 2

    dist1 = CategoricalDistribution({'a': aa, 'b': bb, 'c': bb})
    dist2 = CategoricalDistribution({'a': bb, 'b': aa, 'c': bb})
    dist3 = CategoricalDistribution({'a': bb, 'b': bb, 'c': aa})
    cond_dist = ConditionalDistribution({'a': dist1, 'b': dist2, 'c': dist3})
    given_dist = MultinomialDistribution(CategoricalDistribution({'a': 0.3, 'b': 0.2, 'c': 0.5}),
                                         len_dist=CategoricalDistribution({5: 1.0}))
    len_dist = CategoricalDistribution({7: 1.0})
    dist = HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist)
    
    # Generate data from sampler
    sampler = dist.sampler(1)
    data = sampler.sample(1000)
    # Print out a few samples
    for i, x in enumerate(data[:5]):
        print(f'Data obs {i}: {x}')
        
    # Define estimator
    len_est = CategoricalEstimator()
    given_est = MultinomialEstimator(CategoricalEstimator(), len_estimator=CategoricalEstimator())
    cond_est = ConditionalDistributionEstimator({v: CategoricalEstimator() for v in ['a', 'b', 'c']})
    est = HiddenAssociationEstimator(cond_est, given_estimator=given_est, len_estimator=len_est)
    # Estimate the model
    model = optimize(data, est, max_its=1000, print_iter=100, rng=np.random.RandomState(1))
    print(str(model))
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data[:5]: {ll[:5]}')

