"""Example for OptionalDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.
""" 
import numpy as np
from dmx.stats import *
from dmx.utils.estimation import optimize

if __name__ == '__main__':
    rng = np.random.RandomState(1)
    pmap={'a': 0.5, 'b': 0.4, 'c': 0.1}
    d = CategoricalDistribution(pmap=pmap)
    # default missing value is None
    dist0 = OptionalDistribution(d, p=0.7)
    # we can also set missing_value (e.g. np.nan)
    dist1 = OptionalDistribution(d, p=0.7, missing_value=np.nan)
    print(dist1)
    # Generate samples from both
    sampler0 = dist0.sampler(1)
    sampler1 = dist1.sampler(1)

    data0 = sampler0.sample(1000)
    data1 = sampler1.sample(1000)

    
    for x in data0[:5]:
        print(f'Missing is {dist0.missing_value}: Obs {x}')
    for x in data1[:5]:
        print(f'Missing is {dist1.missing_value}: Obs {x}')

    # Estimator
    est0 = CategoricalEstimator()
    est1 = OptionalEstimator(est0)
    est2 = OptionalEstimator(est0, missing_value=np.nan)

    # Estimate model
    m0 = optimize(data0, est1, rng=rng)
    m1 = optimize(data1, est2, rng=rng)

    print(m0)
    print(m1)

