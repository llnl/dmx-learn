"""Example for ConditionalDistribution. Define distribution, generate data,
estimate, and evaluate likelihoods."""
from dmx.stats import *
from dmx.utils.estimation import optimize
from numpy.random import RandomState

if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    d1 = GaussianDistribution(1.0, 1.0)
    d2 = GaussianDistribution(2.0, 1.0)
    d3 = GaussianDistribution(3.0, 1.0)

    d0 = CategoricalDistribution({'a': 0.5, 'b': 0.2, 'c': 0.2, 'd': 0.1})

    dist = ConditionalDistribution(
            dmap={'a': d1, 'b': d2},
            default_dist=d3,
            given_dist=d0)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = GaussianEstimator()
    est1 = CategoricalEstimator()
    emap = {'a': est0, 'b': est0}
    est = ConditionalDistributionEstimator(
            estimator_map=emap,
            default_estimator=est0,
            given_estimator=est1)
    # Estimate Model
    model = optimize(data, est, max_its=100, rng=rng, print_iter=1)
    print(str(model))
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data: {ll}')
