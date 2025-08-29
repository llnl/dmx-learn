"""Example for HeterogeneousMixtureDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.

HeterogeneousMixtureDistribution allows for a mixture with differnt
distribtions as the components. The comps must have the same support.

""" 
from numpy.random import RandomState
from dmx.stats import *
from dmx.utils.estimation import optimize


if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    w = [0.5, 0.5]
    dist0 = PoissonDistribution(lam=2.0)
    dist1 = BinomialDistribution(n=20, p=0.50)
    dist = HeterogeneousMixtureDistribution(
            components=[dist0, dist1],
            w=w)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = PoissonEstimator()
    est1 = BinomialEstimator()
    est = HeterogeneousMixtureEstimator(estimators=[est0, est1])
    # Estimate model
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
    # Fast evaluation of the posterior
    post = model.seq_posterior(enc_data)
    print(f'Posterior of each observed point: {post}')
