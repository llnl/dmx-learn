"""Example for DiracMixtureDistribution. Define distribution, generate data,
estimate, and evaluate likelihoods.

This distribution is a mixture between a length distribution and a Dirac Delta
on a user-specified value. It is a great for use on length distributions that
are zero-inflated. 

f(x) = p*F(X=x) + (1-p) * Delta(v)

"""
from dmx.stats import *
from dmx.utils.estimation import optimize
from numpy.random import RandomState

if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    len_dist = PoissonDistribution(lam=10.0)
    dist = DiracMixtureDistribution(
            dist=len_dist,
            p=0.80,
            v=0)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est0 = PoissonEstimator()
    est = DiracMixtureEstimator(
            estimator=est0,
            v=0)
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
