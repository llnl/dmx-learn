"""Example for MarkovChainDistribution. Define distribution 
generate data, estimate, and evaluate likelihoods.

""" 
from numpy.random import RandomState
from dmx.stats import *
from dmx.utils.estimation import optimize

if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the initial state probabilities 
    vals = ['a', 'b', 'c', 'd', 'e']
    pi = rng.dirichlet(alpha=[1.]*len(vals)).tolist()
    init_prob_map = {k: v for k, v in zip(vals, pi)}
    print(f'Initial state map: {init_prob_map}')
    # Define the state transition map
    trans_map = {v: {} for v in vals}
    for x in vals:
        w = rng.dirichlet(alpha=[1.]*len(vals)).tolist()
        trans_map[x] = {k: v for k, v in zip(vals, w)}
    print(f'Transition map: {trans_map}')
    # Define the Markov chain model with length dist
    len_dist = CategoricalDistribution({3: 0.5, 5: 0.5})
    dist = MarkovChainDistribution(
            init_prob_map=init_prob_map,
            transition_map=trans_map,
            len_dist=len_dist)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    len_est = CategoricalEstimator()
    est = MarkovChainEstimator(len_estimator=len_est)
    # Estimate model
    model = optimize(data, est, max_its=100, rng=rng, print_iter=1)
    print(str(model))
    print('\n')
    print(model.transition_map)
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs {x}, LL {y}')
