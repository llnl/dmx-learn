"""Simulate data and estimate from a Mixture distribution.
This example demonstrates the use of keys for mixture comps 
as well as the use of composite mixtures. """
import numpy as np
from dmx.stats import *
from dmx.utils.estimation import optimize, partition_data

if __name__ == '__main__':
    rng = np.random.RandomState(1)

    # Create the example distribution (we'll sample data from this)
    # This is a 5 component composite. Note that the first component is a
    # mixture with components the same as d20! We will use keys for the
    # estimator later.

    d10 = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
    d11 = OptionalDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.4, 'c': 0.1}), p=0.1)
    d12 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.2, 'b': 0.8}, 'b': {'a': 0.8, 'b': 0.2}},
                                  len_dist=PoissonDistribution(8.0))
    d13 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d14 = MultivariateGaussianDistribution([-1.0, -1.0], [[2.0, 1.0], [1.0, 2.0]])

    # 5-component composite distribution for the first mixture component.
    d1 = CompositeDistribution([d10, d11, d12, d13, d14])

    # d20 has same components as d10, just different weights!
    d20 = MixtureDistribution([GaussianDistribution(-3.0, 1.0),
                               GaussianDistribution(0.0, 1.0)], [0.25, 0.75])
    d21 = OptionalDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}), p=0.2)
    d22 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.8, 'b': 0.2}, 'b': {'a': 0.2, 'b': 0.8}},
                                  len_dist=PoissonDistribution(8.0))
    d23 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d24 = MultivariateGaussianDistribution([1.0, 1.0], [[2.0, 1.0], [1.0, 2.0]])

    # 5-comp composite distirbution for second mixture component.
    d2 = CompositeDistribution([d20, d21, d22, d23, d24])

    # Define the two-component composite mixture.
    dist = MixtureDistribution([d1, d2], [0.5, 0.5])

    # Sample data from the distribution
    sampler = dist.sampler(seed=rng.randint(2 ** 31))
    data = sampler.sample(size=2000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # Specify the model estimator

    # Use regularization when initializing
    # Note: Since the first component of the composite is a mixture with the
    # same topics/components in each outer-mixture component (i.e. only weights
    # change), we add the keys=(None, 'comps0')!
    e0 = MixtureEstimator([GaussianEstimator()]*2, keys=(None, 'comps0'), pseudo_count=1.0)
    e1 = OptionalEstimator(CategoricalEstimator(pseudo_count=1.0), est_prob=False, pseudo_count=1.0)
    e2 = MarkovChainEstimator(pseudo_count=1.0, len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    iest = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, pseudo_count=1.0, fixed_weights=[0.2, 0.8])

    # And one without regularization
    # Don't forget the keys on the inner mixture!
    e0 = MixtureEstimator([GaussianEstimator()] * 2, keys=(None, 'comps0'))
    e1 = OptionalEstimator(CategoricalEstimator(), est_prob=False)
    e2 = MarkovChainEstimator(len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    est = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, fixed_weights=[0.2, 0.8])

    # Estimate parameters
    mm = optimize(train_data, est, max_its=100, print_iter=20, init_estimator=iest, rng=np.random.RandomState(1))
    
    # 
    print(f'Model weights: {mm.w.tolist()}')
    print(f'Mixture component 1: {mm.components[0]}')
    print(f'Mixture component 2: {mm.components[1]}')
    
    s0 = f"""Mixture component 1, Composite comp 1:
    {mm.components[0].dists[0]}"""
    print(s0)
    s1 = f"""Mixture component 2, Composite comp 1:
    {mm.components[1].dists[0]}"""
    print(s1)
    print('With the keys set, Only the weights differ!')
    # Encode data for vectorized function calls.
    enc_data = seq_encode(train_data, model=mm)[0][1]
    # Evaluate the likleihood vectorized 
    ll = mm.seq_log_density(enc_data)
    for x, y in zip(train_data[:5], ll[:5]):
        print(f'Obs {x}, LL {float(y)}')

    # Obtain posterior probs for mix-comps
    post = mm.seq_posterior(enc_data)
    for x, y in zip(train_data[:5], post[:5]):
        print(f'Obs {x}, Posterior {y.tolist()}')


