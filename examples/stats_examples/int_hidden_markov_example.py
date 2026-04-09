"""Example of Integer HMM sampling and estimation using 'best_of' to find optimal model fit with different initial conditions.
Note that Numba should be used. """

import numpy as np

from dmx.stats import *
from dmx.utils.estimation import best_of, partition_data

rng = np.random.RandomState(2)

if __name__ == "__main__":
    n = int(1e3)

    num_states = 5
    num_words = 100

    ## Define the model

    # initial state distribution
    init_w = rng.dirichlet(alpha=np.ones(num_states) / num_states)

    # transition density
    p = 0.70
    trans = np.full((num_states, num_states), (1 - p) / (num_states - 1), dtype=float)
    np.fill_diagonal(trans, p)

    # Vocab probs P(W=w| Z=z) should be num words by num states
    pmat = rng.dirichlet(alpha=np.ones(num_words) / num_words, size=num_states).T

    # length distribution for sequences
    len_dist = CategoricalDistribution({5: 1.0})

    # Define the dist
    dist = IntegerHiddenMarkovModelDistribution(
        pmat=pmat, w=init_w, transitions=trans, len_dist=len_dist
    )

    ## Generate data
    sampler = dist.sampler(seed=rng.randint(2**32 - 1))
    data = sampler.sample(n)
    print(data[:10])

    ## Create an initial estimator
    len_est = CategoricalEstimator()
    # regularize the init states, transitions, and word distribution
    iest = IntegerHiddenMarkovEstimator(
        num_words=num_words,
        num_states=num_states,
        len_estimator=len_est,
        pseudo_count=(1.0, 1.0, 1.0),
    )

    # Create the estimator
    est = IntegerHiddenMarkovEstimator(
        num_words=num_words, num_states=num_states, len_estimator=len_est
    )
    train_data, valid_data = partition_data(data, [0.9, 0.1], rng=rng)

    # Fit model, finding the best model
    ll, mm = best_of(
        data=train_data,
        vdata=valid_data,
        est=est,
        trials=5,
        max_its=25,
        init_p=0.10,
        delta=1.0e-8,
        rng=rng,
        init_estimator=iest,
    )

    # Eval likelihood on an observed value
    ll0 = mm.log_density(train_data[0])
    print(f"Likelihood of estimated model at {train_data[0]}: {ll0}")
    # Encode data vectorized calls
    enc_data = seq_encode(train_data, model=mm)[0][1]
    # Eval the likelihood at all data points
    ll = mm.seq_log_density(enc_data)
    print(f"Likelihood of estimated model on data: {ll[:5]}")
