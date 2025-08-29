"""Integer PLSI example on generated data.
Note: Model fit is significantly faster with numba use.
"""
import numpy as np
from dmx.stats import *
from dmx.utils.estimation import optimize

if __name__ == '__main__':
    rng = np.random.RandomState(1)
    # Define the model
    n_docs = 10000
    num_states = 3
    num_authors = 10
    num_words = 50

    state_word_mat = rng.dirichlet(alpha=np.ones(num_words), size=num_states).T
    doc_state_mat = rng.dirichlet(alpha=np.ones(num_states), size=num_authors)
    doc_vec = rng.dirichlet(alpha=np.ones(num_authors))
    pmap = {5: 0.25, 6: 0.25, 10: 0.5}
    len_dist = CategoricalDistribution(pmap=pmap)

    dist = IntegerPLSIDistribution(
            state_word_mat=state_word_mat,
            doc_state_mat=doc_state_mat,
            doc_vec=doc_vec,
            len_dist=len_dist)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n_docs)
    # Define estimator
    len_est = CategoricalEstimator()
    est = IntegerPLSIEstimator(
            num_vals=num_words,
            num_states=num_states,
            num_docs=num_authors,
            len_estimator=len_est)
    # Estimate model
    model = optimize(
            data=data,
            estimator=est,
            init_p=0.10,
            rng=rng,
            max_its=1000,
            print_iter=100)

    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs: {str(x)}, Likelihood: {y}.')

