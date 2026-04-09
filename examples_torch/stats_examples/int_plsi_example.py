"""Integer PLSI example on generated data.
Note: Model fit is significantly faster with numba use.
"""

import numpy as np

from dmx.torch_stats import *
from dmx.torch_utils import detect_device
from dmx.torch_utils.estimation import optimize

device = detect_device()

if __name__ == "__main__":
    rng = np.random.RandomState(1)
    # Define the model
    n_docs = 10000
    num_states = 3
    num_authors = 10
    num_words = 50

    state_word_mat = rng.dirichlet(alpha=np.ones(num_words), size=num_states).T
    doc_state_mat = rng.dirichlet(alpha=np.ones(num_states), size=num_authors)
    doc_vec = rng.dirichlet(alpha=np.ones(num_authors))
    len_pvec = np.array([0.25, 0.25, 0.0, 0.0, 0.0, 0.5])  # support {5, 6, 10}
    len_dist = IntegerCategoricalDistribution(min_val=5, p_vec=len_pvec)

    dist = IntegerPLSIDistribution(
        state_word_mat=state_word_mat,
        doc_state_mat=doc_state_mat,
        doc_vec=doc_vec,
        len_dist=len_dist,
    )
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n_docs)
    # Define estimator
    len_est = IntegerCategoricalEstimator()
    est = IntegerPLSIEstimator(
        num_vals=num_words,
        num_states=num_states,
        num_docs=num_authors,
        len_estimator=len_est,
    )
    # Estimate model
    model = optimize(
        data=data,
        estimator=est,
        init_p=0.10,
        max_its=1000,
        seed=1,
        print_iter=100,
        device=device,
    )

    # Eval likelihood on a an observation
    ll0 = model.log_density(data[0])
    print(f"Likelihood of estimated model eval at {data[0]}: {ll0}")
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likelihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f"Obs: {str(x)}, Likelihood: {y}.")

    # Check model device and move it to cpu (or some other device if prefered)
    print(f"\nEstimated model is on {model.model_device()}.\nMoving it to the cpu...")
    model.to(torch.device("cpu"))
