import os
import numpy as np
import torch as tn
from dmx.torch_stats import *

# import optimize from torch_utils NOT utils
from dmx.torch_utils.estimation import optimize

device = tn.device('cuda:0') if tn.cuda.is_available() else tn.device('cpu')

if __name__ == "__main__":
    rng = np.random.RandomState(1)
    num_states = 5
    N = 1000

    # declare HMM as example
    topics = [GaussianDistribution(mu=x, sigma2=1.0) for x in np.linspace(-20*num_states, 20*num_states, num_states)]
    w = np.ones(num_states) / num_states
    transitions = np.zeros((num_states, num_states))
    p = 0.85
    transitions.fill((1.0 - p) / (num_states-1))
    np.fill_diagonal(transitions, p)

    # sample data
    len_dist = IntegerCategoricalDistribution(min_val=10, p_vec=np.ones(1) / 1.0)
    d = HiddenMarkovModelDistribution(topics=topics, w=w, transitions=transitions, len_dist=len_dist)
    data = d.sampler(1).sample(N)

    # declare estimator
    est = HiddenMarkovEstimator([GaussianEstimator()] * num_states, len_estimator=IntegerCategoricalEstimator())

    # fit on cuda device (if detected)
    fit = optimize(data, estimator=est, seed=1, init_p=1.0, max_its=10, device=device)
    print(fit)


