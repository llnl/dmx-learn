"""Generate and fit simple random graph mixtures with Bernoulli edge models."""

from typing import cast

import numpy as np

from dmx.stats import (
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
    MixtureDistribution,
    MixtureEstimator,
)
from dmx.utils.estimation import optimize

if __name__ == "__main__":
    p_mat = np.zeros((4, 4))
    p_mat[0, :] = [0.0, 0.8, 0.1, 0.8]
    p_mat[1, :] = [0.8, 0.0, 0.8, 0.1]
    p_mat[2, :] = [0.1, 0.8, 0.0, 0.8]
    p_mat[3, :] = [0.8, 0.1, 0.8, 0.0]

    p_vec = p_mat.flatten()
    log_pvec = np.zeros_like(p_vec)
    log_pvec.fill(-np.inf)
    log_pvec[p_vec != 0.0] = np.log(p_vec[p_vec != 0.0])

    dist1 = IntegerBernoulliSetDistribution(log_pvec)

    p_mat = np.zeros((4, 4))
    p_mat[0, :] = [0.0, 0.1, 0.8, 0.1]
    p_mat[1, :] = [0.1, 0.0, 0.1, 0.8]
    p_mat[2, :] = [0.8, 0.1, 0.0, 0.1]
    p_mat[3, :] = [0.1, 0.8, 0.1, 0.0]

    p_vec = p_mat.flatten()
    log_pvec = np.zeros_like(p_vec)
    log_pvec.fill(-np.inf)
    log_pvec[p_vec != 0.0] = np.log(p_vec[p_vec != 0.0])

    dist2 = IntegerBernoulliSetDistribution(log_pvec)
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    data = dist.sampler(1).sample(1000)

    est = MixtureEstimator([IntegerBernoulliSetEstimator(16)] * 2)

    model = cast(
        MixtureDistribution,
        optimize(data, est, max_its=100, rng=np.random.RandomState(1)),
    )

    comp0 = cast(IntegerBernoulliSetDistribution, model.components[0])
    comp1 = cast(IntegerBernoulliSetDistribution, model.components[1])
    print(list(np.exp(comp0.log_pvec)))
    print(list(np.exp(comp1.log_pvec)))
