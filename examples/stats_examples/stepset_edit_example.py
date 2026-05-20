"""Fit a stepwise Bernoulli set edit model and inspect transition rates."""

# pylint: disable=duplicate-code

import numpy as np

from dmx.stats import (
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEstimator,
    MixtureDistribution,
    MixtureEstimator,
)
from dmx.utils.estimation import optimize

if __name__ == "__main__":

    # P(set_1)
    d0 = IntegerBernoulliSetDistribution(np.log([0.3, 0.3, 0.3]))
    # P(set_2 | set_1, Z=0)
    dist1 = IntegerStepBernoulliEditDistribution(
        np.log([[0.01, 0.5, 0.5], [0.05, 0.9, 0.9]]).T, init_dist=d0
    )
    # P(set_2 | set_1, Z=1)
    dist2 = IntegerStepBernoulliEditDistribution(
        np.log([[0.5, 0.01, 0.01], [0.9, 0.05, 0.05]]).T, init_dist=d0
    )

    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])
    data = dist.sampler(1).sample(5000)

    est0 = IntegerBernoulliSetEstimator(3, pseudo_count=1.0, keys="init_prob")
    est = MixtureEstimator(
        [IntegerStepBernoulliEditEstimator(3, pseudo_count=1.0, init_estimator=est0)]
        * 2
    )

    model = optimize(
        data,
        est,
        max_its=100,
        init_p=0.10,
        rng=np.random.RandomState(1),
        delta=None,
        print_iter=1,
    )

    for i, m in enumerate(model.components):
        print(str(np.exp(m.init_dist.log_pvec)))
        print(f"P(Missing | Missing, Z={i}) = {np.exp(m.log_edit_pmat[:, 0])}")
        print(f"P(Missing | Present, Z={i}) = {np.exp(m.log_edit_pmat[:, 1])}")
        print(f"P(Present | Missing, Z={i}) = {np.exp(m.log_edit_pmat[:, 2])}")
        print(f"P(Present | Present, Z={i}) = {np.exp(m.log_edit_pmat[:, 3])}")
