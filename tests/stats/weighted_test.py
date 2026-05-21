"""Test cases for Weighted Distribution and related classes."""

# pylint: disable=duplicate-code,wildcard-import,unused-wildcard-import,line-too-long
# pylint: disable=too-many-instance-attributes,unnecessary-comprehension
# pylint: disable=redefined-builtin,broad-exception-caught

import numpy as np
import pytest

from dmx.stats import *
from dmx.stats.intrange import *
from dmx.stats.weighted import *
from tests.stats.stats_tests import *


def weighted_log_density_test(dist, encoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        count_data = {}
        for x in data:
            count_data[x] = count_data.get(x, 0) + 1
        count_data = list(count_data.items())

        try:
            enc_data = encoder.seq_encode(count_data)
        except Exception:
            return False, "encoder.seq_encode(data)"

        seq_ll = dist.seq_log_density(enc_data)
        for i, item in enumerate(count_data):
            if seq_ll[i] == 0:
                seq_ll[i] = np.abs(dist.log_density(item))
            else:
                seq_ll[i] = np.abs(seq_ll[i] - dist.log_density(item)) / np.abs(
                    seq_ll[i]
                )

        rv.append(max(seq_ll))
    return max(rv) < 1.0e-14, "max(rv) test"


def weighted_seq_update_test(dist, estimator, encoder):
    seeds = [1, 2, 3]
    sz = 1000
    rv = []
    for seed in seeds:
        data = dist.sampler(seed=seed).sample(sz)
        count_data = {}
        for x in data:
            count_data[x] = count_data.get(x, 0) + 1
        count_data = list(count_data.items())

        enc_data = [(len(count_data), encoder.seq_encode(count_data))]

        rng = np.random.RandomState(seed)
        prev_estimate = seq_initialize(enc_data, estimator, rng)
        updated_estimate = seq_estimate(enc_data, estimator, prev_estimate)

        ll_prev = np.sum(prev_estimate.seq_log_density(enc_data[0][1]))
        ll = np.sum(updated_estimate.seq_log_density(enc_data[0][1]))
        log_diff = ll - ll_prev
        rv.append(log_diff >= 0)

    return np.all(rv), rv


class WeightedDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        dist = IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(3) / 3.0)
        est = IntegerCategoricalEstimator()
        fac = IntegerCategoricalAccumulatorFactory()
        acc = IntegerCategoricalAccumulator()
        enc = IntegerCategoricalDataEncoder()

        keys = ["keys", "keys", None, None]
        name = ["name" if i % 2 == 0 else None for i in range(4)]
        self.eval_dists = []
        self._ests = []
        self._factories = []
        self._accumulators = []

        for i in range(4):
            self.eval_dists.append(
                WeightedDistribution(dist=dist, name=name[i], keys=keys[i])
            )
            self._ests.append(
                WeightedEstimator(estimator=est, name=name[i], keys=keys[i])
            )
            self._factories.append(
                WeightedAccumulatorFactory(factory=fac, name=name[i], keys=keys[i])
            )
            self._accumulators.append(
                WeightedAccumulator(accumulator=acc, name=name[i], keys=keys[i])
            )

        self._encoders = [WeightedDataEncoder(encoder=enc)] * len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.seq_update_configs = [
            (
                WeightedDistribution(
                    dist=IntegerCategoricalDistribution(0, np.ones(20) / 20.0)
                ),
                WeightedEstimator(
                    IntegerCategoricalEstimator(
                        suff_stat=[0, np.ones(20)], pseudo_count=1.0e-6
                    )
                ),
                WeightedDataEncoder(encoder=enc),
            )
        ]

    @pytest.mark.dependency(depends=["sampler"], name="log_density")
    def test_05_log_density(self):
        for x in self.density_dist_encoder:
            res = weighted_log_density_test(*x)
            self.assertTrue(res[0], str(res[1]))

    @pytest.mark.dependency(
        depends=["estimator", "log_density", "estimator_factory", "factory_make"]
    )
    def test_09_seq_update(self):
        for x in self.seq_update_configs:
            res = weighted_seq_update_test(*x)
            self.assertTrue(res[0], str(res[1]))

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert (
                str(e.value)
                == "WeightedEncodedDataSequence required for seq_log_density()."
            )
