"""Test cases for Spike and Slab Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.int_spike import *
import numpy as np
import pytest 


def seq_update_test(dist, encoder, est):
    seeds = [1, 2, 3]
    sz = 1000
    rv = []
    for seed in seeds:
        data = dist.sampler(seed=seed).sample(sz)
        enc_data = [(sz, encoder.seq_encode(data))]
        rng = np.random.RandomState(seed)
        prev_estimate = seq_initialize(enc_data, est, rng)
        estimate = seq_estimate(enc_data, est, prev_estimate)

        ll_prev = np.sum(prev_estimate.seq_log_density(enc_data[0][1]))
        ll = np.sum(estimate.seq_log_density(enc_data[0][1]))
        log_diff = ll-ll_prev
        rv.append(log_diff > 0)

    return np.all(rv), rv

class SpikeAndSlabDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            SpikeAndSlabDistribution(k=4, num_vals=10, p=0.80, min_val=0, name='name', keys='keys'),
            SpikeAndSlabDistribution(k=4, num_vals=10, p=0.80, min_val=3, name='name'),
            SpikeAndSlabDistribution(k=4, num_vals=10, p=0.80, keys='keys'),
            SpikeAndSlabDistribution(k=4, num_vals=10, p=0.80)
        ]
        self._ests = [
            SpikeAndSlabEstimator(min_val=0, max_val=10, name='name', keys='keys'),
            SpikeAndSlabEstimator(min_val=3, max_val=13, name='name'),
            SpikeAndSlabEstimator(min_val=0, max_val=10, keys='keys'),
            SpikeAndSlabEstimator(min_val=0, max_val=10)
        ]
        self._factories = [
            SpikeAndSlabAccumulatorFactory(min_val=0, max_val=10, name='name', keys='keys'),
            SpikeAndSlabAccumulatorFactory(min_val=3, max_val=13, name='name'),
            SpikeAndSlabAccumulatorFactory(min_val=0, max_val=10, keys='keys'),
            SpikeAndSlabAccumulatorFactory(min_val=0, max_val=10)
        ]
        self._accumulators = [
            SpikeAndSlabAccumulator(min_val=0, max_val=10, name='name', keys='keys'),
            SpikeAndSlabAccumulator(min_val=3, max_val=13, name='name'),
            SpikeAndSlabAccumulator(min_val=0, max_val=10, keys='keys'),
            SpikeAndSlabAccumulator(min_val=0, max_val=10)
        ]
        self._encoders = [SpikeAndSlabDataEncoder()]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]

        self._init_ests = [
            SpikeAndSlabEstimator(min_val=0, max_val=10, pseudo_count=1.0e-10),
            SpikeAndSlabEstimator(min_val=3, max_val=13, pseudo_count=1.0e-10),
            SpikeAndSlabEstimator(min_val=0, max_val=10, pseudo_count=1.0e-10),
            SpikeAndSlabEstimator(min_val=0, max_val=10, pseudo_count=1.0e-10)
        ]

        self.type_check_keys = [(None, None), 1.0, ('keys', None)]
        
    @pytest.mark.dependency(depends=["estimator", "log_density", "estimator_factory", "factory_make"])
    def test_09_seq_update(self):
        for x in zip(self.eval_dists, self._encoders, self._init_ests):
            res = seq_update_test(*x)
            self.assertTrue(res[0], str(res[1]))

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "SpikeAndSlabEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                SpikeAndSlabEstimator(keys=x)
                
            assert str(e.value) == "SpikeAndSlabEstimator requires keys to be of type 'str'."
