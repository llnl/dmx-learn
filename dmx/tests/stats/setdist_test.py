"""Test cases for Bernoulli Set Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.setdist import *
import numpy as np
import pytest 


class BernoulliSetDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            BernoulliSetDistribution(pmap={'a': 0.3, 'b': 0.50, 'c': 0.20}, min_prob=1.0e-16, name='name', keys='keys'),
            BernoulliSetDistribution(pmap={0: 0.5, 10: 0.5}, keys='keys'),
            BernoulliSetDistribution(pmap={'a':0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25})
        ]
        self._ests = [
            BernoulliSetEstimator(min_prob=1.0e-16, name='name', keys='keys'),
            BernoulliSetEstimator(keys='keys'),
            BernoulliSetEstimator(),
        ]
        self._factories = [
            BernoulliSetAccumulatorFactory(keys='keys', name='name'),
            BernoulliSetAccumulatorFactory(keys='keys'),
            BernoulliSetAccumulatorFactory()
        ]
        self._accumulators = [
            BernoulliSetAccumulator(keys='keys', name='name'),
            BernoulliSetAccumulator(keys='keys'),
            BernoulliSetAccumulator()
        ]
        self._encoders = [BernoulliSetDataEncoder()]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

        self._init_ests = [
        ]
    

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "BernoulliSetEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                BernoulliSetEstimator(keys=x)
                
            assert str(e.value) == "BernoulliSetEstimator requires keys to be of type 'str'."