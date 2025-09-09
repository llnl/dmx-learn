"""Tests for the Exponential distribution and its related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.exponential import *
import numpy as np
import pytest 

class ExponentialDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            ExponentialDistribution(beta=1.0, name='name', keys='key'),
            ExponentialDistribution(beta=10.0, keys='key'),
            ExponentialDistribution(beta=1.0, name='name'),
            ExponentialDistribution(beta=5.0)
        ]
        self._ests = [
            ExponentialEstimator(name='name', keys='key'),
            ExponentialEstimator(keys='key'),
            ExponentialEstimator(name='name'),
            ExponentialEstimator()
        ]
        self._factories = [
            ExponentialAccumulatorFactory(name='name', keys='key'),
            ExponentialAccumulatorFactory(keys='key'),
            ExponentialAccumulatorFactory(name='name'),
            ExponentialAccumulatorFactory()
        ]
        self._accumulators = [
            ExponentialAccumulator(name='name', keys='key'),
            ExponentialAccumulator(keys='key'),
            ExponentialAccumulator(name='name'),
            ExponentialAccumulator()
        ]
        self._encoders = [ExponentialDataEncoder()]*len(self.eval_dists)

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

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
                
            assert str(e.value) == "ExponentialEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                ExponentialEstimator(keys=x)
                
            assert str(e.value) == "ExponentialEstimator requires keys to be of type 'str'."