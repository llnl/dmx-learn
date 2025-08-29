"""Tests for the Geometric distribution and its related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.geometric import *
import numpy as np
import pytest 

class GeometricDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            GeometricDistribution(p=0.30, name='name', keys='key'),
            GeometricDistribution(p=0.30, keys='key'),
            GeometricDistribution(p=0.30, name='name'),
            GeometricDistribution(p=0.20)
        ]
        self._ests = [
            GeometricEstimator(name='name', keys='key'),
            GeometricEstimator(keys='key'),
            GeometricEstimator(name='name'),
            GeometricEstimator()
        ]
        self._factories = [
            GeometricAccumulatorFactory(name='name', keys='key'),
            GeometricAccumulatorFactory(keys='key'),
            GeometricAccumulatorFactory(name='name'),
            GeometricAccumulatorFactory()
        ]
        self._accumulators = [
            GeometricAccumulator(name='name', keys='key'),
            GeometricAccumulator(keys='key'),
            GeometricAccumulator(name='name'),
            GeometricAccumulator()
        ]
        self._encoders = [GeometricDataEncoder()]*len(self.eval_dists)

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
                
            assert str(e.value) == "GeometricEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                GeometricEstimator(keys=x)
                
            assert str(e.value) == "GeometricEstimator requires keys to be of type 'str'."
