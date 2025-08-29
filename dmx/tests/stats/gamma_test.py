"""Tests for the Gamma distribution and its related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.gamma import *
import numpy as np
import pytest 

class GammaDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            GammaDistribution(k=2.0, theta=1.0, name='name', keys='key'),
            GammaDistribution(k=1.0, theta=1.0, keys='key'),
            GammaDistribution(k=4.0, theta=3.0, name='name'),
            GammaDistribution(k=2.0, theta=2.0)
        ]
        self._ests = [
            GammaEstimator(name='name', keys='key'),
            GammaEstimator(keys='key'),
            GammaEstimator(name='name'),
            GammaEstimator()
        ]
        self._factories = [
            GammaAccumulatorFactory(name='name', keys='key'),
            GammaAccumulatorFactory(keys='key'),
            GammaAccumulatorFactory(name='name'),
            GammaAccumulatorFactory()
        ]
        self._accumulators = [
            GammaAccumulator(name='name', keys='key'),
            GammaAccumulator(keys='key'),
            GammaAccumulator(name='name'),
            GammaAccumulator()
        ]
        self._encoders = [GammaDataEncoder()]*len(self.eval_dists)

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
                
            assert str(e.value) == "GammaEncodedDataSequence required for seq_log_density()."

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                GammaEstimator(keys=x)
                
            assert str(e.value) == "GammaEstimator requires keys to be of type 'str'."
