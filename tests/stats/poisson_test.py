"""Test cases for Poisson Distribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.poisson import *
import numpy as np
import pytest 

class PoissonDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            PoissonDistribution(lam=1.0, name='p', keys='key'),
            PoissonDistribution(lam=10.0, keys='key'),
            PoissonDistribution(lam=1.0, name='p'),
            PoissonDistribution(lam=5.0)
        ]
        self._ests = [
            PoissonEstimator(name='p', keys='key'),
            PoissonEstimator(keys='key'),
            PoissonEstimator(name='p'),
            PoissonEstimator()
        ]
        self._factories = [
            PoissonAccumulatorFactory(name='p', keys='key'),
            PoissonAccumulatorFactory(keys='key'),
            PoissonAccumulatorFactory(name='p'),
            PoissonAccumulatorFactory()
        ]
        self._accumulators = [
            PoissonAccumulator(name='p', keys='key'),
            PoissonAccumulator(keys='key'),
            PoissonAccumulator(name='p'),
            PoissonAccumulator()
        ]
        self._encoders = [PoissonDataEncoder()]*len(self.eval_dists)

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
            assert str(e.value) == "PoissonEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                PoissonEstimator(keys=x)
                
            assert str(e.value) == "PoissonEstimator requires keys to be of type 'str'."