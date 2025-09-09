"""Test cases for Log Gaussian Distribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.log_gaussian import *
import numpy as np
import pytest 

class LogGaussianDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self.eval_dists = [
            LogGaussianDistribution(mu=0.0, sigma2=1.0, name='name', keys='keys'),
            LogGaussianDistribution(mu=0.0, sigma2=1.0, keys='keys'),
            LogGaussianDistribution(mu=0.0, sigma2=1.0, name='name'),
            LogGaussianDistribution(mu=0.0, sigma2=1.0)
        ]
        self._ests = [
            LogGaussianEstimator(name='name', keys='keys'),
            LogGaussianEstimator(keys='keys'),
            LogGaussianEstimator(name='name'),
            LogGaussianEstimator()
        ]
        self._factories = [
            LogGaussianAccumulatorFactory(name='name', keys='keys'),
            LogGaussianAccumulatorFactory(keys='keys'),
            LogGaussianAccumulatorFactory(name='name'),
            LogGaussianAccumulatorFactory()
        ]
        self._accumulators = [
            LogGaussianAccumulator(name='name', keys='keys'),
            LogGaussianAccumulator(keys='keys'),
            LogGaussianAccumulator(name='name'),
            LogGaussianAccumulator()
        ]
        self._encoders = [LogGaussianDataEncoder()]*len(self.eval_dists)

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
            assert str(e.value) == "LogGaussianEncodedDataSequence required for seq_log_density()."

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                LogGaussianEstimator(keys=x)
                
            assert str(e.value) == "LogGaussianEstimator requires keys to be of type 'str'."
