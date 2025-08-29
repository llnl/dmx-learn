"""Test cases for Von Mises Fisher Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.vmf import *
import numpy as np
import pytest 

class VonMisesFisherDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self.eval_dists = [
            VonMisesFisherDistribution(mu=np.ones(3) / 3., kappa=1.0, name='name', keys='keys'),
            VonMisesFisherDistribution(mu=np.ones(3) / 3., kappa=1.0, keys='keys'),
            VonMisesFisherDistribution(mu=np.ones(4) / 4., kappa=0.10),
        ]
        self._ests = [
            VonMisesFisherEstimator(dim=3, name='name', keys='keys'),
            VonMisesFisherEstimator(dim=3, keys='keys'),
            VonMisesFisherEstimator(dim=4)
        ]
        self._factories = [
            VonMisesFisherAccumulatorFactory(dim=3, name='name', keys='keys'),
            VonMisesFisherAccumulatorFactory(dim=3, keys='keys'),
            VonMisesFisherAccumulatorFactory(dim=4)
        ]
        self._accumulators = [
            VonMisesFisherAccumulator(dim=3, name='name', keys='keys'),
            VonMisesFisherAccumulator(dim=3, keys='keys'),
            VonMisesFisherAccumulator(dim=4)
        ]
        self._encoders = [VonMisesFisherDataEncoder()]*len(self.eval_dists)

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
            assert str(e.value) == "VonMisesFisherEncodedDataSequence required for seq_log_density()."

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                VonMisesFisherEstimator(keys=x)
                
            assert str(e.value) == "VonMisesFisherEstimator requires keys to be of type 'str'."
