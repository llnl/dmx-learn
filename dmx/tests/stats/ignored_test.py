"""Test cases for IgnoredDistribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.geometric import *
from dmx.stats.gaussian import *
from dmx.stats.ignored import *
import numpy as np

class IgnoredDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self._dists = [
            GaussianDistribution(mu=1.0, sigma2=1.0),
            GeometricDistribution(p=0.30)
        ]
        self._ests = [
            GaussianEstimator(),
            GeometricEstimator()
        ]
        self._facts = [
            GaussianAccumulatorFactory(),
            GeometricAccumulatorFactory()
        ]
        self._accs = [
            GaussianAccumulator(),
            GeometricAccumulator()
        ]
        self._encs = [
            GaussianDataEncoder(),
            GeometricDataEncoder()
        ]


        self.eval_dists = [
            IgnoredDistribution(dist=self._dists[0], name='name', keys='keys'),
            IgnoredDistribution(dist=self._dists[0]),
            IgnoredDistribution(dist=self._dists[1], name='name', keys='keys')
        ]
        self._estimators = [
            IgnoredEstimator(dist=self._dists[0], keys='keys', name='name'),
            IgnoredEstimator(dist=self._dists[0]),
            IgnoredEstimator(dist=self._dists[1], keys='keys', name='name')
        ]
        self._factories = [
            IgnoredAccumulatorFactory(encoder=self._encs[0], name='name', keys='keys'),
            IgnoredAccumulatorFactory(encoder=self._encs[0]),
            IgnoredAccumulatorFactory(encoder=self._encs[1], name='name', keys='keys')
        ]
        self._accumulators = [
            IgnoredAccumulator(encoder=self._encs[0], name='name', keys='keys'),
            IgnoredAccumulator(encoder=self._encs[0]),
            IgnoredAccumulator(encoder=self._encs[1], name='name', keys='keys')
        ]
        self._encoders = [
            IgnoredDataEncoder(encoder=self._encs[0]),
            IgnoredDataEncoder(encoder=self._encs[0]),
            IgnoredDataEncoder(encoder=self._encs[1])
        ]

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._estimators)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._estimators, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

    def test_09_seq_update(self):
        # there is no seq update for this class
        assert True
        

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                IgnoredEstimator(keys=x)
                
            assert str(e.value) == "IgnoredEstimator requires keys to be of type 'str'."






