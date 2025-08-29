"""Tests for the Gaussian distribution and its related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.gaussian import *
import numpy as np
import pytest 

class GaussianDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self.eval_dists = [
                GaussianDistribution(mu=0.0, sigma2=1.0, name='g'),
                GaussianDistribution(mu=0.0, sigma2=1.0),
        ]
        self.dist_est = [
            (GaussianDistribution(mu=0.0, sigma2=1.0, name='g'), GaussianEstimator(name='g')),
            (GaussianDistribution(mu=0.0, sigma2=1.0), GaussianEstimator())
        ]
        self.dist_encoder = [(self.eval_dists[0], GaussianDataEncoder())]
        self.sampler_dist = self.eval_dists[0]
        self.density_dist_encoder = self.dist_encoder
        self.est_factory = [
            (GaussianEstimator(name='name'), GaussianAccumulatorFactory(name='name')),
            (GaussianEstimator(keys='key'), GaussianAccumulatorFactory(keys='key')),
            (GaussianEstimator(name='name', keys='key'), GaussianAccumulatorFactory(name='name', keys='key')),
            (GaussianEstimator(), GaussianAccumulatorFactory())
        ]
        self.factory_acc = [
            (GaussianAccumulatorFactory(name='b', keys='key'), GaussianAccumulator(name='b', keys='key')),
            (GaussianAccumulatorFactory(name='b'), GaussianAccumulator(name='b')),
            (GaussianAccumulatorFactory(keys='key'), GaussianAccumulator(keys='key')),
            (GaussianAccumulatorFactory(), GaussianAccumulator())
        ]
        self.acc_encoder = [
            (GaussianAccumulator(name='b', keys='key'), GaussianDataEncoder())
        ]
        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

    
    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "GaussianDistribution.seq_log_density() requires GaussianEncodedDataSequence."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                BinomialEstimator(keys=x)
                
            assert str(e.value) == "BinomialEstimator requires keys to be of type 'str'."
