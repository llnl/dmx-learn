"""Test cases for Multivariate Gaussian Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.mvn import *
import numpy as np
import pytest 

def gen_covar(seed, sz):
    x = np.random.RandomState(seed).rand(sz, sz)
    return x.T @ x

class MultivariateGaussianDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:

        self.mu = [
            [0., 5., 15.],
            np.arange(0, 20, 4),
            [0., 3.],
            np.asarray([-10., -5.0, 0.0, 5.0, 10.])
        ]
        self.dim = [len(x) for x in self.mu]
        self.covar = [gen_covar(i, x) for i, x in enumerate(self.dim)]
        
        self.eval_dists = [
            MultivariateGaussianDistribution(mu=self.mu[0], covar=self.covar[0], name='name', keys='keys'),
            MultivariateGaussianDistribution(mu=self.mu[1], covar=self.covar[1], keys='keys'),
            MultivariateGaussianDistribution(mu=self.mu[2], covar=self.covar[2], name='name'),
            MultivariateGaussianDistribution(mu=self.mu[3], covar=self.covar[3])

        ]
        self._ests = [
            MultivariateGaussianEstimator(dim=self.dim[0], name='name', keys='keys'),
            MultivariateGaussianEstimator(dim=self.dim[1], keys='keys'),
            MultivariateGaussianEstimator(dim=self.dim[2], name='name'),
            MultivariateGaussianEstimator(dim=self.dim[3])
        ]
        self._factories = [
            MultivariateGaussianAccumulatorFactory(dim=self.dim[0], name='name', keys='keys'),
            MultivariateGaussianAccumulatorFactory(dim=self.dim[1], keys='keys'),
            MultivariateGaussianAccumulatorFactory(dim=self.dim[2], name='name'),
            MultivariateGaussianAccumulatorFactory(dim=self.dim[3])
        ]
        self._accumulators = [
            MultivariateGaussianAccumulator(dim=self.dim[0], name='name', keys='keys'),
            MultivariateGaussianAccumulator(dim=self.dim[1], keys='keys'),
            MultivariateGaussianAccumulator(dim=self.dim[2], name='name'),
            MultivariateGaussianAccumulator(dim=self.dim[3])
        ]
        self._encoders = [MultivariateGaussianDataEncoder(dim=x) for x in self.dim]

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
            assert str(e.value) == 'MultivariateGaussianEncodedDataSequence required for seq_log_density().'  

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                MultivariateGaussianEstimator(keys=x)
                
            assert str(e.value) == "MultivariateGaussianEstimator requires keys to be of type 'str'."
