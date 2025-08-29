"""Tests for the Gaussian Mixture Model (GMM) distribution and its related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.gmm import *
import numpy as np
import pytest 


def component_log_density_test(dist, encoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_comp_ll = dist.seq_component_log_density(enc_data)

        for i in range(sz):
            
            if np.all(seq_comp_ll[i] == 0):
                seq_comp_ll[i] = np.abs(dist.component_log_density(data[i]))
            else:
                seq_comp_ll[i] = np.abs(seq_comp_ll[i] - dist.component_log_density(data[i])) / np.abs(seq_comp_ll[i])
        rv.append(np.max(seq_comp_ll))

    return max(rv) < 1.0e-14, "max(rv) test"

def posterior_test(dist, encoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_post = dist.seq_posterior(enc_data)

        for i in range(sz):
            
            if np.all(seq_post[i] == 0):
                seq_post[i] = np.abs(dist.posterior(data[i]))
            else:
                seq_post[i] = np.abs(seq_post[i] - dist.component_log_density(data[i])) / np.abs(seq_post[i])
        rv.append(np.max(seq_post))

    return max(rv) < 1.0e-14, "max(rv) test"



class GaussianMixtureDistributionTestCase(StatsTestClass):


    
    def setUp(self) -> None:
        mu = [
            [0., 5.0, 10.0],
            [-10., 20.],
            [0, 5., 10., 20.],
            [0., 5., -10., 20., 100.]
        ]
        sigma2 = [
            1.0,
            [1.0, 0.90],
            [1., 0.5, 0.25, .40],
            [1.] * 5
        ]
        sz = [len(x) for x in mu]
        w = [np.ones(x) / x for x in sz]
        keys = [
            ('w', 'comps'),
            ('w', None),
            (None, 'comps'),
            (None, None)
        ]

        self.eval_dists = [
            GaussianMixtureDistribution(mu=mu[0], sigma2=sigma2[0], w=w[0], name='name', keys=keys[0]),
            GaussianMixtureDistribution(mu=mu[1], sigma2=sigma2[1], w=w[1], name='name', keys=keys[1]),
            GaussianMixtureDistribution(mu=mu[2], sigma2=sigma2[2], w=w[2], name='name', keys=keys[2]),
            GaussianMixtureDistribution(mu=mu[3], sigma2=sigma2[3], w=w[3])
        ]
        self._ests = [
            GaussianMixtureEstimator(num_components=sz[0], tied=True, name='name', keys=keys[0]),
            GaussianMixtureEstimator(num_components=sz[1], name='name', keys=keys[1]),
            GaussianMixtureEstimator(num_components=sz[2], name='name', keys=keys[2]),
            GaussianMixtureEstimator(num_components=sz[3])
        ]
        self._factories = [
            GaussianMixtureAccumulatorFactory(num_components=sz[0], keys=keys[0], name='name', tied=True),
            GaussianMixtureAccumulatorFactory(num_components=sz[1], keys=keys[1], name='name'),
            GaussianMixtureAccumulatorFactory(num_components=sz[2], keys=keys[2], name='name'),
            GaussianMixtureAccumulatorFactory(num_components=sz[3])
        ]
        self._accumulators = [
            GaussianMixtureAccumulator(num_components=sz[0], keys=keys[0], name='name', tied=True),
            GaussianMixtureAccumulator(num_components=sz[1], keys=keys[1], name='name'),
            GaussianMixtureAccumulator(num_components=sz[2], keys=keys[2], name='name'),
            GaussianMixtureAccumulator(num_components=sz[3])
        ]
        self._encoders = [GaussianMixtureDataEncoder()]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [None, 'keys', (None, None, None), (1, 'keys')]

    def test_component_log_density(self):
        for x in self.density_dist_encoder:
            self.assertTrue(component_log_density_test(*x))

    def test_posterior(self):
        for x in self.density_dist_encoder:
            self.assertTrue(posterior_test(*x))

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
                
            assert str(e.value) == "GaussianMixtureEncodedDataSequence required for seq_log_density()."

    def test_seq_component_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_component_log_density(x)
                
            assert str(e.value) == "GaussianMixtureEncodedDataSequence required for seq_component_log_density()."

    def test_seq_posterior_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_posterior(x)
                
            assert str(e.value) == "GaussianMixtureEncodedDataSequence required for seq_posterior()."

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                GaussianMixtureEstimator(num_components=1, keys=x)
                
            assert str(e.value) == "GaussianMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."

