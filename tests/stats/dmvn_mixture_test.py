"""Tests for the Diagonal Gaussian Mixture Distribution."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.dmvn_mixture import *
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
                seq_post[i] = np.where(seq_post[i] != 0, np.abs(seq_post[i] - dist.posterior(data[i])) / np.abs(seq_post[i]), dist.posterior(data[i]))
        rv.append(np.max(seq_post))

    return max(rv) < 1.0e-14, "max(rv) test"


def fast_posterior_test(dist, encoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_post = dist.seq_posterior(enc_data).flatten()
        fast_post = dist.fast_seq_posterior(enc_data).flatten()
        
        rv.append(np.max(np.where(seq_post > 0.0, np.abs(seq_post-fast_post) / np.abs(seq_post), fast_post)))

    return max(rv) < 1.0e-14, "max(rv) test"


class DiagonalGaussianMixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        dim_tups = [(2,3), (3, 5), (5, 4), (6, 2)]

        mu = [np.reshape(np.linspace(-100, 100, num=k*d, endpoint=True), (k, d)) for k, d in dim_tups]
        covar = [np.ones((k, d)) for (k, d) in dim_tups]

        ncomps = [x for x, _ in dim_tups]
        dims = [x for _, x in dim_tups]
        w = [np.ones(x) / x for x in ncomps]

        keys = [
            ('w', 'comps'),
            ('w', None),
            (None, 'comps'),
            (None, None)
        ]

        self.eval_dists = [
            DiagonalGaussianMixtureDistribution(mu=mu[0], covar=covar[0], w=w[0], name='name', keys=keys[0], tied=True),
            DiagonalGaussianMixtureDistribution(mu=mu[1], covar=covar[1], w=w[1], name='name', keys=keys[1]),
            DiagonalGaussianMixtureDistribution(mu=mu[2], covar=covar[2], w=w[2], name='name', keys=keys[2]),
            DiagonalGaussianMixtureDistribution(mu=mu[3], covar=covar[3], w=w[3])
        ]
        self._ests = [
            DiagonalGaussianMixtureEstimator(num_components=ncomps[0], dim=dims[0], tied=True, name='name', keys=keys[0]),
            DiagonalGaussianMixtureEstimator(num_components=ncomps[1], dim=dims[1], name='name', keys=keys[1]),
            DiagonalGaussianMixtureEstimator(num_components=ncomps[2], dim=dims[2], name='name', keys=keys[2]),
            DiagonalGaussianMixtureEstimator(num_components=ncomps[3], dim=dims[3])
        ]
        self._factories = [
            DiagonalGaussianMixtureAccumulatorFactory(num_components=ncomps[0], dim=dims[0], keys=keys[0], name='name', tied=True),
            DiagonalGaussianMixtureAccumulatorFactory(num_components=ncomps[1], dim=dims[1], keys=keys[1], name='name'),
            DiagonalGaussianMixtureAccumulatorFactory(num_components=ncomps[2], dim=dims[2], keys=keys[2], name='name'),
            DiagonalGaussianMixtureAccumulatorFactory(num_components=ncomps[3], dim=dims[3])
        ]
        self._accumulators = [
            DiagonalGaussianMixtureAccumulator(num_components=ncomps[0], dim=dims[0], keys=keys[0], name='name', tied=True),
            DiagonalGaussianMixtureAccumulator(num_components=ncomps[1], dim=dims[1], keys=keys[1], name='name'),
            DiagonalGaussianMixtureAccumulator(num_components=ncomps[2], dim=dims[2], keys=keys[2], name='name'),
            DiagonalGaussianMixtureAccumulator(num_components=ncomps[3], dim=dims[3])
        ]
        self._encoders = [DiagonalGaussianMixtureDataEncoder()]*len(self.eval_dists)

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
            self.assertTrue(component_log_density_test(*x)[0])

    def test_posterior(self):
        for x in self.density_dist_encoder:
            self.assertTrue(posterior_test(*x)[0])

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
                
            assert str(e.value) == "DiagonalGaussianMixtureEncodedDataSequence required for seq_log_density()."

    def test_seq_component_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_component_log_density(x)
                
            assert str(e.value) == "DiagonalGaussianMixtureEncodedDataSequence required for seq_component_log_density()."

    def test_seq_posterior_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_posterior(x)
                
            assert str(e.value) == "DiagonalGaussianMixtureEncodedDataSequence required for seq_posterior()."

    def test_fast_seq_posterior(self):
        for x in self.density_dist_encoder:
            self.assertTrue(fast_posterior_test(*x))
        
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                DiagonalGaussianMixtureEstimator(num_components=1, dim=1, keys=x)
                
            assert str(e.value) == "DiagonalGaussianMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."