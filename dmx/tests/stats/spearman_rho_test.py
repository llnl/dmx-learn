"""Test cases for Spearman Ranking Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.spearman_rho import *
import numpy as np
import pytest 

def get_sigma(seed, sz):
    rng = np.random.RandomState(seed)
    rv = np.arange(sz)
    rng.shuffle(rv)
    return rv

class SpearmanRankingDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.dim = [3, 5, 8, 9]
        self.sigma = [get_sigma(i, x) for i, x in enumerate(self.dim)]
        self.eval_dists = [
            SpearmanRankingDistribution(sigma=self.sigma[0], rho=1.0, name='name', keys='keys'),
            SpearmanRankingDistribution(sigma=self.sigma[1], rho=1.0, name='name'),
            SpearmanRankingDistribution(sigma=self.sigma[2], rho=0.90, keys='keys'),
            SpearmanRankingDistribution(sigma=self.sigma[3], rho=0.80)
        ]
        self._ests = [
            SpearmanRankingEstimator(dim=self.dim[0], name='name', keys='keys'),
            SpearmanRankingEstimator(dim=self.dim[1], name='name'),
            SpearmanRankingEstimator(dim=self.dim[2], keys='keys'),
            SpearmanRankingEstimator(dim=self.dim[3])
        ]
        self._factories = [
            SpearmanRankingAccumulatorFactory(dim=self.dim[0], name='name', keys='keys'),
            SpearmanRankingAccumulatorFactory(dim=self.dim[1], name='name'),
            SpearmanRankingAccumulatorFactory(dim=self.dim[2], keys='keys'),
            SpearmanRankingAccumulatorFactory(dim=self.dim[3])
        ]
        self._accumulators = [
            SpearmanRankingAccumulator(dim=self.dim[0], name='name', keys='keys'),
            SpearmanRankingAccumulator(dim=self.dim[1], name='name'),
            SpearmanRankingAccumulator(dim=self.dim[2], keys='keys'),
            SpearmanRankingAccumulator(dim=self.dim[3])
        ]
        self._encoders = [SpearmanRankingDataEncoder()] * len(self.dim)

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
            assert str(e.value) == "SpearmanRankingEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                SpearmanRankingEstimator(dim=1, keys=x)
                
            assert str(e.value) == "SpearmanRankingEstimator requires keys to be of type 'str'."

