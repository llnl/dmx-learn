"""Test cases for Markov Chain Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.markovchain import *
from dmx.stats.categorical import * 
from dmx.stats.sequence import * 
import numpy as np
import pytest 



class MarkovChainDistributionTestCase(StatsTestClass):
    
    def setUp(self) -> None:
        rng = np.random.RandomState(1)
        vals = ['a', 'b', 'c', 'd', 'e']
        pi = rng.dirichlet(alpha=[1.]*len(vals)).tolist()
        init_prob_map = {k: v for k, v in zip(vals, pi)}
        trans_map = {v: {} for v in vals}
        for x in vals:
            w = rng.dirichlet(alpha=[1.]*len(vals)).tolist()
            trans_map[x] = {k: v for k, v in zip(vals, w)}

        # Length distribution 
        len_dist = CategoricalDistribution({3: 0.5, 5: 0.5})
        len_acc = CategoricalAccumulator()
        len_fac = CategoricalAccumulatorFactory()
        len_est = CategoricalEstimator()
        len_enc = CategoricalDataEncoder()

        self.eval_dists = [
            MarkovChainDistribution(init_prob_map=init_prob_map, transition_map=trans_map, len_dist=len_dist, keys='keys', name='name'),
            MarkovChainDistribution(init_prob_map=init_prob_map, transition_map=trans_map, len_dist=len_dist, keys='keys'),
            MarkovChainDistribution(init_prob_map=init_prob_map, transition_map=trans_map, len_dist=len_dist)
        ]
        self._ests = [
            MarkovChainEstimator(len_estimator=len_est, keys='keys', name='name'),
            MarkovChainEstimator(len_estimator=len_est, keys='keys'),
            MarkovChainEstimator(len_estimator=len_est)
        ]
        self._factories = [
            MarkovChainAccumulatorFactory(len_factory=len_fac, keys='keys', name='name'),
            MarkovChainAccumulatorFactory(len_factory=len_fac, keys='keys'),
            MarkovChainAccumulatorFactory(len_factory=len_fac)
        ]
        self._accumulators = [
            MarkovChainAccumulator(len_accumulator=len_acc, keys='keys', name='name'),
            MarkovChainAccumulator(len_accumulator=len_acc, keys='keys'),
            MarkovChainAccumulator(len_accumulator=len_acc)
        ]
        self._encoders = [MarkovChainDataEncoder(len_encoder=len_enc)]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "MarkovChainEncodedDataSequence required for seq_log_density()."