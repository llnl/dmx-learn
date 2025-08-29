"""Test cases for Integer PLSI Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.int_plsi import *
from dmx.stats.categorical import * 
import numpy as np
import pytest 

class IntegerPLSIDistributionTestCase(StatsTestClass):
    
    def setUp(self) -> None:
        rng = np.random.RandomState(1)
        num_states = 3
        num_authors = 10
        num_words = 50

        state_word_mat = rng.dirichlet(alpha=np.ones(num_words), size=num_states).T
        doc_state_mat = rng.dirichlet(alpha=np.ones(num_states), size=num_authors)
        doc_vec = rng.dirichlet(alpha=np.ones(num_authors))

        # Length distribution 
        len_dist = CategoricalDistribution({20: 0.5, 30: 0.5})
        len_acc = CategoricalAccumulator()
        len_fac = CategoricalAccumulatorFactory()
        len_est = CategoricalEstimator()
        len_enc = CategoricalDataEncoder()

        keys = [('a' if i // 4 == 0 else None, 'b' if (i // 2) % 2 == 0 else None, 'c' if i % 2 == 0 else None) for i in range(8)]
        name = ['name' if i % 2 == 0 else None for i in range(8)]
        self.eval_dists = []
        self._ests = []
        self._factories = []
        self._accumulators = []

        for i in range(8):
            self.eval_dists.append(IntegerPLSIDistribution(state_word_mat=state_word_mat, doc_state_mat=doc_state_mat, doc_vec=doc_vec, len_dist=len_dist, name=name[i], keys=keys[i]))
            self._ests.append(IntegerPLSIEstimator(num_vals=num_words, num_states=num_states, num_docs=num_authors, len_estimator=len_est, name=name[i], keys=keys[i]))
            self._factories.append(IntegerPLSIAccumulatorFactory(num_vals=num_words, num_states=num_states, num_docs=num_authors, len_factory=len_fac, name=name[i], keys=keys[i]))
            self._accumulators.append(IntegerPLSIAccumulator(num_vals=num_words, num_states=num_states, num_docs=num_authors, len_acc=len_acc, name=name[i], keys=keys[i]))

        self._encoders = [IntegerPLSIDataEncoder(len_encoder=len_enc)]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]

    @pytest.mark.dependency(depends=["estimator", "log_density", "estimator_factory", "factory_make"])
    def test_09_seq_update(self):
        for x in self.density_dist_encoder:
            res = seq_estimation_test(*x)
            self.assertTrue(res[0], str(res[1]))

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "IntegerPLSIEncodedDataSequence required for seq_log_density()."

    def test_seq_component_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_component_log_density(x)
            assert str(e.value) == "IntegerPLSIEncodedDataSequence required for seq_component_log_density()."