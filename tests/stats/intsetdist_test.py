"""Test cases for Integer Bernoulli Set Distribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.intsetdist import *
import numpy as np
import pytest 


class IntegerBernoulliSetDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            IntegerBernoulliSetDistribution(log_pvec=np.log(np.ones(3) / 3.), name='name', keys='keys'),
            IntegerBernoulliSetDistribution(log_pvec=np.log(np.ones(2) / 2.), keys='keys'),
            IntegerBernoulliSetDistribution(log_pvec=np.log(np.ones(5) / 5.))
        ]
        self._ests = [
            IntegerBernoulliSetEstimator(num_vals=3, name='name', keys='keys'),
            IntegerBernoulliSetEstimator(num_vals=2, keys='keys'),
            IntegerBernoulliSetEstimator(num_vals=5),
        ]
        self._factories = [
            IntegerBernoulliSetAccumulatorFactory(num_vals=3, keys='keys', name='name'),
            IntegerBernoulliSetAccumulatorFactory(num_vals=2, keys='keys'),
            IntegerBernoulliSetAccumulatorFactory(num_vals=5)
        ]
        self._accumulators = [
            IntegerBernoulliSetAccumulator(num_vals=3, keys='keys', name='name'),
            IntegerBernoulliSetAccumulator(num_vals=2, keys='keys'),
            IntegerBernoulliSetAccumulator(num_vals=5)
        ]
        self._encoders = [IntegerBernoulliSetDataEncoder()]*len(self.eval_dists)

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

        self._init_ests = [
        ]
    

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "IntegerBernoulliSetEncodedDataSequence required for seq_log_density()."
   
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                IntegerBernoulliSetEstimator(num_vals=1, keys=x)
                
            assert str(e.value) == "IntegerBernoulliSetEstimator requires keys to be of type 'str'."
