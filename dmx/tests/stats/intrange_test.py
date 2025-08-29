"""Test cases for Integer Categorical Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.intrange import *
import numpy as np
import pytest 


class IntegerCategoricalDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(3) / 3., name='name', keys='keys'),
            IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(2) / 2., name='name'),
            IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(4) / 4., keys='keys'),
            IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(5) / 5.)
        ]
        self._ests = [
            [
                IntegerCategoricalEstimator(name='name', keys='keys'),
                IntegerCategoricalEstimator(name='name'),
                IntegerCategoricalEstimator(keys='keys'),
                IntegerCategoricalEstimator()
            ],
            [
                IntegerCategoricalEstimator(min_val=0, max_val=3, name='name', keys='keys'),
                IntegerCategoricalEstimator(max_val=1, name='name'),
                IntegerCategoricalEstimator()
            ]

        ]
        self._factories = [
            [
                IntegerCategoricalAccumulatorFactory(min_val=0, max_val=3, name='name', keys='keys'),
                IntegerCategoricalAccumulatorFactory(name='name'),
                IntegerCategoricalAccumulatorFactory()
            ],
            [
                IntegerCategoricalAccumulatorFactory(min_val=0, max_val=2, name='name', keys='keys'),
                IntegerCategoricalAccumulatorFactory()
            ],

        ]
        self._accumulators = [
            IntegerCategoricalAccumulator(min_val=0, max_val=2, name='name', keys='keys'),
            IntegerCategoricalAccumulator()
        ]
        self._encoders = [IntegerCategoricalDataEncoder()]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests[0])]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests[1], self._factories[0])]
        self.factory_acc = [(f, a) for f, a in zip(self._factories[1], self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

        self._init_ests = [
        ]

    

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "IntegerCategoricalEncodedDataSequence required for seq_log_density()."            
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                IntegerCategoricalEstimator(keys=x)
                
            assert str(e.value) == "IntegerCategoricalEstimator requires keys to be of type 'str'."

