"""Tests for CategoricalDistribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import CategoricalDistribution, CategoricalAccumulator, CategoricalAccumulatorFactory, CategoricalDataEncoder, CategoricalEstimator
import numpy as np
import pytest 

class CategoricalDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:

        self.eval_dists = [
            CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, default_value=0.0, name='cat'),
            CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, name='cat'),
            CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, default_value=0.0),
            CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6})
        ]
        self.dist_est = [
            (CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}), CategoricalEstimator()),
            (CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, name='c', keys='ckey'), CategoricalEstimator(name='c', keys='ckey'))
        ]
        
        self.dist_encoder = [(self.eval_dists[0], CategoricalDataEncoder())]
        self.sampler_dist = self.eval_dists[0]
        self.density_dist_encoder = [
            (CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, default_value=0.0, name='cat'), CategoricalDataEncoder()),
            (CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, default_value=0.50, name='cat'), CategoricalDataEncoder())
        ]
        self.est_factory = [
            (CategoricalEstimator(name='c', keys='key'), CategoricalAccumulatorFactory(name='c', keys='key')),
            (CategoricalEstimator(keys='key'), CategoricalAccumulatorFactory(keys='key')),
            (CategoricalEstimator(name='c'), CategoricalAccumulatorFactory(name='c')),
            (CategoricalEstimator(), CategoricalAccumulatorFactory())
        ]
        self.factory_acc = [
            (CategoricalAccumulatorFactory(name='c', keys='key'), CategoricalAccumulator(name='c', keys='key')),
            (CategoricalAccumulatorFactory(keys='key'), CategoricalAccumulator(keys='key')),
            (CategoricalAccumulatorFactory(name='c'), CategoricalAccumulator(name='c')),
            (CategoricalAccumulatorFactory(), CategoricalAccumulator())
        ]
        self.acc_encoder = [
            (CategoricalAccumulator(name='c', keys='key'), CategoricalDataEncoder())
        ]
        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "CategoricalDistribution.seq_log_density() requires CategoricalEncodedDataSequence."

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                CategoricalEstimator(keys=x)
                
            assert str(e.value) == "CategoricalEstimator requires keys to be of type 'str'."


# Test edge cases on initialize as needed
@pytest.mark.parametrize("default_value", [2.0])
def test_categorical_bad_default_value(default_value):
    dist = CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, default_value=default_value, name='cat')
    assert dist.default_value == 1.0




