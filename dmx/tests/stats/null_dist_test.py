"""Test cases for Null Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.null_dist import NullDistribution, NullAccumulator, NullAccumulatorFactory, NullDataEncoder, NullEstimator
import numpy as np
import unittest
import pytest 

class NullDistributionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dists = [
                NullDistribution(name='a')
        ]
        self.dist_encoder = [(self.dists[0], NullDataEncoder())]
        self.est_factory = [
            (NullEstimator(keys='null'), NullAccumulatorFactory(keys='null'))
        ]
        self.factory_acc = [
            (NullAccumulatorFactory(keys='null'), NullAccumulator(keys='null'))
        ]
        self.acc_encoder = [
            (NullAccumulator(keys='null'), NullDataEncoder())
        ]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]

    def test_01_str_eval(self):
        for dist in self.dists:
            self.assertTrue(str_eval_test(dist))

    @pytest.mark.dependency(name="estimator")
    def test_02_estimator(self):
        rv = estimator_test(self.dists[0], NullEstimator(name='a'))
        self.assertTrue(rv)

    @pytest.mark.dependency(name="dist_to_encoder")
    def test_03_dist_to_encoder(self):
        for dist, encoder in self.dist_encoder:
            self.assertTrue(dist_to_encoder_test(dist, encoder))

    @pytest.mark.dependency(name="log_density")
    def test_05_log_density(self):
        for dist, _ in self.dist_encoder:
            self.assertTrue(dist.seq_log_density([10]*3) == 3*dist.log_density([10]))


    @pytest.mark.dependency(name="estimator_factory")
    def test_06_estimator_factory(self):
        for est_factory in self.est_factory:
            rv = estimator_factory_test(est_factory)
            self.assertTrue(rv)
    
    @pytest.mark.dependency(name="factory_make")
    def test_07_factory_make(self):
        for x in self.factory_acc:
            rv = factory_make_test(x)
            self.assertTrue(rv)

    @pytest.mark.dependency(name="acc_to_encoder")
    def test_08_acc_to_encoder(self):
        for x in self.acc_encoder:
            rv = acc_to_encoder_test(x)
            self.assertTrue(rv)
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                NullEstimator(keys=x)
                
            assert str(e.value) == "NullEstimator requires keys to be of type 'str'."




