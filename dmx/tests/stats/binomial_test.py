"""Tests for CategoricalDistribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.binomial import BinomialDataEncoder, BinomialAccumulator, BinomialAccumulatorFactory, BinomialDistribution, BinomialEstimator
import numpy as np
import pytest 

class BinomialDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:

        self.eval_dists = [
                BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key'),
                BinomialDistribution(p=0.7, n=20, name='a', keys='bino_key'),
                BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key')
        ]
        self.dist_est = [
            (BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key'), BinomialEstimator(name='a', keys='bino_key'))
        ]
        self.dist_encoder = [(BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key'), BinomialDataEncoder())]
        self.sampler_dist = BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key')
        self.density_dist_encoder = [
            (BinomialDistribution(p=0.7, n=20, min_val=-3, name='a', keys='bino_key'), BinomialDataEncoder())
        ]
        self.est_factory = [
            (BinomialEstimator(min_val=0, max_val=100, name='b', keys='bino'), BinomialAccumulatorFactory(min_val=0, max_val=100, name='b', keys='bino'))
        ]
        self.factory_acc = [
            (BinomialAccumulatorFactory(min_val=0, max_val=100, name='b', keys='bino'), BinomialAccumulator(min_val=0, max_val=100, name='b', keys='bino'))
        ]
        self.acc_encoder = [
            (BinomialAccumulator(min_val=0, max_val=100, name='b', keys='bino'), BinomialDataEncoder())
        ]
        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]


    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.density_dist_encoder[0][0].seq_log_density(x)
            assert str(e.value) == "BinomialDistribution.seq_log_density() requires BinomialEncodedDataSequence."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                BinomialEstimator(keys=x)
                
            assert str(e.value) == "BinomialEstimator requires keys to be of type 'str'."


# Test edge cases on initialize as needed
@pytest.mark.parametrize("p", [-10.0, 0.0, 1.0, 2.0, np.inf, np.nan])
def test_binomial_bad_p(p):
    with pytest.raises(Exception) as e:
        BinomialDistribution(p=p, n=20)
    
    assert str(e.value) == "Binomial distribution requires p in [0,1]"
    
@pytest.mark.parametrize("n", [-10, np.inf, np.nan])
def test_binomial_bad_n(n):
    with pytest.raises(Exception) as e:
        BinomialDistribution(p=0.50, n=n)

    assert str(e.value) == "Binomial distribution requires n > 0."