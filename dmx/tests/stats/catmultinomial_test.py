"""Tests for the MultinomialDistribution class and related components."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.catmultinomial import * 
from dmx.stats.binomial import BinomialDataEncoder, BinomialAccumulator, BinomialAccumulatorFactory
from dmx.stats.categorical import CategoricalDataEncoder, CategoricalAccumulator, CategoricalAccumulatorFactory
from dmx.stats.null_dist import NullDataEncoder, NullAccumulator
import numpy as np
import pytest 

class MultinomialDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        
        self.base_dist = CategoricalDistribution(pmap={'a': 0.1, 'b': 0.4, 'c': 0.5})
        self.base_acc = CategoricalAccumulator(name='cat', keys='cat_keys')
        self.base_factory = CategoricalAccumulatorFactory(name='cat', keys='cat_keys')
        self.base_est = CategoricalEstimator(name='cat', keys='cat_keys')

        self.len_dist = BinomialDistribution(p=0.5, n=3, min_val=10)
        self.len_acc = BinomialAccumulator(name='b', keys='b_keys')
        self.len_factory = BinomialAccumulatorFactory(name='b', keys='b_keys')
        self.len_est = BinomialEstimator(name='b', keys='b_keys')

        self.encoder_list = [
            MultinomialDataEncoder(encoder=CategoricalDataEncoder(), len_encoder=BinomialDataEncoder()),
            MultinomialDataEncoder(encoder=CategoricalDataEncoder(), len_encoder=NullDataEncoder())
        ]

        self.eval_dists = [
                MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, len_normalized=True),
                MultinomialDistribution(dist=self.base_dist),
                MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, name='a'),
                MultinomialDistribution(dist=self.base_dist, name='a')
        ]
        self.sampler_dist = MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, len_normalized=True)
        self.density_dist_encoder = [
                (
                    MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, len_normalized=True),
                    self.encoder_list[0]
                ),
                (
                    MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, len_normalized=False),
                    self.encoder_list[0]
                )
        ]

        self.dist_est = [
            (MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, name='a'), 
             MultinomialEstimator(estimator=CategoricalEstimator(), 
                                  len_estimator=BinomialEstimator(), name='a')),
            (MultinomialDistribution(dist=self.base_dist, len_dist=self.len_dist, name='a', keys='mkey'), 
             MultinomialEstimator(estimator=CategoricalEstimator(), 
                                  len_estimator=BinomialEstimator(), name='a', keys='mkey'))
        ]
        self.dist_encoder = [(x, y) for x, y in zip(self.eval_dists[:2], self.encoder_list)]
        self.est_factory = [
            (
                MultinomialEstimator(estimator=self.base_est, len_estimator=self.len_est, name='m', keys='mkeys'), 
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=self.len_factory, len_normalized=False, name='m', keys='mkeys')
            ),
            (
                MultinomialEstimator(estimator=self.base_est, len_estimator=self.len_est, len_normalized=True, name='m', keys='mkeys'), 
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=self.len_factory, len_normalized=True, name='m', keys='mkeys')
            ),
            (
                MultinomialEstimator(estimator=self.base_est, len_estimator=NullEstimator(), name='m', keys='mkeys'), 
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=NullAccumulatorFactory(), len_normalized=False, name='m', keys='mkeys')
            )
        ]
        self.factory_acc = [
            (
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=self.len_factory, len_normalized=False, name='m', keys='mkeys'), 
                MultinomialAccumulator(accumulator=self.base_acc, len_accumulator=self.len_acc, len_normalized=False, name='m', keys='mkeys')
            ),
            (
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=self.len_factory, len_normalized=True, name='m', keys='mkeys'), 
                MultinomialAccumulator(accumulator=self.base_acc, len_accumulator=self.len_acc, len_normalized=True, name='m', keys='mkeys')
            ),
            (
                MultinomialAccumulatorFactory(est_factory=self.base_factory, len_factory=NullAccumulatorFactory(), len_normalized=False, name='m', keys='mkeys'), 
                MultinomialAccumulator(accumulator=self.base_acc, len_accumulator=NullAccumulator(), len_normalized=False, name='m', keys='mkeys')
            ),

        ]
        self.acc_encoder = [
            (MultinomialAccumulator(accumulator=self.base_acc, len_accumulator=self.len_acc, len_normalized=False), self.encoder_list[0]),
            (MultinomialAccumulator(accumulator=self.base_acc, len_accumulator=NullAccumulator(), len_normalized=False), self.encoder_list[1])
        ]
        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]


    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.density_dist_encoder[0][0].seq_log_density(x)
            assert str(e.value) == "MultinomialDistribution.seq_log_density() requires MultinomialEncodedDataSequence."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                MultinomialEstimator(estimator=ParameterEstimator(), keys=x)
                
            assert str(e.value) == "MultinomialEstimator requires keys to be of type 'str'."
