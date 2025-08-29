"""Test cases for Integer Multinomial Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.intmultinomial import *
from dmx.stats.categorical import *
from dmx.stats.poisson import * 
import numpy as np
import pytest 


class IntMultinomialDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:

        self._len_dists = [
            CategoricalDistribution({5: 0.5, 6: 0.5}),
            PoissonDistribution(lam=10.0)
        ]

        self.eval_dists = [
            IntegerMultinomialDistribution(min_val=0, p_vec=np.ones(3) / 3., len_dist=self._len_dists[0], name='name', keys='keys'),
            IntegerMultinomialDistribution(min_val=4, p_vec=np.ones(5) / 5., len_dist=self._len_dists[1])
        ]
        self._ests = [
            IntegerMultinomialEstimator(len_estimator=CategoricalEstimator(), name='name', keys='keys'),
            IntegerMultinomialEstimator(len_estimator=PoissonEstimator())
        ]
        self._factories = [
            IntegerMultinomialAccumulatorFactory(len_factory=CategoricalAccumulatorFactory(), name='name', keys='keys'),
            IntegerMultinomialAccumulatorFactory(len_factory=PoissonAccumulatorFactory())
        ]
        self._accumulators = [
            IntegerMultinomialAccumulator(name='name', keys='keys', len_accumulator=CategoricalAccumulator()),
            IntegerMultinomialAccumulator(len_accumulator=PoissonAccumulator())
        ]
        self._encoders = [
            IntegerMultinomialDataEncoder(len_encoder=CategoricalDataEncoder()),
            IntegerMultinomialDataEncoder(len_encoder=PoissonDataEncoder())
        ]

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
            assert str(e.value) == "IntegerMultinomialEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                IntegerMultinomialEstimator(keys=x)
                
            assert str(e.value) == "IntegerMultinomialEstimator requires keys to be of type 'str'."
