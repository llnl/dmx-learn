"""Test cases for Optional Distribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.poisson import *
from dmx.stats.optional import * 
import numpy as np
import pytest 

class OptionalDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self._base_dist = PoissonDistribution(lam=3.0)
        self._base_acc = PoissonAccumulator()
        self._base_fac = PoissonAccumulatorFactory()
        self._base_est = PoissonEstimator()
        self._base_encoder = PoissonDataEncoder()


        self.eval_dists = [
            OptionalDistribution(self._base_dist, p=0.10, missing_value=np.nan, name='name', keys='keys'),
            OptionalDistribution(self._base_dist, p=0.10, keys='keys'),
            OptionalDistribution(self._base_dist, p=0.10, name='name'),
            OptionalDistribution(self._base_dist)

        ]
        self._ests = [
            OptionalEstimator(estimator=self._base_est, missing_value=np.nan, name='name', keys='keys', est_prob=True),
            OptionalEstimator(estimator=self._base_est, keys='keys', est_prob=True),
            OptionalEstimator(estimator=self._base_est, name='name', est_prob=True),
            OptionalEstimator(estimator=self._base_est)
        ]
        self._factories = [
            OptionalEstimatorAccumulatorFactory(estimator=self._base_est, missing_value=np.nan, name='name', keys='keys'),
            OptionalEstimatorAccumulatorFactory(estimator=self._base_est, keys='keys'),
            OptionalEstimatorAccumulatorFactory(estimator=self._base_est, name='name'),
            OptionalEstimatorAccumulatorFactory(estimator=self._base_est)
        ]
        self._accumulators = [
            OptionalEstimatorAccumulator(accumulator=self._base_acc, missing_value=np.nan, name='name', keys='keys'),
            OptionalEstimatorAccumulator(accumulator=self._base_acc, keys='keys'),
            OptionalEstimatorAccumulator(accumulator=self._base_acc, name='name'),
            OptionalEstimatorAccumulator(accumulator=self._base_acc)
        ]
        self._encoders = [
            OptionalDataEncoder(encoder=self._base_encoder, missing_value=np.nan),
            OptionalDataEncoder(encoder=self._base_encoder),
            OptionalDataEncoder(encoder=self._base_encoder),
            OptionalDataEncoder(encoder=self._base_encoder)
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
            assert str(e.value) == "OptionalEncodedDataSequence required for seq_log_density()."
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                OptionalEstimator(estimator=ParameterEstimator(), keys=x)
                
            assert str(e.value) == "OptionalEstimator requires keys to be of type 'str'."

