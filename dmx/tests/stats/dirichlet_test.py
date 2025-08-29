"""Tests for DirichletDistribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.dirichlet import *
import numpy as np
import pytest 

class DirichletDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            DirichletDistribution(alpha=np.ones(3) / 3., name='name', keys='keys'),
            DirichletDistribution(alpha=np.ones(3), keys='keys'),
            DirichletDistribution(alpha=np.arange(1, 5), name='name'),
            DirichletDistribution(alpha=np.arange(1, 6)),
        ]
        self._ests = [
            DirichletEstimator(dim=3, name='name', keys='keys'),
            DirichletEstimator(dim=3, keys='keys'),
            DirichletEstimator(dim=4, name='name'),
            DirichletEstimator(dim=5)
        ]
        self._factories = [
            DirichletAccumulatorFactory(dim=3, keys='keys', name='name'),
            DirichletAccumulatorFactory(dim=3, keys='keys'),
            DirichletAccumulatorFactory(dim=4, name='name'),
            DirichletAccumulatorFactory(dim=5)
        ]
        self._accumulators = [
            DirichletAccumulator(dim=3, keys='keys', name='name'),
            DirichletAccumulator(dim=3, keys='keys'),
            DirichletAccumulator(dim=4, name='name'),
            DirichletAccumulator(dim=5)
        ]
        self._encoders = [DirichletDataEncoder()] * len(self.eval_dists)

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
            assert str(e.value) == 'DirichletEncodedDataSequence required for DirichletDistribution.seq_log_density().'

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                DirichletEstimator(dim=1, keys=x)
                
            assert str(e.value) == "DirichletEstimator requires keys to be of type 'str'."