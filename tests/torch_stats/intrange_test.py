"""Tests for IntegerCategoricalDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.intrange import (
    IntegerCategoricalAccumulator,
    IntegerCategoricalAccumulatorFactory,
    IntegerCategoricalDataEncoder,
    IntegerCategoricalTorchSequence,
)


class IntegerCategoricalDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            IntegerCategoricalDistribution(min_val=0, p_vec=[0.2, 0.5, 0.3]),
            IntegerCategoricalDistribution(min_val=2, p_vec=[0.4, 0.4, 0.2]),
            IntegerCategoricalDistribution(min_val=-1, p_vec=[0.1, 0.3, 0.4, 0.2]),
        ]
        self._encoders = [d.dist_to_encoder() for d in self._dists]
        self._ests = [d.estimator() for d in self._dists]
        self._factories = [e.accumulator_factory() for e in self._ests]
        self._accs = [f.make(device=self.device) for f in self._factories]
        for dist in self._dists:
            dist.to(self.device)

        self.sampler_dist = self._dists[0]
        self.density_dist_encoder = list(zip(self._dists, self._encoders))
        self.dist_encoder = list(zip(self._dists, self._encoders))
        self.estimators = self._ests
        self.factories = self._factories
        self.accumulators = self._accs

    def test_seq_log_density_type(self):
        """seq_log_density must raise Exception when passed wrong encoded type."""
        for bad_input in [None, np.ones(10)]:
            with pytest.raises(Exception):
                self._dists[0].seq_log_density(bad_input)

    def test_encoder_type(self):
        """dist_to_encoder() must return an IntegerCategoricalDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), IntegerCategoricalDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return an IntegerCategoricalAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), IntegerCategoricalAccumulator)

    def test_sampler_range(self):
        """Samples must be within [min_val, min_val + len(p_vec) - 1]."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=500)
            min_v = dist.min_val
            max_v = min_v + len(dist.p_vec) - 1
            self.assertTrue(all(min_v <= x <= max_v for x in data),
                            f"Sample out of [{min_v}, {max_v}]")

    def test_prob_approx(self):
        """Empirical frequencies should approximate the probability vector."""
        dist = IntegerCategoricalDistribution(min_val=0, p_vec=[0.2, 0.5, 0.3])
        data = dist.sampler(seed=1).sample(size=10000)
        for val, expected_p in enumerate([0.2, 0.5, 0.3]):
            empirical = sum(1 for x in data if x == val) / len(data)
            self.assertAlmostEqual(empirical, expected_p, delta=0.03)
