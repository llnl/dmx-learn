"""Tests for GeometricDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.geometric import (
    GeometricAccumulator,
    GeometricAccumulatorFactory,
    GeometricDataEncoder,
    GeometricTorchEncodedSequence,
)


class GeometricDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = torch.device("cpu")

        self._dists = [
            GeometricDistribution(p=0.3),
            GeometricDistribution(p=0.5),
            GeometricDistribution(p=0.8),
        ]
        self._encoders = [d.dist_to_encoder() for d in self._dists]
        self._ests = [d.estimator() for d in self._dists]
        self._factories = [e.accumulator_factory() for e in self._ests]
        self._accs = [f.make(device=self.device) for f in self._factories]

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
        """dist_to_encoder() must return a GeometricDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), GeometricDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a GeometricAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), GeometricAccumulator)

    def test_sampler_nonnegative(self):
        """Sampled values must be non-negative integers."""
        data = self._dists[0].sampler(seed=1).sample(size=500)
        self.assertTrue(all(x >= 0 for x in data), "Geometric samples must be non-negative")

    def test_mean_approx(self):
        """Sample mean should be close to 1/p (support starts at 1)."""
        dist = GeometricDistribution(p=0.5)
        data = dist.sampler(seed=1).sample(size=5000)
        sample_mean = float(np.mean(data))
        expected_mean = 1.0 / 0.5
        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.5)
