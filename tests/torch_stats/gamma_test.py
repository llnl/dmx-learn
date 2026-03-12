"""Tests for GammaDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.gamma import (
    GammaAccumulator,
    GammaAccumulatorFactory,
    GammaDataEncoder,
    GammaTorchEncodedSequence,
)


class GammaDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = torch.device("cpu")

        self._dists = [
            GammaDistribution(k=2.0, theta=1.0),
            GammaDistribution(k=1.0, theta=0.5),
            GammaDistribution(k=5.0, theta=2.0),
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
        """dist_to_encoder() must return a GammaDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), GammaDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a GammaAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), GammaAccumulator)

    def test_sampler_positive(self):
        """All sampled values must be positive (Gamma has support on (0, inf))."""
        data = self._dists[0].sampler(seed=1).sample(size=500)
        self.assertTrue(all(x > 0 for x in data), "Gamma samples must be positive")

    def test_mean_approx(self):
        """Sample mean should be close to k * theta."""
        dist = GammaDistribution(k=4.0, theta=2.0)
        data = dist.sampler(seed=1).sample(size=5000)
        sample_mean = float(np.mean(data))
        expected_mean = 4.0 * 2.0
        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.5)
