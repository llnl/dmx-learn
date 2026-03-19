"""Tests for PoissonDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.poisson import (
    PoissonAccumulator,
    PoissonAccumulatorFactory,
    PoissonDataEncoder,
    PoissonTorchSequence,
)


class PoissonDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            PoissonDistribution(lam=1.0),
            PoissonDistribution(lam=5.0),
            PoissonDistribution(lam=10.0),
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
        """dist_to_encoder() must return a PoissonDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), PoissonDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a PoissonAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), PoissonAccumulator)

    def test_sampler_nonnegative(self):
        """All sampled values must be non-negative integers."""
        data = self._dists[0].sampler(seed=1).sample(size=500)
        self.assertTrue(all(x >= 0 for x in data), "Poisson samples must be non-negative")

    def test_mean_approx(self):
        """Sample mean should be close to lambda."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=5000)
            sample_mean = float(np.mean(data))
            self.assertAlmostEqual(sample_mean, dist.lam, delta=0.3)
