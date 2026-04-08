"""Tests for BinomialDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.binomial import (
    BinomialAccumulator,
    BinomialAccumulatorFactory,
    BinomialDataEncoder,
    BinomialTorchEncodedSequence,
)


class BinomialDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            BinomialDistribution(p=0.3, n=10),
            BinomialDistribution(p=0.5, n=20),
            BinomialDistribution(p=0.8, n=5),
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
        """dist_to_encoder() must return a BinomialDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), BinomialDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a BinomialAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), BinomialAccumulator)

    def test_sampler_range(self):
        """Sampled values must be in [0, n]."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=500)
            self.assertTrue(all(0 <= x <= dist.n for x in data),
                            f"Binomial samples out of [0, {dist.n}]")

    def test_mean_approx(self):
        """Sample mean should be close to n * p."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=5000)
            sample_mean = float(np.mean(data))
            expected_mean = dist.n * dist.p
            self.assertAlmostEqual(sample_mean, expected_mean, delta=0.5)
