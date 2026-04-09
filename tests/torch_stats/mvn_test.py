"""Tests for MultivariateGaussianDistribution and related torch_stats classes."""

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.mvn import (
    MultivariateGaussianAccumulator,
    MultivariateGaussianAccumulatorFactory,
    MultivariateGaussianDataEncoder,
    MultivariateGaussianTorchSequence,
)
from tests.torch_stats.torch_stats_tests import *


class MultivariateGaussianDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            MultivariateGaussianDistribution(
                mu=[0.0, 0.0],
                covar=[[1.0, 0.0], [0.0, 1.0]],
            ),
            MultivariateGaussianDistribution(
                mu=[2.0, -1.0],
                covar=[[2.0, 0.5], [0.5, 1.0]],
            ),
            MultivariateGaussianDistribution(
                mu=[0.0, 0.0, 0.0],
                covar=[[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]],
            ),
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
        """dist_to_encoder() must return a MultivariateGaussianDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(
                dist.dist_to_encoder(), MultivariateGaussianDataEncoder
            )

    def test_accumulator_type(self):
        """factory.make() must return a MultivariateGaussianAccumulator."""
        for f in self._factories:
            self.assertIsInstance(
                f.make(device=self.device), MultivariateGaussianAccumulator
            )

    def test_sample_shape(self):
        """Each sample must be a 1-D array of the correct dimension."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=10)
            dim = len(dist.mu)
            for obs in data:
                self.assertEqual(len(obs), dim, f"Expected dim {dim}, got {len(obs)}")

    def test_sample_mean_approx(self):
        """Column-wise mean of samples should be close to mu."""
        dist = MultivariateGaussianDistribution(
            mu=[3.0, -1.0],
            covar=[[1.0, 0.0], [0.0, 1.0]],
        )
        data = np.array(dist.sampler(seed=1).sample(size=5000))
        for i, mu_i in enumerate([3.0, -1.0]):
            self.assertAlmostEqual(float(np.mean(data[:, i])), mu_i, delta=0.15)
