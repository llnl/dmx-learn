"""Tests for JointMixtureDistribution and related torch_stats classes."""

# pylint: disable=duplicate-code,line-too-long,wildcard-import
# pylint: disable=unused-wildcard-import,unused-import

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.jmixture import (
    JointMixtureDataEncoder,
    JointMixtureEstimatorAccumulator,
    JointMixtureEstimatorAccumulatorFactory,
    JointMixtureTorchEncodedSequence,
)
from tests.torch_stats.torch_stats_tests import *


class JointMixtureDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        taus12 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        taus21 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        self._dist1 = JointMixtureDistribution(
            components1=[
                GaussianDistribution(mu=-6.0, sigma2=1.0),
                GaussianDistribution(mu=0.0, sigma2=1.0),
                GaussianDistribution(mu=6.0, sigma2=1.0),
            ],
            components2=[
                GaussianDistribution(mu=-4.0, sigma2=1.0),
                GaussianDistribution(mu=0.0, sigma2=1.0),
                GaussianDistribution(mu=4.0, sigma2=1.0),
            ],
            w1=[0.6, 0.3, 0.1],
            w2=[0.7, 0.2, 0.1],
            taus12=taus12,
            taus21=taus21,
        )

        self._dist2 = JointMixtureDistribution(
            components1=[
                GaussianDistribution(mu=-3.0, sigma2=0.5),
                GaussianDistribution(mu=3.0, sigma2=0.5),
            ],
            components2=[
                GaussianDistribution(mu=-2.0, sigma2=0.5),
                GaussianDistribution(mu=2.0, sigma2=0.5),
            ],
            w1=[0.5, 0.5],
            w2=[0.4, 0.6],
            taus12=[[0.9, 0.1], [0.1, 0.9]],
            taus21=[[0.8, 0.2], [0.2, 0.8]],
        )

        self._dists = [self._dist1, self._dist2]
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
        """dist_to_encoder() must return a JointMixtureDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), JointMixtureDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a JointMixtureEstimatorAccumulator."""
        for f in self._factories:
            self.assertIsInstance(
                f.make(device=self.device), JointMixtureEstimatorAccumulator
            )

    def test_sample_is_pair(self):
        """Each sample must be a (x1, x2) pair."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            for obs in data:
                self.assertEqual(
                    len(obs), 2, f"Expected (x1, x2) pair, got length {len(obs)}"
                )

    def test_log_density_finite(self):
        """log_density must return finite values for sampled observations."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            for obs in data:
                ll = dist.log_density(obs)
                self.assertTrue(
                    np.isfinite(ll), f"log_density returned {ll} for obs {obs}"
                )
