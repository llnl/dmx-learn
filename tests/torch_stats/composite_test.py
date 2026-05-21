"""Tests for CompositeDistribution and related torch_stats classes."""

# pylint: disable=duplicate-code,line-too-long,wildcard-import
# pylint: disable=unused-wildcard-import,unused-import

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.composite import (
    CompositeAccumulator,
    CompositeAccumulatorFactory,
    CompositeDataEncoder,
    CompositeEstimator,
    CompositeTorchEncodedSequence,
)
from tests.torch_stats.torch_stats_tests import *


class CompositeDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            CompositeDistribution(
                dists=[
                    GaussianDistribution(mu=0.0, sigma2=1.0),
                    PoissonDistribution(lam=3.0),
                ],
            ),
            CompositeDistribution(
                dists=[
                    GaussianDistribution(mu=2.0, sigma2=0.5),
                    ExponentialDistribution(beta=1.0),
                    PoissonDistribution(lam=5.0),
                ],
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
        """dist_to_encoder() must return a CompositeDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), CompositeDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a CompositeAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), CompositeAccumulator)

    def test_estimator_is_composite(self):
        """dist.estimator() must return a CompositeEstimator."""
        for dist in self._dists:
            self.assertIsInstance(dist.estimator(), CompositeEstimator)

    def test_sample_tuple_length(self):
        """Each sample must be a tuple/list with one element per component distribution."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            num_dists = len(dist.dists)
            for obs in data:
                self.assertEqual(
                    len(obs),
                    num_dists,
                    f"Expected tuple of length {num_dists}, got {len(obs)}",
                )

    def test_log_density_additivity(self):
        """log_density of composite == sum of component log densities."""
        dist = self._dists[0]  # Gaussian + Poisson
        data = dist.sampler(seed=1).sample(size=20)
        for obs in data:
            composite_ll = dist.log_density(obs)
            component_ll = sum(
                comp.log_density(obs[i]) for i, comp in enumerate(dist.dists)
            )
            self.assertAlmostEqual(composite_ll, component_ll, places=8)
