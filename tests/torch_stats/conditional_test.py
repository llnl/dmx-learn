"""Tests for ConditionalDistribution and related torch_stats classes."""

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.conditional import (
    ConditionalDistributionAccumulator,
    ConditionalDistributionAccumulatorFactory,
    ConditionalDistributionDataEncoder,
    ConditionalTorchEncodedSequence,
)
from tests.torch_stats.torch_stats_tests import *


class ConditionalDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        # Condition on integer categories (0, 1), each with its own Gaussian
        given_dist1 = IntegerCategoricalDistribution(min_val=0, p_vec=[0.5, 0.5])
        self._dist1 = ConditionalDistribution(
            dmap={
                0: GaussianDistribution(mu=-3.0, sigma2=1.0),
                1: GaussianDistribution(mu=3.0, sigma2=1.0),
            },
            given_dist=given_dist1,
        )

        # Condition on integer categories (0, 1, 2), each with a Poisson
        given_dist2 = IntegerCategoricalDistribution(min_val=0, p_vec=[0.3, 0.4, 0.3])
        self._dist2 = ConditionalDistribution(
            dmap={
                0: PoissonDistribution(lam=1.0),
                1: PoissonDistribution(lam=5.0),
                2: PoissonDistribution(lam=10.0),
            },
            given_dist=given_dist2,
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
        """dist_to_encoder() must return a ConditionalDistributionDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(
                dist.dist_to_encoder(), ConditionalDistributionDataEncoder
            )

    def test_accumulator_type(self):
        """factory.make() must return a ConditionalDistributionAccumulator."""
        for f in self._factories:
            self.assertIsInstance(
                f.make(device=self.device), ConditionalDistributionAccumulator
            )

    def test_sample_tuple_structure(self):
        """Each sample must be a (given, obs) pair where given is in the dmap keys."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=50)
            for obs in data:
                # obs should be a 2-element tuple: (condition_val, observation)
                self.assertEqual(
                    len(obs), 2, f"Expected pair (given, obs), got length {len(obs)}"
                )
                cond_val = obs[0]
                self.assertIn(
                    cond_val, dist.dmap, f"Condition value {cond_val} not in dmap keys"
                )

    def test_log_density_finite(self):
        """log_density should return finite values for all samples."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=50)
            for obs in data:
                ll = dist.log_density(obs)
                self.assertTrue(
                    np.isfinite(ll), f"log_density returned {ll} for obs {obs}"
                )
