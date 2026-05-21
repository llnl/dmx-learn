"""Tests for SequenceDistribution and related torch_stats classes."""

# pylint: disable=duplicate-code,line-too-long,wildcard-import
# pylint: disable=unused-wildcard-import,unused-import

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.sequence import (
    SequenceAccumulator,
    SequenceAccumulatorFactory,
    SequenceDataEncoder,
    SequenceTorchEncodedSequence,
)
from tests.torch_stats.torch_stats_tests import *


class SequenceDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        # Exponential base with IntegerCategorical length distribution
        self._dist1 = SequenceDistribution(
            dist=ExponentialDistribution(beta=1.0),
            len_dist=IntegerCategoricalDistribution(
                min_val=3, p_vec=np.array([0.5, 0.0, 0.5])
            ),
            len_normalized=True,
        )

        # Gaussian base with Poisson length distribution
        self._dist2 = SequenceDistribution(
            dist=GaussianDistribution(mu=0.0, sigma2=1.0),
            len_dist=PoissonDistribution(lam=5.0),
            len_normalized=False,
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
        """dist_to_encoder() must return a SequenceDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), SequenceDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a SequenceAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), SequenceAccumulator)

    def test_sample_is_list(self):
        """Each sample must be a list/sequence of observations."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            for seq in data:
                self.assertIsInstance(
                    seq, (list, np.ndarray), f"Expected sequence, got {type(seq)}"
                )

    def test_sample_length_distribution(self):
        """Sample lengths should follow the specified length distribution (approx)."""
        # dist1 has support on {3, 5} only
        data = self._dist1.sampler(seed=1).sample(size=1000)
        lengths = [len(seq) for seq in data]
        valid_lengths = set(lengths)
        for l in valid_lengths:
            self.assertIn(l, {3, 5}, f"Unexpected sequence length {l}")

    def test_log_density_finite(self):
        """log_density must return finite values for sampled data."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            for seq in data:
                ll = dist.log_density(seq)
                self.assertTrue(np.isfinite(ll), f"log_density returned {ll}")
