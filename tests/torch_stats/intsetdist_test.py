"""Tests for IntegerBernoulliSetDistribution and related torch_stats classes."""

# pylint: disable=duplicate-code,line-too-long,wildcard-import
# pylint: disable=unused-wildcard-import,unused-import

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.intsetdist import (
    IntegerBernoulliSetAccumulator,
    IntegerBernoulliSetAccumulatorFactory,
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetTorchSequence,
)
from tests.torch_stats.torch_stats_tests import *


class IntegerBernoulliSetDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            IntegerBernoulliSetDistribution(log_pvec=np.log([0.3, 0.6, 0.5, 0.2])),
            IntegerBernoulliSetDistribution(log_pvec=np.log([0.8, 0.2, 0.9])),
            IntegerBernoulliSetDistribution(log_pvec=np.log([0.1, 0.5, 0.3, 0.7, 0.4])),
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
        """dist_to_encoder() must return an IntegerBernoulliSetDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(
                dist.dist_to_encoder(), IntegerBernoulliSetDataEncoder
            )

    def test_accumulator_type(self):
        """factory.make() must return an IntegerBernoulliSetAccumulator."""
        for f in self._factories:
            self.assertIsInstance(
                f.make(device=self.device), IntegerBernoulliSetAccumulator
            )

    def test_sampler_returns_sets(self):
        """Each sample must be a set-like collection of non-negative integers."""
        data = self._dists[0].sampler(seed=1).sample(size=50)
        num_vals = len(self._dists[0].log_pvec)
        for obs in data:
            for elem in obs:
                self.assertGreaterEqual(elem, 0)
                self.assertLess(elem, num_vals)

    def test_inclusion_frequency(self):
        """Empirical inclusion frequency of each element should match its probability."""
        dist = IntegerBernoulliSetDistribution(log_pvec=np.log([0.7, 0.3, 0.5]))
        data = dist.sampler(seed=1).sample(size=5000)
        pvec = [0.7, 0.3, 0.5]
        for i, p in enumerate(pvec):
            freq = sum(1 for obs in data if i in obs) / len(data)
            self.assertAlmostEqual(freq, p, delta=0.04)
