"""Tests for GaussianDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.gaussian import (
    GaussianAccumulator,
    GaussianAccumulatorFactory,
    GaussianDataEncoder,
    GaussianTorchEncodedSequence,
)


class GaussianDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            GaussianDistribution(mu=0.0, sigma2=1.0),
            GaussianDistribution(mu=-5.0, sigma2=2.0),
            GaussianDistribution(mu=10.0, sigma2=0.5),
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
        """dist_to_encoder() must return a GaussianDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), GaussianDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a GaussianAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), GaussianAccumulator)
