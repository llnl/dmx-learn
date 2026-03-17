"""Tests for ExponentialDistribution and related torch_stats classes."""
import torch
import numpy as np
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.exponential import (
    ExponentialAccumulator,
    ExponentialAccumulatorFactory,
    ExponentialDataEncoder,
    ExponentialTorchEncodedSequence,
)


class ExponentialDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = torch.device("cpu")

        self._dists = [
            ExponentialDistribution(beta=1.0),
            ExponentialDistribution(beta=0.5),
            ExponentialDistribution(beta=2.0),
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
            with self.assertRaises(Exception):
                self._dists[0].seq_log_density(bad_input)

    def test_encoder_type(self):
        """dist_to_encoder() must return an ExponentialDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), ExponentialDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return an ExponentialAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), ExponentialAccumulator)

    def test_sampler_positive(self):
        """All sampled values must be positive (exponential has support on (0, inf))."""
        data = self._dists[0].sampler(seed=1).sample(size=500)
        self.assertTrue(np.all(np.asarray(data) > 0), "Exponential samples must be positive")

    def test_beta_effect(self):
        """Larger beta must yield larger mean sample value."""
        n = 2000
        mean1 = float(np.mean(self._dists[1].sampler(seed=42).sample(n)))  # beta=0.5
        mean2 = float(np.mean(self._dists[2].sampler(seed=42).sample(n)))  # beta=2.0
        self.assertLess(mean1, mean2)

    def test_estimator_uses_mean_parameterization(self):
        """Estimator must recover beta as weighted sum divided by weighted count."""
        est = ExponentialEstimator()
        model = est.estimate(nobs=None, suff_stat=(6.0, 3.0), device=self.device)
        self.assertAlmostEqual(model.beta, 2.0)
