"""Tests for MixtureDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.mixture import (
    MixtureAccumulator,
    MixtureAccumulatorFactory,
    MixtureDataEncoder,
    MixtureTorchEncodedSequence,
)


class MixtureDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            MixtureDistribution(
                components=[
                    GaussianDistribution(mu=-3.0, sigma2=1.0),
                    GaussianDistribution(mu=3.0, sigma2=1.0),
                ],
                w=[0.6, 0.4],
            ),
            MixtureDistribution(
                components=[
                    GaussianDistribution(mu=-5.0, sigma2=0.5),
                    GaussianDistribution(mu=0.0, sigma2=1.0),
                    GaussianDistribution(mu=5.0, sigma2=0.5),
                ],
                w=[0.3, 0.4, 0.3],
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
        """dist_to_encoder() must return a MixtureDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), MixtureDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a MixtureAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), MixtureAccumulator)

    def test_seq_posterior_sums_to_one(self):
        """Posterior probabilities must sum to 1 across components for each observation."""
        for dist, encoder in zip(self._dists, self._encoders):
            data = dist.sampler(seed=1).sample(size=100)
            enc = encoder.seq_encode(data, device=self.device)
            posterior = dist.seq_posterior(enc)
            row_sums = posterior.sum(dim=1)
            self.assertTrue(
                torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
                f"Posterior rows do not sum to 1: {row_sums[:5]}",
            )

    def test_seq_component_log_density_consistency(self):
        """seq_component_log_density should be consistent with seq_log_density.

        The log-sum-exp of (log_weights + component_log_densities) must equal
        seq_log_density up to tolerance.
        """
        import math
        for dist, encoder in zip(self._dists, self._encoders):
            data = dist.sampler(seed=1).sample(size=50)
            enc = encoder.seq_encode(data, device=self.device)
            comp_ll = dist.seq_component_log_density(enc)   # shape (n, k)
            log_w = torch.log(dist.w)                        # shape (k,)
            # log-sum-exp manually: logsumexp over components
            combined = comp_ll + log_w.unsqueeze(0)
            lse = torch.logsumexp(combined, dim=1)
            seq_ll = dist.seq_log_density(enc)
            self.assertTrue(
                torch.allclose(lse, seq_ll, atol=1e-5),
                "seq_component_log_density inconsistent with seq_log_density",
            )

    def test_num_components(self):
        """num_components must match the length of the component list."""
        for dist in self._dists:
            self.assertEqual(dist.num_components, len(dist.components))
