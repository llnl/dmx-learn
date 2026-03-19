"""Tests for HeterogeneousMixtureDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.heterogenous_mixture import (
    HeterogeneousMixtureAccumulator,
    HeterogeneousMixtureAccumulatorFactory,
    HeterogeneousMixtureDataEncoder,
    HeterogeneousMixtureTorchSequence,
)


class HeterogeneousMixtureDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        self._dists = [
            HeterogeneousMixtureDistribution(
                components=[
                    PoissonDistribution(lam=3.0),
                    BinomialDistribution(p=0.4, n=10),
                ],
                w=[0.5, 0.5],
            ),
            HeterogeneousMixtureDistribution(
                components=[
                    PoissonDistribution(lam=1.0),
                    PoissonDistribution(lam=8.0),
                    BinomialDistribution(p=0.7, n=5),
                ],
                w=[0.3, 0.4, 0.3],
            ),
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
        """dist_to_encoder() must return a HeterogeneousMixtureDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), HeterogeneousMixtureDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a HeterogeneousMixtureAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), HeterogeneousMixtureAccumulator)

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

    def test_num_components(self):
        """num_components must match the length of the component list."""
        for dist in self._dists:
            self.assertEqual(dist.num_components, len(dist.components))

    def test_seq_estimate_with_shared_encoder_group(self):
        """seq_estimate must handle multiple components that share one encoder type."""
        dist = self._dists[1]
        encoder = self._encoders[1]
        data = dist.sampler(seed=7).sample(size=200)
        enc_data = [(len(data), encoder.seq_encode(data, device=self.device))]

        init_model = seq_initialize(
            enc_data=enc_data,
            estimator=dist.estimator(),
            seed=7,
            device=self.device,
        )
        next_model = seq_estimate(
            enc_data=enc_data,
            estimator=dist.estimator(),
            prev_estimate=init_model,
        )
        _, ll = seq_log_density_sum(enc_data, next_model)

        self.assertIsInstance(next_model, HeterogeneousMixtureDistribution)
        self.assertTrue(np.isfinite(ll), f"Expected finite log-likelihood, got {ll}")
