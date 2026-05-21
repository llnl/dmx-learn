"""Tests for HiddenMarkovModelDistribution and related torch_stats classes."""

# pylint: disable=duplicate-code,line-too-long,wildcard-import
# pylint: disable=unused-wildcard-import,unused-import

from typing import List, cast

import numpy as np
import pytest
import torch

from dmx.torch_stats import *
from dmx.torch_stats.hmm import (
    HiddenMarkovAccumulator,
    HiddenMarkovAccumulatorFactory,
    HiddenMarkovDataEncoder,
    HiddenMarkovTorchSequence,
)
from tests.torch_stats.torch_stats_tests import *


class HiddenMarkovModelDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        transitions = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]

        self._dist1 = HiddenMarkovModelDistribution(
            topics=[
                GaussianDistribution(mu=-5.0, sigma2=1.0),
                GaussianDistribution(mu=0.0, sigma2=1.0),
                GaussianDistribution(mu=5.0, sigma2=1.0),
            ],
            w=[0.4, 0.4, 0.2],
            transitions=transitions,
            len_dist=PoissonDistribution(lam=8.0),
        )

        self._dist2 = HiddenMarkovModelDistribution(
            topics=[
                GaussianDistribution(mu=-3.0, sigma2=0.5),
                GaussianDistribution(mu=3.0, sigma2=0.5),
            ],
            w=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.1, 0.9]],
            len_dist=PoissonDistribution(lam=5.0),
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
        """dist_to_encoder() must return a HiddenMarkovDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), HiddenMarkovDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return a HiddenMarkovAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), HiddenMarkovAccumulator)

    def test_sample_is_list(self):
        """Each sample must be a list/sequence of observations."""
        data = self._dist1.sampler(seed=1).sample(size=20)
        for seq in data:
            self.assertIsInstance(
                seq, (list, np.ndarray), f"Expected sequence, got {type(seq)}"
            )

    def test_log_density_finite(self):
        """log_density must return finite values for sampled observations."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=10)
            for seq in data:
                ll = dist.log_density(cast(List[float], seq))
                self.assertTrue(np.isfinite(ll), f"log_density returned {ll}")

    def test_viterbi_state_range(self):
        """viterbi() must return state indices within [0, num_states-1] for each step."""
        for dist in self._dists:
            num_states = len(dist.topics)
            # viterbi() accepts a single raw sequence (List[T])
            seq = cast(List[float], dist.sampler(seed=1).sample(size=5)[0])
            states = dist.viterbi(seq)
            self.assertTrue(
                all(0 <= int(s) < num_states for s in states),
                f"Viterbi state out of [0, {num_states - 1}]: {states}",
            )

    def test_viterbi_length_matches_data(self):
        """viterbi() state sequence must match the length of the observed sequence."""
        data = self._dist1.sampler(seed=1).sample(size=5)
        for seq in data:
            seq_list = cast(List[float], seq)
            states = self._dist1.viterbi(seq_list)
            self.assertEqual(
                len(states), len(seq_list), "Viterbi state sequence length mismatch"
            )

    def test_seq_initialize_with_empty_sequences(self):
        """seq_initialize must handle batches containing empty sequences."""
        data = [[], [0.5, -0.2], [], [1.3], [0.0, 0.1, -0.1]]
        encoder = self._dist2.dist_to_encoder()
        enc_data = [(len(data), encoder.seq_encode(data, device=self.device))]

        model = seq_initialize(
            enc_data=enc_data,
            estimator=self._dist2.estimator(),
            seed=11,
            p=1.0,
            device=self.device,
        )
        _, ll = seq_log_density_sum(enc_data, model)

        self.assertIsInstance(model, HiddenMarkovModelDistribution)
        self.assertTrue(np.isfinite(ll), f"Expected finite log-likelihood, got {ll}")
