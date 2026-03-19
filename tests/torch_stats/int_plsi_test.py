"""Tests for IntegerPLSIDistribution and related torch_stats classes.

IntegerPLSIDistribution expects:
  state_word_mat: shape (num_words, num_states), columns sum to 1.
  doc_state_mat:  shape (num_docs, num_states),  rows sum to 1.
  doc_vec:        shape (num_docs,),              sums to 1.

Each sample is (doc_id, [(word_id, count), ...]).
"""
import torch
import numpy as np
import pytest
from typing import Sequence, Tuple, cast
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.int_plsi import (
    IntegerPLSIAccumulator,
    IntegerPLSIAccumulatorFactory,
    IntegerPLSIDataEncoder,
    IntegerPLSITorchSequence,
)


class IntegerPLSIDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = get_test_torch_device()

        # 4 words, 3 states, 3 documents.
        # Columns of state_word_mat sum to 1 (p(word | state)).
        state_word_mat1 = np.array([
            [0.70, 0.05, 0.05],   # word 0
            [0.20, 0.70, 0.05],   # word 1
            [0.05, 0.20, 0.20],   # word 2
            [0.05, 0.05, 0.70],   # word 3
        ])  # shape (4, 3); cols sum to 1
        # Rows of doc_state_mat sum to 1 (p(state | doc)).
        doc_state_mat1 = np.array([
            [0.8, 0.1, 0.1],   # doc 0
            [0.1, 0.8, 0.1],   # doc 1
            [0.1, 0.1, 0.8],   # doc 2
        ])  # shape (3, 3); rows sum to 1
        doc_vec1 = np.array([0.4, 0.3, 0.3])

        self._dist1 = IntegerPLSIDistribution(
            state_word_mat=state_word_mat1,
            doc_state_mat=doc_state_mat1,
            doc_vec=doc_vec1,
            len_dist=PoissonDistribution(lam=10.0),
        )

        # 3 words, 2 states, 2 documents.
        state_word_mat2 = np.array([
            [0.6, 0.1],   # word 0
            [0.3, 0.3],   # word 1
            [0.1, 0.6],   # word 2
        ])  # shape (3, 2); cols sum to 1
        doc_state_mat2 = np.array([
            [0.9, 0.1],   # doc 0
            [0.1, 0.9],   # doc 1
        ])  # shape (2, 2); rows sum to 1
        doc_vec2 = np.array([0.5, 0.5])

        self._dist2 = IntegerPLSIDistribution(
            state_word_mat=state_word_mat2,
            doc_state_mat=doc_state_mat2,
            doc_vec=doc_vec2,
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
        """dist_to_encoder() must return an IntegerPLSIDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), IntegerPLSIDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return an IntegerPLSIAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), IntegerPLSIAccumulator)

    def test_sample_structure(self):
        """Each sample must be a (doc_id, [(word_id, count), ...]) pair."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            num_docs = dist.num_docs
            for obs in data:
                obs = cast(Tuple[int, Sequence[Tuple[int, float]]], obs)
                self.assertEqual(len(obs), 2,
                                 f"Expected (doc_id, word_counts) pair, got length {len(obs)}")
                doc_idx, word_counts = obs
                self.assertGreaterEqual(doc_idx, 0)
                self.assertLess(doc_idx, num_docs,
                                f"doc_id {doc_idx} out of [0, {num_docs - 1}]")

    def test_word_range(self):
        """All word indices in samples must be in [0, num_vals - 1]."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=50)
            num_vals = dist.num_vals
            for obs in data:
                obs = cast(Tuple[int, Sequence[Tuple[int, float]]], obs)
                _, word_counts = obs
                for w_id, cnt in word_counts:
                    self.assertGreaterEqual(w_id, 0)
                    self.assertLess(w_id, num_vals,
                                    f"Word index {w_id} out of [0, {num_vals - 1}]")
                    self.assertGreater(cnt, 0, "Count must be positive")

    def test_log_density_finite(self):
        """log_density must return finite values for sampled data."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=20)
            for obs in data:
                obs = cast(Tuple[int, Sequence[Tuple[int, float]]], obs)
                ll = dist.log_density(obs)
                self.assertTrue(np.isfinite(ll), f"log_density returned {ll}")

    def test_component_log_density_shape(self):
        """component_log_density must return a tensor with one value per state."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=1)
            obs = cast(Tuple[int, Sequence[Tuple[int, float]]], data[0])
            comp_ll = dist.component_log_density(obs)
            self.assertEqual(len(comp_ll), dist.num_states,
                             f"Expected {dist.num_states} component log densities")

    def test_component_log_density_values(self):
        """component_log_density must match the weighted log word probabilities per state."""
        obs = (0, [(0, 2.0), (1, 1.0), (3, 3.0)])
        expected = np.array([
            2.0 * np.log(0.70) + 1.0 * np.log(0.20) + 3.0 * np.log(0.05),
            2.0 * np.log(0.05) + 1.0 * np.log(0.70) + 3.0 * np.log(0.05),
            2.0 * np.log(0.05) + 1.0 * np.log(0.05) + 3.0 * np.log(0.70),
        ])

        comp_ll = self._dist1.component_log_density(obs).cpu().numpy()
        np.testing.assert_allclose(comp_ll, expected)
