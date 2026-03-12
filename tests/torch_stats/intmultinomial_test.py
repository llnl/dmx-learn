"""Tests for IntegerMultinomialDistribution and related torch_stats classes."""
import torch
import numpy as np
import pytest
from tests.torch_stats.torch_stats_tests import *
from dmx.torch_stats import *
from dmx.torch_stats.intmultinomial import (
    IntegerMultinomialAccumulator,
    IntegerMultinomialAccumulatorFactory,
    IntegerMultinomialDataEncoder,
    IntegerMultinomialTorchSequence,
)


class IntegerMultinomialDistributionTestCase(TorchStatsTestClass):

    def setUp(self) -> None:
        self.device = torch.device("cpu")

        self._dists = [
            IntegerMultinomialDistribution(
                min_val=0,
                p_vec=[0.2, 0.3, 0.5],
                len_dist=PoissonDistribution(lam=5.0),
            ),
            IntegerMultinomialDistribution(
                min_val=1,
                p_vec=[0.4, 0.3, 0.3],
                len_dist=PoissonDistribution(lam=8.0),
            ),
            IntegerMultinomialDistribution(
                min_val=0,
                p_vec=[0.1, 0.2, 0.3, 0.4],
                len_dist=PoissonDistribution(lam=3.0),
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
        """dist_to_encoder() must return an IntegerMultinomialDataEncoder."""
        for dist in self._dists:
            self.assertIsInstance(dist.dist_to_encoder(), IntegerMultinomialDataEncoder)

    def test_accumulator_type(self):
        """factory.make() must return an IntegerMultinomialAccumulator."""
        for f in self._factories:
            self.assertIsInstance(f.make(device=self.device), IntegerMultinomialAccumulator)

    def test_sampler_element_range(self):
        """Each (value, count) pair in samples must have value in [min_val, max_val]."""
        for dist in self._dists:
            data = dist.sampler(seed=1).sample(size=50)
            min_v = dist.min_val
            max_v = min_v + len(dist.p_vec) - 1
            for bag in data:
                for val, cnt in bag:
                    self.assertGreaterEqual(val, min_v,
                                            f"Value {val} below min_val={min_v}")
                    self.assertLessEqual(val, max_v,
                                         f"Value {val} above max_val={max_v}")
                    self.assertGreater(cnt, 0, "Count must be positive")
