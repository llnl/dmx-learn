"""Base test cases for torch statistical distributions and their properties.

This module provides a set of unit tests for torch_stats distributions,
mirroring tests/stats/stats_tests.py but adapted for the torch_stats API.

Key differences from the stats version:
- TorchProbabilityDistribution has no __eq__, so equality-based tests are
  replaced with isinstance / type-correctness checks.
- Encoders accept an optional ``device=`` argument:
  ``encoder.seq_encode(data, device=device)``.
- seq_initialize uses a seed (int) instead of a numpy RandomState.
- All log-density comparisons use torch tensors; tolerance is 1e-10 for
  float64 (CPU/CUDA) and 1e-4 for float32 (MPS).
- Tests include a device-movement check: model.to(torch.device('cpu')).

Tests provided:
- test_01_sampler            – same seed yields identical samples
- test_02_log_density        – scalar log_density matches seq_log_density elementwise
- test_03_dist_to_encoder    – dist.dist_to_encoder() returns TorchSequenceEncoder
- test_04_estimator          – dist.estimator() returns TorchParameterEstimator
- test_05_estimator_factory  - est.accumulator_factory() returns
  TorchStatisticAccumulatorFactory
- test_06_factory_make       – factory.make() returns TorchStatisticAccumulator
- test_07_acc_to_encoder     – acc.acc_to_encoder() returns TorchSequenceEncoder
- test_08_seq_update         - one EM step from seq_initialize does not
  decrease log-likelihood
- test_09_seq_initialize     - seq_initialize produces a model with
  finite log-likelihood
- test_10_device             – fitted model can be moved to CPU without error

Required setUp attributes
--------------------------
self.sampler_dist         : TorchProbabilityDistribution
    Distribution used for sampler-repeatability test.

self.density_dist_encoder : List[
    Tuple[TorchProbabilityDistribution, TorchSequenceEncoder]
]
    Pairs used for log-density, seq_update, and seq_initialize tests.

self.dist_encoder         : List[
    Tuple[TorchProbabilityDistribution, TorchSequenceEncoder]
]
    Pairs used for the dist_to_encoder type-correctness check.

self.estimators           : List[TorchParameterEstimator]
    Estimators whose accumulator_factory() is type-checked.

self.factories            : List[TorchStatisticAccumulatorFactory]
    Factories whose make() return type is checked.

self.accumulators         : List[TorchStatisticAccumulator]
    Accumulators whose acc_to_encoder() return type is checked.

self.device               : torch.device  (default: get_test_torch_device())
    Device used when encoding data and initialising models in tests.
"""

# pylint: disable=duplicate-code,line-too-long,broad-exception-caught

import abc
import os
import unittest

import numpy as np
import pytest
import torch

from dmx.torch_stats import seq_estimate, seq_initialize, seq_log_density_sum
from dmx.torch_stats.pdist import (
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)
from dmx.torch_utils.vector import float_dtype_for_device, set_default_float_dtype

# ---------------------------------------------------------------------------
# Tolerance: float64 on CPU/CUDA gives ~1e-14; we allow 1e-10 as a safe
# margin.  MPS uses float32 so we relax to 1e-4 there.
# ---------------------------------------------------------------------------
_LOG_DENSITY_TOL_F64 = 1.0e-10
_LOG_DENSITY_TOL_F32 = 1.0e-4


def get_test_torch_device() -> torch.device:
    """Resolve the torch device for tests from TEST_TORCH_DEVICE."""
    raw_device = os.environ.get("TEST_TORCH_DEVICE", "cpu").strip().lower()

    if raw_device == "cpu":
        return torch.device("cpu")

    if raw_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "TEST_TORCH_DEVICE=mps requested, but MPS is not available in this environment."
            )
        return torch.device("mps")

    if raw_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"TEST_TORCH_DEVICE={raw_device} requested, but CUDA is not available in this environment."
            )
        return torch.device(raw_device)

    raise RuntimeError(
        f"Unsupported TEST_TORCH_DEVICE={raw_device!r}. Use cpu, mps, cuda, or cuda:<index>."
    )


set_default_float_dtype(float_dtype_for_device(get_test_torch_device()))


def _tol_for_device(device: torch.device) -> float:
    return _LOG_DENSITY_TOL_F32 if device.type == "mps" else _LOG_DENSITY_TOL_F64


# ---------------------------------------------------------------------------
# Module-level helper functions (mirror of stats_tests.py style)
# ---------------------------------------------------------------------------


def sampler_repeat_test(dist: TorchProbabilityDistribution):
    """Verify that the same seed produces identical samples across two calls.

    Args:
        dist: Distribution instance with a ``sampler(seed)`` method.

    Returns:
        Tuple[bool, List[bool]]: (all_passed, per_seed_results)
    """
    seeds = [1, 2, 3]
    sz = 20
    per_seed = []
    for seed in seeds:
        d1 = dist.sampler(seed).sample(size=sz)
        d2 = dist.sampler(seed).sample(size=sz)
        same = [str(a) == str(b) for a, b in zip(d1, d2)]
        per_seed.append(all(same))
    return all(per_seed), per_seed


def log_density_test(
    dist: TorchProbabilityDistribution,
    encoder: TorchSequenceEncoder,
    device: torch.device = torch.device("cpu"),
):
    """Verify that scalar log_density matches the corresponding element of seq_log_density.

    For each seed, samples are drawn and encoded; then for every observation the
    relative discrepancy between ``dist.log_density(x)`` and the matching
    element of ``dist.seq_log_density(enc)`` is computed.  If the seq value is
    zero the absolute value of the scalar result is used instead.

    Args:
        dist:    Distribution instance.
        encoder: Encoder compatible with ``dist``.
        device:  Torch device used for encoding.

    Returns:
        Tuple[bool, str]: (passed, description)
    """
    tol = _tol_for_device(device)
    seeds = [1, 2, 3]
    sz = 20
    max_discrepancy = 0.0

    for seed in seeds:
        data = dist.sampler(seed).sample(size=sz)
        try:
            enc_seq = encoder.seq_encode(data, device=device)
        except Exception as exc:
            return False, f"encoder.seq_encode raised: {exc}"

        seq_ll = dist.seq_log_density(enc_seq)

        for i in range(sz):
            scalar_ll = dist.log_density(data[i])
            seq_ll_i = float(seq_ll[i])
            if seq_ll_i == 0.0:
                disc = abs(scalar_ll)
            else:
                disc = abs(seq_ll_i - scalar_ll) / abs(seq_ll_i)
            max_discrepancy = max(max_discrepancy, disc)

    passed = max_discrepancy < tol
    return passed, f"max relative discrepancy = {max_discrepancy:.3e} (tol={tol:.0e})"


def dist_to_encoder_type_test(dist: TorchProbabilityDistribution):
    """Verify that dist.dist_to_encoder() returns a TorchSequenceEncoder.

    Args:
        dist: Distribution instance.

    Returns:
        bool: True if the returned object is an instance of TorchSequenceEncoder.
    """
    enc = dist.dist_to_encoder()
    return isinstance(enc, TorchSequenceEncoder)


def estimator_type_test(dist: TorchProbabilityDistribution):
    """Verify that dist.estimator() returns a TorchParameterEstimator.

    Args:
        dist: Distribution instance.

    Returns:
        bool: True if the returned object is an instance of TorchParameterEstimator.
    """
    est = dist.estimator()
    return isinstance(est, TorchParameterEstimator)


def estimator_factory_type_test(estimator: TorchParameterEstimator):
    """Verify that estimator.accumulator_factory() returns a TorchStatisticAccumulatorFactory.

    Args:
        estimator: Estimator instance.

    Returns:
        bool: True if the returned object is an instance of TorchStatisticAccumulatorFactory.
    """
    factory = estimator.accumulator_factory()
    return isinstance(factory, TorchStatisticAccumulatorFactory)


def factory_make_type_test(
    factory: TorchStatisticAccumulatorFactory,
    device: torch.device = torch.device("cpu"),
):
    """Verify that factory.make() returns a TorchStatisticAccumulator.

    Args:
        factory: Factory instance.
        device:  Device passed to make().

    Returns:
        bool: True if the returned object is an instance of TorchStatisticAccumulator.
    """
    acc = factory.make(device=device)
    return isinstance(acc, TorchStatisticAccumulator)


def acc_to_encoder_type_test(acc: TorchStatisticAccumulator):
    """Verify that acc.acc_to_encoder() returns a TorchSequenceEncoder.

    Args:
        acc: Accumulator instance.

    Returns:
        bool: True if the returned object is an instance of TorchSequenceEncoder.
    """
    enc = acc.acc_to_encoder()
    return isinstance(enc, TorchSequenceEncoder)


def seq_update_test(
    dist: TorchProbabilityDistribution,
    encoder: TorchSequenceEncoder,
    device: torch.device = torch.device("cpu"),
):
    """Verify that one EM step (seq_estimate) does not decrease the log-likelihood.

    For each seed, data is sampled and encoded; a model is initialised with
    seq_initialize and then updated once with seq_estimate.  The test checks
    that the log-likelihood of the updated model is >= that of the initial model.

    Args:
        dist:    Distribution instance (provides sampler and estimator).
        encoder: Encoder compatible with ``dist``.
        device:  Torch device.

    Returns:
        Tuple[bool, List[bool]]: (all_passed, per_seed_results)
    """
    seeds = [1, 2, 3]
    sz = 1000
    per_seed = []

    for seed in seeds:
        data = dist.sampler(seed=seed).sample(sz)
        enc_data = [(sz, encoder.seq_encode(data, device=device))]

        est = dist.estimator()
        prev_model = seq_initialize(
            enc_data=enc_data, estimator=est, seed=seed, device=device
        )
        next_model = seq_estimate(
            enc_data=enc_data, estimator=est, prev_estimate=prev_model
        )

        _, ll_prev = seq_log_density_sum(enc_data, prev_model)
        _, ll_next = seq_log_density_sum(enc_data, next_model)

        per_seed.append(ll_next >= ll_prev)

    return all(per_seed), per_seed


def seq_initialize_test(
    dist: TorchProbabilityDistribution,
    encoder: TorchSequenceEncoder,
    device: torch.device = torch.device("cpu"),
):
    """Verify that seq_initialize produces a model with a finite log-likelihood.

    Args:
        dist:    Distribution instance.
        encoder: Encoder compatible with ``dist``.
        device:  Torch device.

    Returns:
        Tuple[bool, List[float]]: (all_passed, per_seed_total_lls)
    """
    seeds = [1, 2, 3]
    sz = 500
    lls = []

    for seed in seeds:
        data = dist.sampler(seed=seed).sample(sz)
        enc_data = [(sz, encoder.seq_encode(data, device=device))]

        est = dist.estimator()
        model = seq_initialize(
            enc_data=enc_data, estimator=est, seed=seed, device=device
        )
        _, ll = seq_log_density_sum(enc_data, model)
        lls.append(ll)

    all_finite = all(np.isfinite(ll) for ll in lls)
    return all_finite, lls


def device_test(
    dist: TorchProbabilityDistribution,
    encoder: TorchSequenceEncoder,
    device: torch.device = torch.device("cpu"),
):
    """Verify that a fitted model can be moved to CPU without raising an error.

    Fits a small model with seq_initialize then calls model.to(torch.device('cpu')).

    Args:
        dist:    Distribution instance.
        encoder: Encoder compatible with ``dist``.
        device:  Device used during fitting.

    Returns:
        Tuple[bool, str]: (passed, description)
    """
    sz = 200
    data = dist.sampler(seed=42).sample(sz)
    enc_data = [(sz, encoder.seq_encode(data, device=device))]
    est = dist.estimator()
    try:
        model = seq_initialize(enc_data=enc_data, estimator=est, seed=42, device=device)
        model.to(torch.device("cpu"))
        return True, "device movement succeeded"
    except Exception as exc:
        return False, f"device movement raised: {exc}"


# ---------------------------------------------------------------------------
# Abstract base test class
# ---------------------------------------------------------------------------


class TorchStatsTestClass(unittest.TestCase, metaclass=abc.ABCMeta):
    """Abstract base class for torch_stats distribution test cases.

    Subclasses must implement ``setUp`` and populate the attributes listed
    in the module docstring.  All test methods are inherited automatically.

    Minimal setUp example::

        def setUp(self):
            dist = GaussianDistribution(mu=0.0, sigma2=1.0)
            encoder = dist.dist_to_encoder()

            self.device              = get_test_torch_device()
            self.sampler_dist        = dist
            self.density_dist_encoder = [(dist, encoder)]
            self.dist_encoder        = [(dist, encoder)]
            self.estimators          = [GaussianEstimator()]
            self.factories           = [GaussianEstimator().accumulator_factory()]
            self.accumulators        = [
                GaussianEstimator().accumulator_factory().make()
            ]
    """

    @abc.abstractmethod
    def setUp(self):
        """Populate the required test attributes.

        Subclasses must set:
            self.device               torch.device
            self.sampler_dist         TorchProbabilityDistribution
            self.density_dist_encoder List[(dist, encoder)]
            self.dist_encoder         List[(dist, encoder)]
            self.estimators           List[TorchParameterEstimator]
            self.factories            List[TorchStatisticAccumulatorFactory]
            self.accumulators         List[TorchStatisticAccumulator]
        """
        self.device: torch.device = get_test_torch_device()
        self.sampler_dist: TorchProbabilityDistribution
        self.density_dist_encoder: list
        self.dist_encoder: list
        self.estimators: list
        self.factories: list
        self.accumulators: list

    # ------------------------------------------------------------------
    # Test 01 – sampler repeatability
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_sampler")
    def test_01_sampler(self):
        """Same seed must produce identical samples on repeated calls."""
        passed, per_seed = sampler_repeat_test(self.sampler_dist)
        self.assertTrue(passed, f"Sampler not repeatable: {per_seed}")

    # ------------------------------------------------------------------
    # Test 02 – scalar vs vectorised log density
    # ------------------------------------------------------------------
    @pytest.mark.dependency(depends=["torch_sampler"], name="torch_log_density")
    def test_02_log_density(self):
        """log_density(x) must match the corresponding element of seq_log_density."""
        for dist, encoder in self.density_dist_encoder:
            passed, msg = log_density_test(dist, encoder, device=self.device)
            self.assertTrue(passed, msg)

    # ------------------------------------------------------------------
    # Test 03 – dist_to_encoder return type
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_dist_to_encoder")
    def test_03_dist_to_encoder(self):
        """dist.dist_to_encoder() must return a TorchSequenceEncoder."""
        for dist, _ in self.dist_encoder:
            self.assertTrue(
                dist_to_encoder_type_test(dist),
                f"{type(dist).__name__}.dist_to_encoder() did not return TorchSequenceEncoder",
            )

    # ------------------------------------------------------------------
    # Test 04 – estimator return type
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_estimator")
    def test_04_estimator(self):
        """dist.estimator() must return a TorchParameterEstimator."""
        for dist, _ in self.dist_encoder:
            self.assertTrue(
                estimator_type_test(dist),
                f"{type(dist).__name__}.estimator() did not return TorchParameterEstimator",
            )

    # ------------------------------------------------------------------
    # Test 05 – accumulator factory return type
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_estimator_factory")
    def test_05_estimator_factory(self):
        """est.accumulator_factory() must return a TorchStatisticAccumulatorFactory."""
        for est in self.estimators:
            self.assertTrue(
                estimator_factory_type_test(est),
                f"{type(est).__name__}.accumulator_factory() did not return "
                "TorchStatisticAccumulatorFactory",
            )

    # ------------------------------------------------------------------
    # Test 06 – factory.make() return type
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_factory_make")
    def test_06_factory_make(self):
        """factory.make() must return a TorchStatisticAccumulator."""
        for factory in self.factories:
            self.assertTrue(
                factory_make_type_test(factory, device=self.device),
                f"{type(factory).__name__}.make() did not return TorchStatisticAccumulator",
            )

    # ------------------------------------------------------------------
    # Test 07 – acc_to_encoder return type
    # ------------------------------------------------------------------
    @pytest.mark.dependency(name="torch_acc_to_encoder")
    def test_07_acc_to_encoder(self):
        """acc.acc_to_encoder() must return a TorchSequenceEncoder."""
        for acc in self.accumulators:
            self.assertTrue(
                acc_to_encoder_type_test(acc),
                f"{type(acc).__name__}.acc_to_encoder() did not return TorchSequenceEncoder",
            )

    # ------------------------------------------------------------------
    # Test 08 – seq_update does not decrease log-likelihood
    # ------------------------------------------------------------------
    @pytest.mark.dependency(
        depends=[
            "torch_sampler",
            "torch_log_density",
            "torch_estimator_factory",
            "torch_factory_make",
        ],
        name="torch_seq_update",
    )
    def test_08_seq_update(self):
        """One EM step must not decrease the total log-likelihood."""
        for dist, encoder in self.density_dist_encoder:
            passed, per_seed = seq_update_test(dist, encoder, device=self.device)
            self.assertTrue(passed, f"seq_update decreased LL: {per_seed}")

    # ------------------------------------------------------------------
    # Test 09 – seq_initialize produces a model with finite LL
    # ------------------------------------------------------------------
    @pytest.mark.dependency(
        depends=["torch_sampler", "torch_estimator_factory"],
        name="torch_seq_initialize",
    )
    def test_09_seq_initialize(self):
        """seq_initialize must return a model whose total log-likelihood is finite."""
        for dist, encoder in self.density_dist_encoder:
            passed, lls = seq_initialize_test(dist, encoder, device=self.device)
            self.assertTrue(passed, f"seq_initialize produced non-finite LL: {lls}")

    # ------------------------------------------------------------------
    # Test 10 – device movement
    # ------------------------------------------------------------------
    @pytest.mark.dependency(depends=["torch_seq_initialize"], name="torch_device")
    def test_10_device(self):
        """A fitted model must be movable to CPU via model.to(torch.device('cpu'))."""
        for dist, encoder in self.density_dist_encoder:
            passed, msg = device_test(dist, encoder, device=self.device)
            self.assertTrue(passed, msg)
