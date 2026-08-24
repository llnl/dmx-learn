"""Reusable pytest checks for Bayesian probability distributions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pytest

from dmx.bstats import dump_models, load_models

_OPTIONAL_METHODS = frozenset(
    {
        "string_round_trip",
        "get_prior",
        "set_prior",
        "expected_log_density",
        "seq_expected_log_density",
        "entropy",
        "cross_entropy",
    }
)


@dataclass(frozen=True)
class BayesianDistributionTestCase:
    """Describe one distribution to exercise with the shared harness.

    Args:
        distribution_factory: Return a fresh distribution for each test.
        observations: Representative observations accepted by the distribution.
        alternate_prior_factory: Return a prior used to check ``set_prior``.
        unsupported_methods: Optional methods to skip, mapped to explicit reasons.
        sampler_seeds: Seeds used for deterministic sampler checks.
        sample_size: Number of observations drawn for each sampler check.
    """

    __test__: ClassVar[bool] = False

    distribution_factory: Callable[[], Any]
    observations: Sequence[Any]
    alternate_prior_factory: Callable[[], Any] | None = None
    unsupported_methods: Mapping[str, str] = field(default_factory=dict)
    sampler_seeds: tuple[int, ...] = (1, 2, 3)
    sample_size: int = 20

    def __post_init__(self) -> None:
        """Validate capability declarations when the test module is imported."""
        unknown = set(self.unsupported_methods) - _OPTIONAL_METHODS
        if unknown:
            raise ValueError(f"Unknown optional methods: {sorted(unknown)}")
        if any(not reason.strip() for reason in self.unsupported_methods.values()):
            raise ValueError("Unsupported methods require a non-empty skip reason")
        if "set_prior" not in self.unsupported_methods:
            if "get_prior" in self.unsupported_methods:
                raise ValueError("set_prior checks require get_prior support")
            if self.alternate_prior_factory is None:
                raise ValueError("set_prior checks require an alternate prior factory")
        if "seq_expected_log_density" not in self.unsupported_methods:
            if "expected_log_density" in self.unsupported_methods:
                raise ValueError(
                    "sequence expected-density checks require scalar support"
                )


def assert_string_round_trip(distribution: Any) -> None:
    """Assert that the package string serializer reconstructs a distribution."""
    serialized = dump_models(distribution)
    restored = load_models(serialized)

    assert type(restored) is type(distribution)
    assert dump_models(restored) == serialized


def assert_sampler_repeatable(
    distribution: Any, seeds: Sequence[int], sample_size: int
) -> None:
    """Assert repeated samplers yield identical draws for every fixed seed."""
    for seed in seeds:
        first = distribution.sampler(seed=seed).sample(size=sample_size)
        second = distribution.sampler(seed=seed).sample(size=sample_size)
        np.testing.assert_equal(first, second)


def assert_scalar_sequence_density(
    distribution: Any, observations: Sequence[Any]
) -> None:
    """Assert scalar and encoded sequence log-density paths agree."""
    encoded = distribution.seq_encode(observations)
    scalar = np.asarray(
        [distribution.log_density(value) for value in observations], dtype=float
    )
    sequence = np.asarray(distribution.seq_log_density(encoded), dtype=float)
    np.testing.assert_allclose(sequence, scalar, rtol=1.0e-12, atol=1.0e-12)


def assert_scalar_sequence_expected_log_density(
    distribution: Any, observations: Sequence[Any]
) -> None:
    """Assert scalar and sequence expected log-density paths agree."""
    encoded = distribution.seq_encode(observations)
    scalar = np.asarray(
        [distribution.expected_log_density(value) for value in observations],
        dtype=float,
    )
    sequence = np.asarray(distribution.seq_expected_log_density(encoded), dtype=float)
    np.testing.assert_allclose(sequence, scalar, rtol=1.0e-12, atol=1.0e-12)


def _assert_values_equal(actual: Any, expected: Any) -> None:
    """Compare nested sufficient statistics, including NumPy arrays."""
    if actual is None or expected is None:
        assert actual is expected
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_values_equal(actual_value, expected_value)
        return
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_values_equal(actual[key], expected[key])
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if (
        actual_array.dtype.kind not in "biufc"
        or expected_array.dtype.kind not in "biufc"
    ):
        np.testing.assert_equal(actual, expected)
        return
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


class BayesianDistributionTests:
    """Pytest harness inherited by distribution-specific test classes."""

    case: ClassVar[BayesianDistributionTestCase]

    def _distribution(self) -> Any:
        """Return an isolated distribution instance for one test."""
        return self.case.distribution_factory()

    def _skip_if_unsupported(self, method: str) -> None:
        """Skip an optional check only when the case gives an explicit reason."""
        reason = self.case.unsupported_methods.get(method)
        if reason is not None:
            pytest.skip(f"{method} is unsupported: {reason}")

    def test_string_round_trip(self) -> None:
        """Check constructor-like string serialization when supported."""
        self._skip_if_unsupported("string_round_trip")
        assert_string_round_trip(self._distribution())

    def test_sampler_repeatability(self) -> None:
        """Check fixed seeds reproduce the same samples."""
        assert_sampler_repeatable(
            self._distribution(), self.case.sampler_seeds, self.case.sample_size
        )

    def test_scalar_sequence_log_density(self) -> None:
        """Check scalar and vectorized log densities agree."""
        assert_scalar_sequence_density(self._distribution(), self.case.observations)

    def test_estimator(self) -> None:
        """Check that the distribution creates stable estimator types."""
        distribution = self._distribution()
        assert type(distribution.estimator()) is type(distribution.estimator())

    def test_accumulator_factory(self) -> None:
        """Check that the estimator factory creates stable accumulator types."""
        factory = self._distribution().estimator().accumulator_factory()
        assert type(factory.make()) is type(factory.make())

    def test_accumulator_value(self) -> None:
        """Check scalar, sequence, and restored sufficient statistics agree."""
        distribution = self._distribution()
        factory = distribution.estimator().accumulator_factory()
        scalar_accumulator = factory.make()
        for observation in self.case.observations:
            scalar_accumulator.update(observation, 1.0, distribution)
        scalar_value = scalar_accumulator.value()

        sequence_accumulator = factory.make()
        encoded = distribution.seq_encode(self.case.observations)
        weights = np.ones(len(self.case.observations), dtype=float)
        sequence_accumulator.seq_update(encoded, weights, distribution)
        _assert_values_equal(sequence_accumulator.value(), scalar_value)

        restored_accumulator = factory.make()
        restored_accumulator.from_value(copy.deepcopy(scalar_value))
        _assert_values_equal(restored_accumulator.value(), scalar_value)

    def test_encoders(self) -> None:
        """Check distribution and accumulator encoders agree on encoded data."""
        distribution = self._distribution()
        distribution_encoder = distribution.dist_to_encoder()
        accumulator = distribution.estimator().accumulator_factory().make()
        accumulator_encoder = accumulator.acc_to_encoder()

        assert accumulator_encoder == distribution_encoder
        encoded = distribution_encoder.seq_encode(self.case.observations)
        _assert_values_equal(
            encoded.data, distribution.seq_encode(self.case.observations)
        )

    def test_get_prior(self) -> None:
        """Check that a fresh distribution exposes its configured prior."""
        self._skip_if_unsupported("get_prior")
        distribution = self._distribution()
        comparison = self._distribution()
        assert type(distribution.get_prior()) is type(comparison.get_prior())
        assert str(distribution.get_prior()) == str(comparison.get_prior())

    def test_set_prior(self) -> None:
        """Check that replacing a prior is observable through ``get_prior``."""
        self._skip_if_unsupported("set_prior")
        assert self.case.alternate_prior_factory is not None
        distribution = self._distribution()
        alternate_prior = self.case.alternate_prior_factory()

        distribution.set_prior(alternate_prior)

        assert distribution.get_prior() is alternate_prior

    def test_expected_log_density(self) -> None:
        """Check scalar expected log densities are finite."""
        self._skip_if_unsupported("expected_log_density")
        distribution = self._distribution()
        values = [
            distribution.expected_log_density(value) for value in self.case.observations
        ]
        assert np.all(np.isfinite(values))

    def test_seq_expected_log_density(self) -> None:
        """Check scalar and sequence expected log densities agree."""
        self._skip_if_unsupported("seq_expected_log_density")
        assert_scalar_sequence_expected_log_density(
            self._distribution(), self.case.observations
        )

    def test_entropy(self) -> None:
        """Check entropy is finite when implemented."""
        self._skip_if_unsupported("entropy")
        assert np.isfinite(self._distribution().entropy())

    def test_cross_entropy(self) -> None:
        """Check self cross-entropy agrees with entropy when both exist."""
        self._skip_if_unsupported("cross_entropy")
        distribution = self._distribution()
        value = distribution.cross_entropy(distribution)
        assert np.isfinite(value)
        if "entropy" not in self.case.unsupported_methods:
            np.testing.assert_allclose(
                value, distribution.entropy(), rtol=1.0e-12, atol=1.0e-12
            )
