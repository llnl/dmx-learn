"""Tests for dense, symmetric, and dictionary categorical priors."""

from __future__ import annotations

import numpy as np
import pytest

from dmx.bstats import DictDirichletDistribution, DirichletDistribution
from dmx.bstats.categorical import CategoricalDistribution
from dmx.bstats.intrange import IntegerCategoricalDistribution
from dmx.bstats.symdirichlet import SymmetricDirichletDistribution


def test_dense_and_scalar_dirichlet_log_densities_are_finite() -> None:
    """Check fixed dense and dimension-free symmetric parameter paths."""
    observation = np.array([0.2, 0.3, 0.5])
    dense = DirichletDistribution([2.0, 3.0, 4.0])
    scalar = DirichletDistribution(2.0)

    assert np.isfinite(dense.log_density(observation))
    assert np.isfinite(scalar.log_density(observation))
    assert np.isfinite(dense.expected_log_density(observation))

    encoded = dense.seq_encode([observation, observation])
    expected = [dense.log_density(observation), dense.log_density(observation)]
    assert np.allclose(dense.seq_log_density(encoded), expected)


def test_dirichlet_sampling_shapes_and_parameter_access() -> None:
    """Check scalar and batch sampling plus cached parameter replacement."""
    distribution = DirichletDistribution([2.0, 3.0, 4.0])
    assert np.asarray(distribution.sampler(seed=4).sample()).shape == (3,)
    assert distribution.sampler(seed=4).sample(size=5).shape == (5, 3)

    distribution.set_parameters([4.0, 3.0, 2.0])
    np.testing.assert_equal(distribution.get_parameters(), [4.0, 3.0, 2.0])
    assert np.isfinite(distribution.log_density([0.5, 0.3, 0.2]))

    scalar = DirichletDistribution(1.5)
    assert scalar.get_parameters() == 1.5
    with pytest.raises(ValueError, match="known Dirichlet dimension"):
        scalar.sampler(seed=4)


def test_dense_dirichlet_entropy_and_cross_entropy() -> None:
    """Check finite information measures and their self-consistency."""
    distribution = DirichletDistribution([2.0, 3.0, 4.0])
    assert np.isfinite(distribution.entropy())
    np.testing.assert_allclose(
        distribution.cross_entropy(distribution), distribution.entropy()
    )
    assert np.isfinite(distribution.cross_entropy(DirichletDistribution(1.5)))


def test_symmetric_dirichlet_density_sampling_and_information() -> None:
    """Check dimension inference for scoring and explicit sampling dimension."""
    inferred = SymmetricDirichletDistribution(2.0)
    assert np.isfinite(inferred.log_density([0.2, 0.3, 0.5]))

    distribution = SymmetricDirichletDistribution(2.0, ndim=3)
    assert distribution.get_parameters() == 2.0
    assert np.asarray(distribution.sampler(seed=5).sample()).shape == (3,)
    assert distribution.sampler(seed=5).sample(size=4).shape == (4, 3)
    assert np.isfinite(distribution.entropy())
    np.testing.assert_allclose(
        distribution.cross_entropy(distribution), distribution.entropy()
    )


def test_dictionary_dirichlet_fixed_and_scalar_paths() -> None:
    """Check explicit dictionary support and scalar default support inference."""
    observation = {"a": 0.2, "b": 0.3, "c": 0.5}
    fixed = DictDirichletDistribution({"a": 2.0, "b": 3.0, "c": 4.0})
    scalar = DictDirichletDistribution(2.0)

    assert np.isfinite(fixed.log_density(observation))
    assert np.isfinite(scalar.log_density(observation))
    assert np.isfinite(fixed.expected_log_density(observation))
    assert scalar.get_parameters() == 2.0
    assert fixed.get_parameters() == {"a": 2.0, "b": 3.0, "c": 4.0}

    sample = fixed.sampler(seed=6).sample()
    samples = fixed.sampler(seed=6).sample(size=4)
    assert set(sample) == set(observation)
    assert len(samples) == 4
    assert all(set(item) == set(observation) for item in samples)
    np.testing.assert_allclose(sum(sample.values()), 1.0)

    assert np.isfinite(fixed.entropy())
    np.testing.assert_allclose(fixed.cross_entropy(fixed), fixed.entropy())
    assert np.isfinite(fixed.cross_entropy(scalar))


def test_dimension_free_dictionary_sampling_requires_keys() -> None:
    """Check that the scalar compatibility form reports its missing support."""
    with pytest.raises(ValueError, match="known dictionary keys"):
        DictDirichletDistribution(2.0).sampler(seed=1)


def test_existing_categorical_default_prior_construction_is_compatible() -> None:
    """Check categorical modules retain their scalar default prior families."""
    categorical = CategoricalDistribution({"a": 0.4, "b": 0.6})
    integer = IntegerCategoricalDistribution(np.array([0.4, 0.6]))

    assert isinstance(categorical.get_prior(), DictDirichletDistribution)
    assert isinstance(integer.get_prior(), DirichletDistribution)
    assert isinstance(categorical.get_prior().get_parameters(), float)
    assert isinstance(integer.get_prior().get_parameters(), float)
    assert np.isfinite(categorical.expected_log_density("a"))
