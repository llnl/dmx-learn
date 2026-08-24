"""Dictionary-keyed Dirichlet priors for categorical probability mappings.

A mapping assigns a concentration to each fixed key and supports dictionary-
shaped sampling. A scalar is the compatibility form used by categorical
defaults and applies to whatever keys an observation contains. With no stored
keys, that form cannot be sampled or have an entropy computed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union, cast, overload

import numpy as np

from dmx.bstats.pdist import DistributionSampler, ProbabilityDistribution
from dmx.utils.special import digamma, gammaln

# This prior-only distribution intentionally has no estimator or encoder.
# pylint: disable=abstract-method

Key = Any
ConcentrationMap = dict[Key, float]
DictDirichletParameters = Union[ConcentrationMap, float]
Observation = Mapping[Key, float]


class DictDirichletDistribution(
    ProbabilityDistribution[Observation, DictDirichletParameters, Any]
):
    """Represent a prior or posterior over keyed categorical probabilities.

    Mapping parameters fix the category set and assign one positive
    concentration per key. A scalar parameter applies the same concentration
    to the keys of each scored observation, but does not provide enough support
    information for sampling or entropy.
    """

    def __init__(self, alpha: Union[Mapping[Key, float], float]) -> None:
        """Initialize mapping concentrations or a dimension-free scalar.

        Args:
            alpha: Nonempty mapping of category keys to concentrations, or one
                positive concentration shared by observation-defined keys.

        Raises:
            ValueError: If concentrations are empty, non-finite, or nonpositive.
        """
        super().__init__()
        self.set_parameters(alpha)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return f"DictDirichletDistribution({self.alpha!r})"

    def get_parameters(self) -> DictDirichletParameters:
        """Return mapping concentrations or the scalar compatibility value."""
        return self.alpha

    def set_parameters(self, value: Union[Mapping[Key, float], float]) -> None:
        """Replace fixed-key or dimension-free concentrations.

        Args:
            value: Nonempty concentration mapping or positive shared scalar.

        Raises:
            ValueError: If concentrations are empty, non-finite, or nonpositive.
        """
        if isinstance(value, Mapping):
            concentrations = {key: float(item) for key, item in value.items()}
            if not concentrations:
                raise ValueError("Dictionary Dirichlet parameters cannot be empty.")
            if not all(
                np.isfinite(item) and item > 0 for item in concentrations.values()
            ):
                raise ValueError(
                    "Dirichlet concentrations must be finite and positive."
                )
            self.alpha: DictDirichletParameters = concentrations
            self.is_unbounded = False
            return
        concentration = float(value)
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError("Dirichlet concentrations must be finite and positive.")
        self.alpha = concentration
        self.is_unbounded = True

    def concentrations_for_keys(
        self, keys: list[Key]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return dense concentrations in the requested key order.

        Args:
            keys: Ordered category keys for the returned vector.

        Returns:
            Concentration array of shape ``(len(keys),)``.

        Raises:
            ValueError: If ``keys`` differ from fixed mapping parameters.
        """
        if isinstance(self.alpha, float):
            return np.full(len(keys), self.alpha, dtype=np.float64)
        if set(keys) != set(self.alpha):
            raise ValueError("Observation keys must match prior keys.")
        return np.asarray([self.alpha[key] for key in keys], dtype=np.float64)

    def log_density(self, x: Observation) -> float:
        """Evaluate the log-density of a keyed simplex observation.

        Args:
            x: Mapping of category keys to finite, strictly positive
                probabilities summing to one.

        Returns:
            Log-density, or ``-inf`` when the probabilities are outside the
            simplex.

        Raises:
            ValueError: If observation keys differ from fixed mapping
                parameters.
        """
        if not x:
            return float(-np.inf)
        keys = list(x)
        values = np.asarray([x[key] for key in keys], dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values <= 0)
            or not np.isclose(values.sum(), 1.0)
        ):
            return float(-np.inf)
        alpha = self.concentrations_for_keys(keys)
        log_const = float(np.sum(gammaln(alpha)) - gammaln(alpha.sum()))
        return float(np.dot(np.log(values), alpha - 1.0) - log_const)

    def cross_entropy(self, dist: Any) -> float:
        """Return ``-E_self[log(dist)]`` for a compatible keyed distribution.

        Args:
            dist: Distribution whose log-density is averaged.

        Returns:
            Analytic cross-entropy from this distribution to ``dist``.

        Raises:
            ValueError: If neither distribution defines keys or their fixed
                key sets differ.
            NotImplementedError: If ``dist`` is not dictionary Dirichlet.
        """
        if not isinstance(dist, DictDirichletDistribution):
            return super().cross_entropy(dist)
        if isinstance(self.alpha, float) and isinstance(dist.alpha, float):
            raise ValueError("Cross-entropy requires known dictionary keys.")
        source = self.alpha if isinstance(self.alpha, dict) else dist.alpha
        keys = list(cast(ConcentrationMap, source))
        own = self.concentrations_for_keys(keys)
        other = dist.concentrations_for_keys(keys)
        expected_logs = digamma(own) - digamma(own.sum())
        return float(
            np.sum(gammaln(other))
            - gammaln(other.sum())
            - np.dot(other - 1.0, expected_logs)
        )

    def entropy(self) -> float:
        """Return differential entropy for fixed dictionary parameters.

        Raises:
            ValueError: If scalar parameters do not define category keys.
        """
        if isinstance(self.alpha, float):
            raise ValueError("Entropy requires known dictionary keys.")
        return self.cross_entropy(self)

    def sampler(self, seed: Optional[int] = None) -> "DictDirichletSampler":
        """Create a keyed sampler for fixed dictionary parameters.

        Args:
            seed: Optional deterministic random seed.

        Returns:
            Sampler preserving the parameter mapping's key order.

        Raises:
            ValueError: If scalar parameters do not define category keys.
        """
        if isinstance(self.alpha, float):
            raise ValueError("Sampling requires known dictionary keys.")
        return DictDirichletSampler(self, seed)


class DictDirichletSampler(DistributionSampler[Observation]):
    """Draw keyed simplex mappings from a dictionary Dirichlet prior."""

    def __init__(
        self, dist: DictDirichletDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler and preserve parameter key order."""
        if isinstance(dist.alpha, float):
            raise ValueError("Sampling requires known dictionary keys.")
        super().__init__(dist, seed)
        self.keys = list(dist.alpha)
        self.concentrations = np.asarray(
            [dist.alpha[key] for key in self.keys], dtype=np.float64
        )

    # The overload preserves scalar-versus-batch return shapes.
    # pylint: disable=signature-differs
    @overload
    def sample(self, size: None = None) -> Observation: ...

    @overload
    def sample(self, size: int) -> list[dict[Key, float]]: ...

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one mapping or a list of ``size`` mappings."""
        values = self.rng.dirichlet(self.concentrations, size=size)
        if size is None:
            return dict(zip(self.keys, np.asarray(values, dtype=float)))
        return [dict(zip(self.keys, row)) for row in values]
