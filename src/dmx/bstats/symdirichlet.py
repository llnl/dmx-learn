"""Symmetric Dirichlet priors for categorical probability vectors.

Density evaluation infers dimension from its observation. Sampling and
information measures require ``ndim`` because no observation is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, overload

import numpy as np

from dmx.bstats.pdist import DistributionSampler, SequenceEncodableDistribution
from dmx.utils.special import digamma, gammaln

# This prior-only distribution intentionally has no estimator or encoder.
# pylint: disable=abstract-method

Array = np.ndarray[Any, np.dtype[np.float64]]


class SymmetricDirichletDistribution(
    SequenceEncodableDistribution[Sequence[float], float, Sequence[Sequence[float]]]
):
    """Represent a symmetric prior or posterior over probability vectors.

    ``alpha`` is repeated across every simplex coordinate. The support is the
    interior of the simplex; ``ndim`` fixes its dimension when supplied and is
    otherwise inferred only for density evaluation.
    """

    def __init__(self, alpha: float, ndim: Optional[int] = None) -> None:
        """Initialize a symmetric concentration and optional dimension.

        Args:
            alpha: Positive concentration shared by all coordinates.
            ndim: Optional positive simplex dimension.

        Raises:
            ValueError: If ``alpha`` is non-finite or nonpositive, or ``ndim``
                is nonpositive.
        """
        super().__init__()
        self.ndim = ndim
        self.set_parameters(alpha)
        if ndim is not None and ndim <= 0:
            raise ValueError("A symmetric Dirichlet dimension must be positive.")

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        if self.ndim is None:
            return f"SymmetricDirichletDistribution({self.alpha!r})"
        return f"SymmetricDirichletDistribution({self.alpha!r}, ndim={self.ndim!r})"

    def get_parameters(self) -> float:
        """Return the shared scalar concentration."""
        return self.alpha

    def set_parameters(self, value: float) -> None:
        """Replace the shared concentration.

        Args:
            value: Positive finite concentration.

        Raises:
            ValueError: If ``value`` is non-finite or nonpositive.
        """
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Dirichlet concentrations must be finite and positive.")
        self.alpha = float(value)

    def dimension(self) -> int:
        """Return the dimension required by non-scoring operations.

        Raises:
            ValueError: If no explicit dimension was configured.
        """
        if self.ndim is None:
            raise ValueError("This operation requires an explicit dimension.")
        return self.ndim

    def log_density(self, x: Sequence[float]) -> float:
        """Evaluate the log-density, inferring dimension from ``x``.

        Args:
            x: Finite, strictly positive probability vector summing to one.

        Returns:
            Log-density, or ``-inf`` outside the configured simplex support.
        """
        observation = np.asarray(x, dtype=np.float64)
        if observation.ndim != 1 or observation.size == 0:
            return float(-np.inf)
        if self.ndim is not None and observation.size != self.ndim:
            return float(-np.inf)
        if (
            not np.all(np.isfinite(observation))
            or np.any(observation <= 0)
            or not np.isclose(observation.sum(), 1.0)
        ):
            return float(-np.inf)
        dimension = len(observation)
        log_const = dimension * gammaln(self.alpha) - gammaln(dimension * self.alpha)
        return float((self.alpha - 1.0) * np.log(observation).sum() - log_const)

    def entropy(self) -> float:
        """Return differential entropy for the explicit dimension.

        Raises:
            ValueError: If no explicit dimension was configured.
        """
        dimension = self.dimension()
        alpha_sum = dimension * self.alpha
        log_beta = dimension * gammaln(self.alpha) - gammaln(alpha_sum)
        return float(
            log_beta
            + (alpha_sum - dimension) * digamma(alpha_sum)
            - dimension * (self.alpha - 1.0) * digamma(self.alpha)
        )

    def cross_entropy(self, dist: Any) -> float:
        """Return ``-E_self[log(dist)]`` for a compatible symmetric prior.

        Args:
            dist: Distribution whose log-density is averaged.

        Returns:
            Analytic cross-entropy from this distribution to ``dist``.

        Raises:
            ValueError: If this distribution has no explicit dimension or the
                configured dimensions differ.
            NotImplementedError: If ``dist`` is not symmetric Dirichlet.
        """
        if not isinstance(dist, SymmetricDirichletDistribution):
            return super().cross_entropy(dist)
        dimension = self.dimension()
        if dist.ndim is not None and dist.ndim != dimension:
            raise ValueError("Dirichlet dimensions must agree.")
        expected_log = digamma(self.alpha) - digamma(dimension * self.alpha)
        return float(
            dimension * gammaln(dist.alpha)
            - gammaln(dimension * dist.alpha)
            - dimension * (dist.alpha - 1.0) * expected_log
        )

    def sampler(self, seed: Optional[int] = None) -> "SymmetricDirichletSampler":
        """Create a sampler, requiring an explicit dimension.

        Args:
            seed: Optional deterministic random seed.

        Returns:
            Sampler for this fixed-dimensional distribution.

        Raises:
            ValueError: If no explicit dimension was configured.
        """
        self.dimension()
        return SymmetricDirichletSampler(self, seed)


class SymmetricDirichletSampler(DistributionSampler[Sequence[float]]):
    """Draw vectors from a fixed-dimensional symmetric Dirichlet prior."""

    def __init__(
        self, dist: SymmetricDirichletDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler for ``dist``."""
        super().__init__(dist, seed)

    # The overload preserves scalar-versus-batch return shapes.
    # pylint: disable=signature-differs
    @overload
    def sample(self, size: None = None) -> Sequence[float]: ...

    @overload
    def sample(self, size: int) -> Array: ...

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one vector or an array shaped ``(size, dimension)``."""
        alpha = np.full(self.dist.dimension(), self.dist.alpha, dtype=np.float64)
        return np.asarray(self.rng.dirichlet(alpha, size=size), dtype=np.float64)
