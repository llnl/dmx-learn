"""Define the beta prior used by scalar Bernoulli likelihoods.

The distribution uses the conventional shape parameters ``a`` and ``b`` and
has support ``0 < x < 1``. In Bayesian likelihood modules an instance records
the current prior or posterior hyperparameters; its optional ``prior`` field is
metadata and is not folded into density, entropy, or cross-entropy calculations.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import scipy.integrate

from dmx.bstats.pdist import DistributionSampler, ProbabilityDistribution
from dmx.utils.special import betaln, digamma, gammaln

# These prior-only distributions intentionally do not implement estimators or
# dedicated encoders; the base class supplies scalar sequence fallbacks.
# pylint: disable=abstract-method

BetaParameters = tuple[float, float]
Model = ProbabilityDistribution[Any, Any, Any]


class BetaDistribution(ProbabilityDistribution[float, BetaParameters, Any]):
    """Beta distribution parameterized by positive shapes ``a`` and ``b``."""

    def __init__(
        self,
        a: float,
        b: float,
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize a beta distribution.

        Args:
            a: Positive first shape parameter.
            b: Positive second shape parameter.
            name: Optional model name.
            prior: Optional metadata describing an earlier prior.

        Raises:
            ValueError: If either shape is not finite and positive.
        """
        super().__init__()
        self.set_parameters((a, b))
        self.name = name
        self.prior = cast(Model, prior)
        self.parents = []

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"BetaDistribution({self.a!r}, {self.b!r}, name={self.name!r}, "
            f"prior={self.prior})"
        )

    def get_parameters(self) -> BetaParameters:
        """Return the ``(a, b)`` shape parameters."""
        return self.a, self.b

    def set_parameters(self, value: BetaParameters) -> None:
        """Replace both shape parameters and refresh the normalizer.

        Args:
            value: Positive ``(a, b)`` shape parameters.

        Raises:
            ValueError: If either shape is not finite and positive.
        """
        a, b = value
        if not np.isfinite(a) or a <= 0 or not np.isfinite(b) or b <= 0:
            raise ValueError("Beta shape parameters must be finite and positive.")
        self.a = float(a)
        self.b = float(b)
        self.norm_const = float(gammaln(a + b) - gammaln(a) - gammaln(b))

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]``.

        The beta-to-beta case is analytic. Other distributions are integrated
        numerically over this distribution's support.

        Args:
            dist: Distribution whose log-density is averaged.

        Returns:
            Cross-entropy from this distribution to ``dist``.
        """
        if isinstance(dist, BetaDistribution):
            return float(
                betaln(dist.a, dist.b)
                - (dist.a - 1) * digamma(self.a)
                - (dist.b - 1) * digamma(self.b)
                + (dist.a + dist.b - 2) * digamma(self.a + self.b)
            )
        value, _ = scipy.integrate.quad(
            lambda x: -dist.log_density(x) * self.density(x), 0.0, 1.0
        )
        return float(value)

    def entropy(self) -> float:
        """Return the differential entropy of the beta distribution."""
        return float(
            betaln(self.a, self.b)
            - (self.a - 1) * digamma(self.a)
            - (self.b - 1) * digamma(self.b)
            + (self.a + self.b - 2) * digamma(self.a + self.b)
        )

    def density(self, x: float) -> float:
        """Evaluate the density, returning zero outside ``0 < x < 1``."""
        if x <= 0.0 or x >= 1.0 or not np.isfinite(x):
            return 0.0
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density, returning ``-inf`` outside the support."""
        if x <= 0.0 or x >= 1.0 or not np.isfinite(x):
            return float(-np.inf)
        return float(
            (self.a - 1) * np.log(x) + (self.b - 1) * np.log1p(-x) + self.norm_const
        )

    def sampler(self, seed: Optional[int] = None) -> "BetaSampler":
        """Create a beta sampler using an optional deterministic seed."""
        return BetaSampler(self, seed)


class BetaSampler(DistributionSampler[float]):
    """Draw independent values from a :class:`BetaDistribution`."""

    def __init__(self, dist: BetaDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one float or an array of ``size`` beta values."""
        value = self.rng.beta(self.dist.a, self.dist.b, size=size)
        return float(value) if size is None else value
