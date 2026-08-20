"""Define the normal-gamma prior for a Gaussian mean and precision.

The parameterization is ``tau ~ Gamma(a, scale=1 / b)`` and
``x | tau ~ Normal(mu, variance=1 / (lam * tau))``. Thus ``b`` is a
rate and the joint support is a real location paired with positive precision.
Instances are used as prior or posterior hyperparameters by Gaussian
likelihoods; the optional ``prior`` attribute is metadata only.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import scipy.integrate

from dmx.bstats.pdist import DistributionSampler, ProbabilityDistribution
from dmx.utils.special import digamma, gammaln

# This conjugate-prior distribution intentionally has no estimator or dedicated
# sequence encoder; inherited scalar fallbacks remain its public behavior.
# pylint: disable=abstract-method

NormalGammaDatum = tuple[float, float]
NormalGammaParameters = tuple[float, float, float, float]
Model = ProbabilityDistribution[Any, Any, Any]


class NormalGammaDistribution(
    ProbabilityDistribution[NormalGammaDatum, NormalGammaParameters, Any]
):
    """Joint normal-gamma distribution over a location and precision."""

    def __init__(
        self,
        mu: float,
        lam: float,
        a: float,
        b: float,
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize normal-gamma hyperparameters.

        Args:
            mu: Finite center for the conditional normal distribution.
            lam: Positive relative precision of the conditional normal.
            a: Positive gamma shape for precision.
            b: Positive gamma rate for precision.
            name: Optional model name.
            prior: Optional metadata describing an earlier prior.

        Raises:
            ValueError: If the parameters do not define a finite distribution.
        """
        super().__init__()
        self.set_parameters((mu, lam, a, b))
        self.parents = []
        self.name = name
        self.prior = cast(Model, prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"NormalGammaDistribution({self.mu!r}, {self.lam!r}, {self.a!r}, "
            f"{self.b!r}, name={self.name!r}, prior={self.prior})"
        )

    def get_parameters(self) -> NormalGammaParameters:
        """Return ``(mu, lam, a, b)`` in the documented parameterization."""
        return self.mu, self.lam, self.a, self.b

    def set_parameters(self, value: NormalGammaParameters) -> None:
        """Replace all normal-gamma hyperparameters.

        Args:
            value: Tuple ``(mu, lam, a, b)``.

        Raises:
            ValueError: If the parameters do not define a finite distribution.
        """
        mu, lam, a, b = value
        if not np.isfinite(mu):
            raise ValueError("Normal-gamma mu must be finite.")
        if any(not np.isfinite(value) or value <= 0 for value in (lam, a, b)):
            raise ValueError("Normal-gamma lam, a, and b must be finite and positive.")
        self.mu = float(mu)
        self.lam = float(lam)
        self.a = float(a)
        self.b = float(b)

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]``.

        The normal-gamma case is analytic. Other distributions are integrated
        numerically over real locations and positive precisions.

        Args:
            dist: Distribution whose joint log-density is averaged.

        Returns:
            Cross-entropy from this distribution to ``dist``.
        """
        if isinstance(dist, NormalGammaDistribution):
            c1 = (
                np.log(dist.b) * dist.a
                + 0.5 * np.log(dist.lam)
                - gammaln(dist.a)
                - 0.5 * np.log(2 * np.pi)
            )
            c2 = (dist.a - 0.5) * (digamma(self.a) - np.log(self.b))
            c2 -= dist.b * (self.a / self.b)
            squared_shift = (self.mu - dist.mu) ** 2
            c3 = -0.5 * dist.lam * ((1 / self.lam) + squared_shift * self.a / self.b)
            return float(-(c1 + c2 + c3))

        def integrand(precision: float, location: float) -> float:
            return -dist.log_density((location, precision)) * self.density(
                (location, precision)
            )

        value, _ = scipy.integrate.dblquad(
            integrand,
            -np.inf,
            np.inf,
            lambda _location: 0.0,
            lambda _location: np.inf,
        )
        return float(value)

    def entropy(self) -> float:
        """Return the differential entropy of the joint distribution."""
        value = (
            (self.a - 0.5) * (digamma(self.a) - np.log(self.b))
            - self.a
            - 0.5
            + np.log(self.b) * self.a
            + 0.5 * np.log(self.lam)
            - gammaln(self.a)
            - 0.5 * np.log(2 * np.pi)
        )
        return float(-value)

    def density(self, x: NormalGammaDatum) -> float:
        """Evaluate the joint density, returning zero for invalid precision."""
        if x[1] <= 0.0 or not np.isfinite(x[1]) or not np.isfinite(x[0]):
            return 0.0
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: NormalGammaDatum) -> float:
        """Evaluate the joint log-density.

        A nonfinite location or nonpositive/nonfinite precision is outside the
        support and returns ``-inf``.
        """
        location, precision = x
        if precision <= 0.0 or not np.isfinite(precision) or not np.isfinite(location):
            return float(-np.inf)
        c0 = (
            np.log(self.b) * self.a
            + 0.5 * np.log(self.lam / (2 * np.pi))
            - gammaln(self.a)
        )
        c1 = np.log(precision) * (self.a - 0.5) - self.b * precision
        c2 = -self.lam * precision * (location - self.mu) ** 2 / 2
        return float(c0 + c1 + c2)

    def sampler(self, seed: Optional[int] = None) -> "NormalGammaSampler":
        """Create a joint sampler using an optional deterministic seed."""
        return NormalGammaSampler(self, seed)


class NormalGammaSampler(DistributionSampler[NormalGammaDatum]):
    """Draw locations and precisions from a normal-gamma distribution."""

    def __init__(
        self, dist: NormalGammaDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize independent child random streams."""
        super().__init__(dist, seed)
        self.grng = np.random.RandomState(self.new_seed())
        self.nrng = np.random.RandomState(self.new_seed())

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one pair or a list of ``size`` location-precision pairs."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        precision = float(self.grng.gamma(self.dist.a, 1 / self.dist.b))
        scale = np.sqrt(1 / (self.dist.lam * precision))
        location = float(self.nrng.normal(self.dist.mu, scale))
        return location, precision
