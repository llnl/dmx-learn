"""Define independent normal-gamma priors for diagonal Gaussian parameters.

For every coordinate, ``tau_i ~ Gamma(a_i, scale=1 / b_i)`` and
``x_i | tau_i ~ Normal(mu_i, variance=1 / (lam_i * tau_i))``. The joint
support contains finite location vectors and strictly positive precision
vectors. Diagonal Gaussian likelihoods use these arrays as current prior or
posterior hyperparameters; the optional ``prior`` attribute is metadata only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.pdist import DistributionSampler, ProbabilityDistribution
from dmx.utils.special import digamma, gammaln

# This conjugate-prior distribution intentionally has no estimator or dedicated
# sequence encoder; inherited scalar fallbacks remain its public behavior.
# pylint: disable=abstract-method

ArrayLike = Sequence[float] | np.ndarray[Any, np.dtype[Any]]
MultivariateNormalGammaDatum = tuple[ArrayLike, ArrayLike]
MultivariateNormalGammaParameters = tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]
Model = ProbabilityDistribution[Any, Any, Any]


class MultivariateNormalGammaDistribution(
    ProbabilityDistribution[
        MultivariateNormalGammaDatum, MultivariateNormalGammaParameters, Any
    ]
):
    """Represent independent normal-gamma factors for diagonal parameters.

    Each hyperparameter is a vector of shape ``(d,)``. Observations are pairs
    ``(location, precision)`` of that shape, with finite locations and strictly
    positive finite precisions. Instances represent current prior or posterior
    hyperparameters for a diagonal Gaussian likelihood.
    """

    def __init__(
        self,
        mu: np.ndarray[Any, Any],
        lam: np.ndarray[Any, Any],
        a: np.ndarray[Any, Any],
        b: np.ndarray[Any, Any],
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize vector normal-gamma hyperparameters.

        Args:
            mu: Finite coordinate centers of shape ``(d,)``.
            lam: Positive coordinate-wise relative precisions of shape ``(d,)``.
            a: Positive gamma shapes for precision of shape ``(d,)``.
            b: Positive gamma rates for precision of shape ``(d,)``.
            name: Optional model name.
            prior: Optional metadata describing an earlier prior.

        Raises:
            ValueError: If arrays differ in shape or contain invalid parameters.
        """
        super().__init__()
        self.name = name
        self.prior = cast(Model, prior)
        self.parents = []
        self.set_parameters((mu, lam, a, b))

    def __str__(self) -> str:
        """Return a constructor-like representation using plain lists."""
        return (
            f"MultivariateNormalGammaDistribution({self.mu.tolist()!r}, "
            f"{self.lam.tolist()!r}, {self.a.tolist()!r}, {self.b.tolist()!r}, "
            f"name={self.name!r}, prior={self.prior})"
        )

    def get_parameters(self) -> MultivariateNormalGammaParameters:
        """Return ``(mu, lam, a, b)`` parameter arrays."""
        return self.mu, self.lam, self.a, self.b

    def set_parameters(self, value: MultivariateNormalGammaParameters) -> None:
        """Replace all vector normal-gamma hyperparameters.

        Args:
            value: Tuple of ``(mu, lam, a, b)`` arrays with matching shapes.

        Raises:
            ValueError: If arrays differ in shape or contain invalid parameters.
        """
        mu, lam, a, b = (np.asarray(item, dtype=float) for item in value)
        if mu.ndim != 1 or not mu.shape == lam.shape == a.shape == b.shape:
            raise ValueError(
                "Multivariate normal-gamma parameters must be matching vectors."
            )
        if not np.all(np.isfinite(mu)):
            raise ValueError("Multivariate normal-gamma mu must be finite.")
        if any(
            not np.all(np.isfinite(item)) or np.any(item <= 0) for item in (lam, a, b)
        ):
            raise ValueError(
                "Multivariate normal-gamma lam, a, and b must be finite "
                "and positive."
            )
        self.mu = mu
        self.lam = lam
        self.a = a
        self.b = b

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` for the same distribution family.

        There is no generic numerical fallback for arbitrary multivariate
        distributions because their dimension and integration contract are not
        available through the base interface.

        Args:
            dist: Multivariate normal-gamma distribution to compare.

        Returns:
            Sum of coordinate-wise analytic cross-entropies.

        Raises:
            NotImplementedError: If ``dist`` is from another family.
        """
        if not isinstance(dist, MultivariateNormalGammaDistribution):
            raise NotImplementedError(
                "Cross-entropy is only defined between multivariate "
                "normal-gamma distributions."
            )
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
        return float(-np.sum(c1 + c2 + c3))

    def entropy(self) -> float:
        """Return the summed differential entropy of all coordinates."""
        value = (
            (self.a - 0.5) * (digamma(self.a) - np.log(self.b))
            - self.a
            - 0.5
            + np.log(self.b) * self.a
            + 0.5 * np.log(self.lam)
            - gammaln(self.a)
            - 0.5 * np.log(2 * np.pi)
        )
        return float(-np.sum(value))

    def density(self, x: MultivariateNormalGammaDatum) -> float:
        """Evaluate the joint density, returning zero outside the support."""
        if not self._in_support(x):
            return 0.0
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: MultivariateNormalGammaDatum) -> float:
        """Evaluate the joint log-density, returning ``-inf`` off support."""
        if not self._in_support(x):
            return float(-np.inf)
        location = np.asarray(x[0], dtype=float)
        precision = np.asarray(x[1], dtype=float)
        c0 = (
            np.log(self.b) * self.a
            + 0.5 * np.log(self.lam / (2 * np.pi))
            - gammaln(self.a)
        )
        c1 = np.log(precision) * (self.a - 0.5) - self.b * precision
        c2 = -self.lam * precision * (location - self.mu) ** 2 / 2
        return float(np.sum(c0 + c1 + c2))

    def _in_support(self, x: MultivariateNormalGammaDatum) -> bool:
        """Return whether one location-precision pair belongs to the support."""
        location = np.asarray(x[0], dtype=float)
        precision = np.asarray(x[1], dtype=float)
        return bool(
            location.shape == self.mu.shape
            and precision.shape == self.mu.shape
            and np.all(np.isfinite(location))
            and np.all(np.isfinite(precision))
            and np.all(precision > 0)
        )

    def sampler(self, seed: Optional[int] = None) -> "MultivariateNormalGammaSampler":
        """Create a joint sampler using an optional deterministic seed."""
        return MultivariateNormalGammaSampler(self, seed)


class MultivariateNormalGammaSampler(DistributionSampler[MultivariateNormalGammaDatum]):
    """Draw vector locations and precisions from normal-gamma factors."""

    def __init__(
        self,
        dist: MultivariateNormalGammaDistribution,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize independent child random streams."""
        super().__init__(dist, seed)
        self.grng = np.random.RandomState(self.new_seed())
        self.nrng = np.random.RandomState(self.new_seed())

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one pair or a list of ``size`` vector pairs."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        precision = self.grng.gamma(self.dist.a, 1 / self.dist.b)
        scale = np.sqrt(1 / (self.dist.lam * precision))
        location = self.nrng.normal(self.dist.mu, scale)
        return location, precision
