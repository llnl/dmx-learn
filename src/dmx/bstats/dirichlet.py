"""Dirichlet priors for dense categorical probability vectors.

Vector parameters define an ordinary fixed-dimensional Dirichlet distribution.
A scalar parameter is a compatibility form for a symmetric prior whose dimension
is inferred from each scored observation. Scalar parameters therefore support
density evaluation but not sampling, entropy, or estimation until a dimension is
known.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Any, Optional, Union, overload

import numpy as np

from dmx.bstats.pdist import (
    DistributionSampler,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma, digammainv, gammaln

# This prior also retains the legacy estimator protocol used by bstats.
# pylint: disable=abstract-method

Array = np.ndarray[Any, np.dtype[np.float64]]
DenseObservation = Union[Sequence[float], Array]
DirichletParameters = Union[float, Array]
EncodedDirichlet = tuple[Array, Array, Array]
DirichletSufficientStatistics = tuple[float, Array, Array, Array]
Model = ProbabilityDistribution[Any, Any, Any]


def dirichlet_param_solve(
    alpha: Array, mean_log_p: Array, delta: float
) -> tuple[Array, int]:
    """Solve the Dirichlet fixed-point equation.

    Args:
        alpha: Initial concentration vector of shape ``(d,)``.
        mean_log_p: Mean log probabilities of shape ``(d,)``. Non-finite
            coordinates are excluded from the iteration.
        delta: Convergence threshold for relative absolute parameter change.

    Returns:
        Estimated concentrations of shape ``(d,)`` and iteration count.
        Excluded coordinates are returned as zero.
    """
    dimension = len(alpha)
    valid = np.isfinite(alpha) & (alpha > 0) & np.isfinite(mean_log_p)
    estimate = alpha[valid]
    log_means = mean_log_p[valid]
    count = 0
    change = (2 * delta) + 1
    while change > delta:
        count += 1
        old_estimate = estimate
        estimate = np.asarray(
            digammainv(log_means + digamma(estimate.sum())), dtype=np.float64
        )
        change = float(np.abs(estimate - old_estimate).sum() / estimate.sum())
    if dimension == estimate.size:
        return estimate, count
    result = np.zeros(dimension, dtype=np.float64)
    result[valid] = estimate
    return result, count


def mpe(
    initial: Array, update: Callable[[Array], Array], epsilon: float
) -> tuple[Array, int]:
    """Accelerate fixed-point iteration with polynomial extrapolation.

    Args:
        initial: Initial parameter vector of shape ``(d,)``.
        update: Callable performing one fixed-point update on that shape.
        epsilon: Absolute-residual convergence threshold.

    Returns:
        Extrapolated parameter vector of shape ``(d,)`` and update count.
    """
    first = update(initial)
    second = update(first)
    third = update(second)
    history = np.asarray([initial, first, second, third])
    result = third
    previous = third
    residual = float(np.abs(third - second).sum())
    count = 2
    while residual > epsilon:
        value = update(history[-1, :])
        difference = value - history[-1, :]
        increments = (history[1:, :] - history[:-1, :]).T
        coefficients = -np.dot(np.linalg.pinv(increments), difference)
        result = (np.dot(history[1:, :].T, coefficients) + value) / (
            coefficients.sum() + 1
        )
        residual = float(np.abs(result - previous).sum())
        previous = result
        history = np.concatenate((history, np.reshape(value, (1, -1))), axis=0)
        count += 1
    return np.asarray(result, dtype=np.float64), count


def alpha_seq_lambda(mean_log_p: Array) -> Callable[[Array], Array]:
    """Create the fixed-point update for Dirichlet concentrations.

    Args:
        mean_log_p: Mean log probabilities of shape ``(d,)``.

    Returns:
        Callable mapping a concentration vector of shape ``(d,)`` to its next
        fixed-point iterate.
    """

    def next_alpha(current_alpha: Array) -> Array:
        return np.asarray(
            digammainv(mean_log_p + digamma(current_alpha.sum())),
            dtype=np.float64,
        )

    return next_alpha


def find_alpha(
    current_alpha: Array, mean_log_p: Array, threshold: float
) -> tuple[Array, int]:
    """Estimate concentrations using accelerated fixed-point iteration.

    Args:
        current_alpha: Initial concentration vector of shape ``(d,)``.
        mean_log_p: Mean log probabilities of shape ``(d,)``.
        threshold: Absolute-residual convergence threshold.

    Returns:
        Estimated concentrations of shape ``(d,)`` and update count.
    """
    return mpe(current_alpha, alpha_seq_lambda(mean_log_p), threshold)


class DirichletDistribution(
    ProbabilityDistribution[DenseObservation, DirichletParameters, EncodedDirichlet]
):
    """Represent a prior or posterior over dense categorical probabilities.

    A vector ``alpha`` fixes dimension ``d`` and supplies one positive
    concentration per coordinate. A scalar supplies the same concentration to
    every coordinate and infers ``d`` from each scored observation. The support
    is the interior of the ``d``-dimensional probability simplex.
    """

    def __init__(self, alpha: Union[float, Sequence[float], Array]) -> None:
        """Initialize positive vector or scalar concentrations.

        Args:
            alpha: Positive scalar or nonempty vector of positive
                concentrations.

        Raises:
            ValueError: If concentrations are empty, non-finite, nonpositive,
                or not scalar or one-dimensional.
        """
        super().__init__()
        self.set_parameters(alpha)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        value: object = (
            self.alpha.tolist() if isinstance(self.alpha, np.ndarray) else self.alpha
        )
        return f"DirichletDistribution({value!r})"

    def get_parameters(self) -> DirichletParameters:
        """Return the scalar or dense concentration parameters."""
        return self.alpha

    def set_parameters(self, value: Union[float, Sequence[float], Array]) -> None:
        """Replace concentrations and refresh cached dimensional state.

        Args:
            value: Positive scalar or nonempty vector of positive
                concentrations.

        Raises:
            ValueError: If concentrations are empty, non-finite, nonpositive,
                or not scalar or one-dimensional.
        """
        if np.isscalar(value):
            concentration = float(value)  # type: ignore[arg-type]
            if not np.isfinite(concentration) or concentration <= 0:
                raise ValueError(
                    "Dirichlet concentrations must be finite and positive."
                )
            self.alpha: DirichletParameters = concentration
            self.dim = 0
            self.log_const: Optional[float] = None
            return
        concentrations = np.asarray(value, dtype=np.float64)
        if concentrations.ndim != 1 or concentrations.size == 0:
            raise ValueError("Dirichlet concentrations must be a nonempty vector.")
        if not np.all(np.isfinite(concentrations) & (concentrations > 0)):
            raise ValueError("Dirichlet concentrations must be finite and positive.")
        self.alpha = concentrations
        self.dim = len(concentrations)
        self.log_const = float(
            np.sum(gammaln(concentrations)) - gammaln(concentrations.sum())
        )

    def concentrations_for_dimension(self, dimension: int) -> Array:
        """Return dense concentrations for a requested dimension.

        Args:
            dimension: Required number of simplex coordinates.

        Returns:
            Concentration vector of shape ``(dimension,)``.

        Raises:
            ValueError: If fixed vector parameters have another dimension.
        """
        if isinstance(self.alpha, np.ndarray):
            if len(self.alpha) != dimension:
                raise ValueError("Observation dimension does not match parameters.")
            return self.alpha
        return np.full(dimension, self.alpha, dtype=np.float64)

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` for another Dirichlet distribution.

        Args:
            dist: Distribution whose log-density is averaged.

        Returns:
            Analytic cross-entropy from this distribution to ``dist``.

        Raises:
            ValueError: If both distributions are dimension-free or their
                fixed dimensions differ.
            NotImplementedError: If ``dist`` is not a Dirichlet distribution.
        """
        if not isinstance(dist, DirichletDistribution):
            return super().cross_entropy(dist)
        if self.dim == 0 and dist.dim == 0:
            raise ValueError("Cross-entropy requires a known Dirichlet dimension.")
        dimension = self.dim or dist.dim
        own = self.concentrations_for_dimension(dimension)
        other = dist.concentrations_for_dimension(dimension)
        expected_logs = digamma(own) - digamma(own.sum())
        return float(
            np.sum(gammaln(other))
            - gammaln(other.sum())
            - np.dot(other - 1.0, expected_logs)
        )

    def entropy(self) -> float:
        """Return differential entropy for fixed-dimensional parameters.

        Raises:
            ValueError: If the distribution has scalar, dimension-free
                parameters.
        """
        if self.dim == 0:
            raise ValueError("Entropy requires a known Dirichlet dimension.")
        return self.cross_entropy(self)

    def log_density(self, x: DenseObservation) -> float:
        """Evaluate the log-density at a dense simplex point.

        Args:
            x: Probability vector of shape ``(d,)`` with finite, strictly
                positive entries summing to one.

        Returns:
            Log-density, or ``-inf`` when ``x`` is outside the simplex.

        Raises:
            ValueError: If ``x`` conflicts with fixed vector parameters.
        """
        observation = np.asarray(x, dtype=np.float64)
        if observation.ndim != 1 or observation.size == 0:
            return float(-np.inf)
        if (
            not np.all(np.isfinite(observation))
            or np.any(observation <= 0)
            or not np.isclose(observation.sum(), 1.0)
        ):
            return float(-np.inf)
        alpha = self.concentrations_for_dimension(len(observation))
        log_const = float(np.sum(gammaln(alpha)) - gammaln(alpha.sum()))
        return float(np.dot(np.log(observation), alpha - 1.0) - log_const)

    def seq_log_density(self, x: EncodedDirichlet) -> Array:
        """Evaluate log-densities for encoded simplex observations.

        Args:
            x: Tuple of log values, values, and squared values, each shaped
                ``(n, d)`` as returned by :meth:`seq_encode`.

        Returns:
            Log-density array of shape ``(n,)``. Rows outside the simplex
            receive ``-inf``.

        Raises:
            ValueError: If encoded values are not a matrix or their dimension
                conflicts with fixed vector parameters.
        """
        logs, values, _ = x
        if values.ndim != 2:
            raise ValueError("Encoded Dirichlet observations must be a matrix.")
        alpha = self.concentrations_for_dimension(values.shape[1])
        log_const = float(np.sum(gammaln(alpha)) - gammaln(alpha.sum()))
        result = np.dot(logs, alpha - 1.0) - log_const
        valid = (
            np.all(np.isfinite(values), axis=1)
            & np.all(values > 0, axis=1)
            & np.isclose(values.sum(axis=1), 1.0)
        )
        return np.where(valid, result, -np.inf).astype(np.float64)

    def seq_encode(self, x: Iterable[DenseObservation]) -> EncodedDirichlet:
        """Encode observations as logs, values, and squared values.

        Args:
            x: Iterable of ``n`` dense observations with dimension ``d``.

        Returns:
            Tuple of log values, values, and squared values, each shaped
            ``(n, d)``. Logs are clipped at the smallest positive float.
        """
        values = np.asarray(x, dtype=np.float64)
        logs = np.log(np.maximum(values, sys.float_info.min))
        return logs, values, values * values

    def sampler(self, seed: Optional[int] = None) -> "DirichletSampler":
        """Create a sampler for fixed-dimensional dense parameters.

        Args:
            seed: Optional deterministic random seed.

        Returns:
            Sampler for this distribution.

        Raises:
            ValueError: If the distribution has scalar, dimension-free
                parameters.
        """
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "DirichletEstimator":
        """Create an estimator for fixed-dimensional dense parameters.

        Args:
            pseudo_count: Optional regularization weight for a log-statistic
                derived from the current concentrations.

        Returns:
            Estimator configured for this distribution's dimension.

        Raises:
            ValueError: If the distribution has scalar, dimension-free
                parameters.
        """
        if self.dim == 0:
            raise ValueError("Estimation requires a known Dirichlet dimension.")
        sufficient_statistic = None
        if pseudo_count is not None:
            alpha = self.concentrations_for_dimension(self.dim)
            sufficient_statistic = np.log(alpha / alpha.sum())
        return DirichletEstimator(
            dim=self.dim,
            pseudo_count=pseudo_count,
            suff_stat=sufficient_statistic,
        )


class DirichletSampler(DistributionSampler[DenseObservation]):
    """Draw dense simplex vectors from fixed-dimensional parameters."""

    def __init__(self, dist: DirichletDistribution, seed: Optional[int] = None) -> None:
        """Initialize a sampler for ``dist``."""
        if dist.dim == 0:
            raise ValueError("Sampling requires a known Dirichlet dimension.")
        super().__init__(dist, seed)

    # The overload preserves scalar-versus-batch return shapes.
    # pylint: disable=signature-differs
    @overload
    def sample(self, size: None = None) -> DenseObservation: ...

    @overload
    def sample(self, size: int) -> Array: ...

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one vector or an array shaped ``(size, dimension)``."""
        alpha = self.dist.concentrations_for_dimension(self.dist.dim)
        return np.asarray(self.rng.dirichlet(alpha=alpha, size=size), dtype=np.float64)


class DirichletAccumulator(
    SequenceEncodableAccumulator[
        DenseObservation, DirichletSufficientStatistics, EncodedDirichlet
    ]
):
    """Accumulate weighted statistics for Dirichlet estimation.

    The public statistic is ``(weight, sum_log_x, sum_x, sum_x_squared)``.
    Each vector has shape ``(d,)`` and contains coordinate-wise weighted sums.
    """

    def __init__(self, dim: int, keys: Optional[str] = None) -> None:
        """Initialize zero statistics for ``dim`` coordinates.

        Args:
            dim: Number of simplex coordinates.
            keys: Optional key used to share statistics between accumulators.
        """
        self.dim = dim
        self.sum_of_logs = np.zeros(dim, dtype=np.float64)
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum2 = np.zeros(dim, dtype=np.float64)
        self.counts = 0.0
        self.key = keys

    def update(
        self, x: DenseObservation, weight: float, estimate: Optional[Model]
    ) -> None:
        """Add one weighted simplex observation.

        Args:
            x: Dense probability vector of shape ``(d,)``.
            weight: Weight applied to the observation.
            estimate: Current estimate; unused for Dirichlet statistics.
        """
        del estimate
        value = np.asarray(x, dtype=np.float64)
        positive = value > 0
        self.sum_of_logs[positive] += np.log(value[positive]) * weight
        self.sum += weight * value
        self.sum2 += weight * value * value
        self.counts += weight

    def seq_update(
        self, x: EncodedDirichlet, weights: Array, estimate: Optional[Model]
    ) -> None:
        """Add encoded observations with corresponding weights.

        Args:
            x: Encoded tuple whose arrays have shape ``(n, d)``.
            weights: Observation weights of shape ``(n,)``.
            estimate: Current estimate; unused for Dirichlet statistics.
        """
        del estimate
        self.sum_of_logs += np.dot(weights, x[0])
        self.counts += float(weights.sum())
        self.sum += np.dot(weights, x[1])
        self.sum2 += np.dot(weights, x[2])

    def combine(
        self, suff_stat: DirichletSufficientStatistics
    ) -> "DirichletAccumulator":
        """Merge another sufficient-statistic tuple.

        Args:
            suff_stat: ``(weight, sum_log_x, sum_x, sum_x_squared)`` to add.

        Returns:
            This mutated accumulator.
        """
        count, sum_of_logs, values, squares = suff_stat
        self.counts += count
        self.sum_of_logs += sum_of_logs
        self.sum += values
        self.sum2 += squares
        return self

    def value(self) -> DirichletSufficientStatistics:
        """Return ``(weight, sum_log_x, sum_x, sum_x_squared)`` statistics."""
        return self.counts, self.sum_of_logs, self.sum, self.sum2

    def from_value(self, x: DirichletSufficientStatistics) -> "DirichletAccumulator":
        """Restore a sufficient-statistic tuple."""
        self.counts, self.sum_of_logs, self.sum, self.sum2 = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics under the optional shared key."""
        if self.key is None:
            return
        if self.key in stats_dict:
            stats_dict[self.key].combine(self.value())
        else:
            stats_dict[self.key] = self

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Restore statistics from the optional shared key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())


class DirichletAccumulatorFactory(
    StatisticAccumulatorFactory[
        DenseObservation, DirichletSufficientStatistics, EncodedDirichlet
    ]
):
    """Create Dirichlet sufficient-statistic accumulators."""

    def __init__(self, dim: int, keys: Optional[str] = None) -> None:
        """Store the accumulator dimension and optional shared key."""
        self.dim = dim
        self.keys = keys

    def make(self) -> DirichletAccumulator:
        """Create a zeroed accumulator."""
        return DirichletAccumulator(self.dim, self.keys)


class DirichletEstimator(
    ParameterEstimator[
        DenseObservation,
        DirichletParameters,
        EncodedDirichlet,
        DirichletSufficientStatistics,
    ]
):
    """Estimate dense Dirichlet concentrations from sufficient statistics.

    With ``pseudo_count``, the estimator adds weighted prior log-statistics
    before solving the concentration fixed-point equation. Without it, the
    empirical mean initializes the solver.
    """

    def __init__(
        self,
        dim: int,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Array] = None,
        delta: float = 1.0e-8,
        keys: Optional[str] = None,
        use_mpe: bool = False,
    ) -> None:
        """Configure dimension, regularization, and solver tolerance.

        Args:
            dim: Number of simplex coordinates.
            pseudo_count: Optional weight assigned to prior log-statistics.
            suff_stat: Optional prior mean-log vector of shape ``(dim,)``. A
                symmetric unit-concentration value is used when omitted.
            delta: Solver convergence threshold.
            keys: Optional key used to share accumulator statistics.
            use_mpe: Use polynomial extrapolation instead of direct fixed-point
                iteration.
        """
        self.dim = dim
        self.pseudo_count = pseudo_count
        self.delta = delta
        self.suff_stat = suff_stat
        self.keys = keys
        self.use_mpe = use_mpe

    def accumulator_factory(self) -> DirichletAccumulatorFactory:
        """Create an accumulator factory for this estimator."""
        return DirichletAccumulatorFactory(self.dim, self.keys)

    # The base estimator supports these legacy call forms via ``*args``.
    # pylint: disable=arguments-differ
    @overload
    def estimate(
        self, suff_stat: DirichletSufficientStatistics, /
    ) -> DirichletDistribution: ...

    @overload
    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: DirichletSufficientStatistics,
        /,
    ) -> DirichletDistribution: ...

    def estimate(self, *args: Any) -> DirichletDistribution:
        """Estimate concentrations using either legacy estimator call form.

        Args:
            *args: Either one sufficient-statistic tuple, or ``nobs`` followed
                by that tuple. ``nobs`` is accepted for protocol compatibility;
                the tuple's accumulated weight is authoritative.

        Returns:
            Fixed-dimensional Dirichlet distribution with fitted
            concentrations.

        Raises:
            TypeError: If called with any other number of arguments.
            ValueError: If accumulated observation weight is not positive.
        """
        if len(args) == 1:
            suff_stat = args[0]
        elif len(args) == 2:
            suff_stat = args[1]
        else:
            raise TypeError("estimate expects statistics, with optional nobs")
        count, sum_of_logs, sum_values, _ = suff_stat
        dimension = len(sum_of_logs)
        if count <= 0:
            raise ValueError("Estimation requires positive observation weight.")
        if self.pseudo_count is not None:
            prior_stat = self.suff_stat
            if prior_stat is None:
                prior_stat = np.full(
                    dimension, digamma(1.0) - digamma(dimension), dtype=np.float64
                )
            combined_logs = sum_of_logs + prior_stat * self.pseudo_count
            initial = combined_logs * (dimension / combined_logs.sum())
            mean_log_p = combined_logs / (count + self.pseudo_count)
        else:
            initial = sum_values / count
            initial[-1] = 1.0 - initial[:-1].sum()
            mean_log_p = sum_of_logs / count
        if count == 1.0:
            return DirichletDistribution(initial)
        solver = find_alpha if self.use_mpe else dirichlet_param_solve
        alpha, _ = solver(np.asarray(initial), mean_log_p, self.delta)
        return DirichletDistribution(alpha)
