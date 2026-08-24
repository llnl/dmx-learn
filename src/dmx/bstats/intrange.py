"""Bayesian categorical likelihoods on contiguous integer ranges.

The probability vector covers integers from ``min_index`` through
``min_index + len(prob_vec) - 1``. Outside observations receive the legacy
escape mass ``default_value / (1 + default_value)`` (zero by default).
The default dimension-free Dirichlet prior has concentration ``1 + 1e-12``
per supported integer. Accumulators store ``(minimum, count_vector)`` and grow
the represented range as observations arrive. Expected log-density uses
Dirichlet expectations on support and the fixed escape mass outside it.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from dmx.bstats.dirichlet import DirichletDistribution
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.bstats.symdirichlet import SymmetricDirichletDistribution
from dmx.utils.special import digamma

# Legacy bstats implementations are intentionally concrete protocol classes.
# pylint: disable=abstract-method

Array = np.ndarray[Any, np.dtype[np.float64]]
IntegerEncoded = np.ndarray[Any, np.dtype[np.int_]]
IntegerSuffStat = tuple[int, Array]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = DirichletDistribution(1.0 + 1.0e-12)


class IntegerCategoricalDistribution(
    ProbabilityDistribution[int, Array, IntegerEncoded]
):
    """Represent a categorical likelihood on a contiguous integer range.

    A probability vector of shape ``(k,)`` corresponds in order to integers
    ``min_index`` through ``min_index + k - 1``. A dense or symmetric Dirichlet
    prior enables posterior-expected scoring; another prior uses the fixed
    probabilities. Values outside the range receive the configured escape mass.
    """

    def __init__(
        self,
        prob_vec: Union[np.ndarray, list[float], sp.spmatrix],
        default_value: float = 0.0,
        min_index: int = 0,
        name: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize probabilities, support origin, escape mass, and prior."""
        super().__init__()
        self.min_index = min_index
        self.name = name
        self.default_value = float(default_value)
        if not np.isfinite(self.default_value) or self.default_value < 0.0:
            raise ValueError("Integer categorical default must be nonnegative.")
        self.set_parameters(prob_vec)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"IntegerCategoricalDistribution({self.prob_vec.tolist()!r}, "
            f"default_value={self.default_value!r}, min_index={self.min_index!r}, "
            f"name={self.name!r}, prior={self.prior})"
        )

    def get_parameters(self) -> Array:
        """Return the probability vector."""
        return self.prob_vec

    def set_parameters(
        self, value: Union[np.ndarray, list[float], sp.spmatrix]
    ) -> None:
        """Replace probabilities and refresh cached support state."""
        raw_params: Any = value
        values: Array
        if sp.issparse(raw_params):
            values = np.asarray(raw_params.toarray(), dtype=np.float64).reshape(-1)
        else:
            values = np.asarray(raw_params, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Integer categorical probabilities must be a vector.")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Integer categorical probabilities must be nonnegative.")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("Integer categorical probabilities must sum to one.")
        self.prob_vec = values.astype(np.float64)
        self.num_vals = len(values)
        self.max_index = self.min_index + self.num_vals - 1
        with np.errstate(divide="ignore"):
            self.log_prob_vec = np.log(self.prob_vec)
            self.log_default_value = float(np.log(self.default_value))
        self.log_const = float(np.log1p(self.default_value))

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache supported expected log probabilities."""
        self.prior = prior
        concentrations: Optional[Array]
        if isinstance(prior, DirichletDistribution):
            concentrations = prior.concentrations_for_dimension(self.num_vals)
        elif isinstance(prior, SymmetricDirichletDistribution):
            concentrations = np.full(
                self.num_vals, prior.get_parameters(), dtype=np.float64
            )
        else:
            concentrations = None
        self.conj_prior_params = concentrations
        self.expected_nparams = (
            None
            if concentrations is None
            else digamma(concentrations) - digamma(concentrations.sum())
        )

    def get_prior(self) -> Model:
        """Return the current Dirichlet or alternate prior."""
        return self.prior

    def get_data_type(self) -> type[int]:
        """Return the intended observation type."""
        return int

    def entropy(self) -> float:
        """Return Shannon entropy over the explicit sampled support."""
        positive = self.prob_vec > 0.0
        return float(-np.dot(self.prob_vec[positive], self.log_prob_vec[positive]))

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` over supported integers."""
        return float(
            -sum(
                probability * dist.log_density(self.min_index + index)
                for index, probability in enumerate(self.prob_vec)
            )
        )

    def moment(self, order: int, origin: float = 0.0) -> float:
        """Return a raw moment about ``origin``."""
        values = np.arange(self.min_index, self.max_index + 1) - origin
        return float(np.dot(np.power(values, order), self.prob_vec))

    def log_density(self, x: int) -> float:
        """Evaluate supported or escape log mass."""
        if x < self.min_index or x > self.max_index:
            return self.log_default_value - self.log_const
        return float(self.log_prob_vec[x - self.min_index] - self.log_const)

    def expected_log_density(self, x: int) -> float:
        """Evaluate prior-averaged log mass when conjugate."""
        if x < self.min_index or x > self.max_index:
            return self.log_default_value - self.log_const
        if self.expected_nparams is None:
            return self.log_density(x)
        return float(self.expected_nparams[x - self.min_index] - self.log_const)

    def seq_log_density(self, x: IntegerEncoded) -> np.ndarray[Any, Any]:
        """Evaluate log masses for encoded integer observations."""
        indices = x - self.min_index
        valid = (indices >= 0) & (indices < self.num_vals)
        result = np.full(len(x), self.log_default_value - self.log_const)
        result[valid] = self.log_prob_vec[indices[valid]] - self.log_const
        return result

    def seq_expected_log_density(self, x: IntegerEncoded) -> np.ndarray[Any, Any]:
        """Evaluate expected log masses for encoded integers."""
        if self.expected_nparams is None:
            return self.seq_log_density(x)
        indices = x - self.min_index
        valid = (indices >= 0) & (indices < self.num_vals)
        result = np.full(len(x), self.log_default_value - self.log_const)
        result[valid] = self.expected_nparams[indices[valid]] - self.log_const
        return result

    def seq_encode(self, x: Iterable[int]) -> IntegerEncoded:
        """Encode ``n`` observations as an integer array of shape ``(n,)``."""
        return np.asarray(tuple(x), dtype=int)

    def sampler(self, seed: Optional[int] = None) -> "IntegerCategoricalSampler":
        """Create a repeatable integer categorical sampler."""
        return IntegerCategoricalSampler(self, seed)

    def estimator(self) -> "IntegerCategoricalEstimator":
        """Create an estimator retaining metadata and the current prior."""
        return IntegerCategoricalEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Create the integer categorical sequence encoder."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalSampler(DistributionSampler[int]):
    """Draw independent integers from a contiguous categorical support."""

    def __init__(
        self, dist: IntegerCategoricalDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one integer or a list of ``size`` integers."""
        values = self.rng.choice(
            np.arange(self.dist.min_index, self.dist.max_index + 1),
            p=self.dist.prob_vec,
            size=size,
        )
        if size is None:
            return int(values)
        return np.asarray(values, dtype=int).tolist()


class IntegerCategoricalAccumulator(
    SequenceEncodableAccumulator[int, IntegerSuffStat, IntegerEncoded]
):
    """Accumulate weighted counts over a contiguous integer range.

    The sufficient statistic is ``(minimum, count_vector)``, where entry ``i``
    stores the weight for integer ``minimum + i``. Unless both constructor
    bounds are supplied, the range grows to include observed values.
    """

    def __init__(
        self,
        min_val: Optional[int],
        max_val: Optional[int],
        keys: tuple[Optional[str], ...] = (None,),
    ) -> None:
        """Initialize an optional fixed range and zero counts."""
        self.minVal = min_val
        self.maxVal = max_val
        self.countVec: Optional[Array]
        if min_val is not None and max_val is not None:
            self.countVec = np.zeros(max_val - min_val + 1, dtype=np.float64)
        else:
            self.countVec = None
        self.key = keys[0]

    def update(self, x: int, weight: float, estimate: Optional[Model]) -> None:
        """Add one weighted integer, extending the represented range."""
        del estimate
        self._merge_counts(x, np.asarray([weight], dtype=np.float64))

    def initialize(self, x: int, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: IntegerEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Add encoded integer observations with corresponding weights."""
        del estimate
        if len(x) == 0:
            return
        minimum = int(x.min())
        counts = np.bincount(x - minimum, weights=weights).astype(np.float64)
        self._merge_counts(minimum, counts)

    def seq_initialize(
        self,
        x: IntegerEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def _merge_counts(self, minimum: int, counts: Array) -> None:
        """Merge a contiguous count vector into the represented range."""
        maximum = minimum + len(counts) - 1
        if self.countVec is None or self.minVal is None or self.maxVal is None:
            self.minVal = minimum
            self.maxVal = maximum
            self.countVec = counts.copy()
            return
        new_minimum = min(self.minVal, minimum)
        new_maximum = max(self.maxVal, maximum)
        merged = np.zeros(new_maximum - new_minimum + 1, dtype=np.float64)
        start = self.minVal - new_minimum
        merged[start : start + len(self.countVec)] += self.countVec
        start = minimum - new_minimum
        merged[start : start + len(counts)] += counts
        self.minVal = new_minimum
        self.maxVal = new_maximum
        self.countVec = merged

    def combine(self, suff_stat: IntegerSuffStat) -> "IntegerCategoricalAccumulator":
        """Merge a minimum and contiguous count vector."""
        self._merge_counts(suff_stat[0], suff_stat[1])
        return self

    def value(self) -> IntegerSuffStat:
        """Return ``(minimum, count_vector)``."""
        if self.minVal is None or self.countVec is None:
            raise ValueError("Integer categorical accumulator is empty.")
        return self.minVal, self.countVec.copy()

    def from_value(self, x: IntegerSuffStat) -> "IntegerCategoricalAccumulator":
        """Restore a minimum and contiguous count vector."""
        self.minVal = x[0]
        self.maxVal = x[0] + len(x[1]) - 1
        self.countVec = x[1].copy()
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics through the configured sharing key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace statistics through the configured sharing key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Create the compatible integer categorical encoder."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(
    StatisticAccumulatorFactory[int, IntegerSuffStat, IntegerEncoded]
):
    """Create integer categorical accumulators for an optional fixed range."""

    def __init__(
        self,
        min_val: Optional[int],
        max_val: Optional[int],
        keys: tuple[Optional[str], ...] = (None,),
    ) -> None:
        """Store range and key settings copied into each accumulator."""
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys

    def make(self) -> IntegerCategoricalAccumulator:
        """Create an empty integer categorical accumulator."""
        return IntegerCategoricalAccumulator(self.min_val, self.max_val, self.keys)


class IntegerCategoricalEstimator(
    ParameterEstimator[int, Array, IntegerEncoded, IntegerSuffStat]
):
    """Estimate contiguous integer categorical probabilities from counts.

    A dense or symmetric Dirichlet prior is updated to dense posterior
    concentrations and probabilities are set to its mode when defined,
    otherwise its mean. With another prior, normalized counts determine the
    probabilities and the prior is retained.
    """

    def __init__(
        self,
        min_index: Optional[int] = None,
        max_index: Optional[int] = None,
        default_value: float = 0.0,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: tuple[Optional[str], ...] = (None,),
    ) -> None:
        """Initialize optional support bounds, metadata, and prior."""
        self.minVal = min_index
        self.maxVal = max_index
        self.default_value = default_value
        self.keys = keys
        self.name = name
        self.set_prior(prior)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the estimator prior."""
        self.prior = prior

    def accumulator_factory(self) -> IntegerCategoricalAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return IntegerCategoricalAccumulatorFactory(self.minVal, self.maxVal, self.keys)

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> IntegerCategoricalDistribution:
        """Estimate probabilities from ``(minimum, count_vector)``.

        Args:
            *args: Either the statistic tuple alone or legacy ``nobs`` followed
                by it. ``nobs`` is ignored; the vector has one weighted count
                per consecutive integer beginning at ``minimum``.

        Returns:
            Fitted integer categorical likelihood on the statistic's range.
        """
        minimum, counts = args[-1]
        if isinstance(self.prior, DirichletDistribution):
            concentrations = self.prior.concentrations_for_dimension(len(counts))
            posterior = concentrations + counts
            weights = np.maximum(posterior - 1.0, 0.0)
            probabilities = (
                weights / weights.sum()
                if weights.sum() > 0.0
                else posterior / posterior.sum()
            )
            next_prior: Model = DirichletDistribution(posterior)
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            concentration = self.prior.get_parameters()
            posterior = counts + concentration
            weights = np.maximum(posterior - 1.0, 0.0)
            probabilities = (
                weights / weights.sum()
                if weights.sum() > 0.0
                else posterior / posterior.sum()
            )
            next_prior = DirichletDistribution(posterior)
        else:
            probabilities = counts / counts.sum()
            next_prior = self.prior
        return IntegerCategoricalDistribution(
            probabilities,
            default_value=self.default_value,
            min_index=minimum,
            name=self.name,
            prior=next_prior,
        )


class IntegerCategoricalDataEncoder(DataSequenceEncoder[int, IntegerEncoded]):
    """Encode integer categorical observations."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "IntegerCategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has integer categorical semantics."""
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(self, x: Iterable[int]) -> "IntegerCategoricalEncodedData":
        """Encode observations in a typed container."""
        return IntegerCategoricalEncodedData(np.asarray(tuple(x), dtype=int))


class IntegerCategoricalEncodedData(EncodedDataSequence[IntegerEncoded]):
    """Contain an encoded integer categorical sequence."""

    def __init__(self, data: IntegerEncoded) -> None:
        """Store the integer array."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"IntegerCategoricalEncodedData(data={self.data!r})"
