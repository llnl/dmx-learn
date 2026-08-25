"""Categorical distributions, sampling, estimation, and sequence encoding.

``CategoricalDistribution`` accepts any hashable category as a scalar
observation. Sequence methods consume the integer-coded representation produced
by ``CategoricalDataEncoder``.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.random import RandomState

from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

T = TypeVar("T")


class CategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a categorical distribution over hashable values.

    ``pmap`` maps explicit support values to their probability weights. A value
    absent from ``pmap`` receives weight ``default_value``. Scalar and sequence
    likelihoods divide either weight by ``1 + default_value``; callers normally
    provide an explicit probability map whose values sum to one.

    Attributes:
        name (Optional[str]): Optional name for the distribution.
        pmap (Dict[Any, float]): Explicit category-to-weight mapping.
        default_value (float): Weight assigned to values outside ``pmap``.
        no_default (bool): Whether the supplied default weight is nonzero.
        log_default_value (float): Logarithm of the default weight.
        log1p_default_value (float): Logarithm of ``1 + default_value``.
        keys (Optional[str]): Optional key for tying sufficient statistics.

    """

    def __init__(
        self,
        pmap: Dict[Any, float],
        default_value: float = 0.0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a categorical distribution.

        Args:
            pmap: Mapping from hashable category labels to probability weights.
            default_value: Weight for any category absent from ``pmap``. The stored
                value is clipped to ``[0, 1]``.
            name: Optional name for the distribution.
            keys: Optional key for tying sufficient statistics.
        """
        super().__init__()
        self.name = name
        self.pmap = pmap
        self.no_default = default_value != 0.0
        self.default_value = max(0.0, min(default_value, 1.0))
        self.log_default_value = float(
            -np.inf if default_value == 0 else math.log(default_value)
        )
        self.log1p_default_value = float(math.log1p(default_value))
        self.keys = keys

    def __str__(self) -> str:
        """Returns a string representation of the CategoricalDistribution object.

        Returns:
            str: String with pmap, default_value, and name printed.
        """
        s1 = ", ".join(
            [
                f"{repr(k)}: {repr(float(v))}"
                for k, v in sorted(self.pmap.items(), key=lambda u: u[0])
            ]
        )
        s2 = repr(self.default_value)
        s3 = repr(self.name)
        s4 = repr(self.keys)

        return (
            f"CategoricalDistribution({{{s1}}}, default_value={s2}, name={s3}, "
            f"keys={s4})"
        )

    def density(self, x: Any) -> float:
        """Evaluate the probability mass at a category.

        Args:
            x: Hashable category label.

        Returns:
            Probability mass at ``x``.
        """
        return self.pmap.get(x, self.default_value) / (1.0 + self.default_value)

    def log_density(self, x: Any) -> float:
        """Evaluate the log-probability mass at a category.

        Args:
            x: Hashable category label.

        Returns:
            Log-probability mass at ``x``.
        """
        return float(
            np.log(self.pmap.get(x, self.default_value)) - self.log1p_default_value
        )

    def seq_log_density(self, x: "CategoricalEncodedDataSequence") -> np.ndarray:
        """Evaluate log-probability masses for an encoded sequence.

        Args:
            x: Encoded sequence containing ``N`` categorical observations.

        Returns:
            Array of shape ``(N,)`` containing one log-probability mass per
            observation.
        """
        if not isinstance(x, CategoricalEncodedDataSequence):
            raise TypeError(
                "CategoricalDistribution.seq_log_density() requires "
                "CategoricalEncodedDataSequence."
            )

        with np.errstate(divide="ignore"):
            xs, val_map_inv = x.data
            mapped_log_prob = np.asarray(
                [self.pmap.get(u, self.default_value) for u in val_map_inv],
                dtype=np.float64,
            )
            np.log(mapped_log_prob, out=mapped_log_prob)
            mapped_log_prob -= self.log1p_default_value
            rv = mapped_log_prob[xs]

        return np.asarray(rv)

    def sampler(self, seed: Optional[int] = None) -> "CategoricalSampler":
        """Creates a CategoricalSampler for sampling from the CategoricalDistribution.

        Args:
            seed (Optional[int], optional): Seed for setting random number generator
                used to sample. Defaults to None.

        Returns:
            CategoricalSampler: Sampler object for the distribution.
        """
        return CategoricalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "CategoricalEstimator":
        """Create an estimator initialized from this distribution.

        When supplied, ``pseudo_count`` weights this distribution's probability
        map as prior category proportions.

        Args:
            pseudo_count (Optional[float], optional): If set, inflates counts for
                currently set sufficient statistic (pmap). Defaults to None.

        Returns:
            CategoricalEstimator: Estimator object for the distribution.
        """
        if pseudo_count is None:
            return CategoricalEstimator(name=self.name, keys=self.keys)

        return CategoricalEstimator(
            pseudo_count=pseudo_count,
            suff_stat=self.pmap,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "CategoricalDataEncoder":
        """Returns a CategoricalDataEncoder for this distribution.

        Returns:
            CategoricalDataEncoder: Encoder for categorical data.
        """
        return CategoricalDataEncoder()


class CategoricalSampler(DistributionSampler):
    """CategoricalSampler object used to generate samples from CategoricalDistribution.

    Attributes:
         rng (RandomState): RandomState with seed set to seed if provided. Else just
             RandomState().
         levels (List[Any]): Category labels for the CategoricalDistribution.
         probs (List[float]): Probabilities for each category in
             CategoricalDistribution.
         num_levels (int): Total number of categories. I.e. len(levels).

    """

    def __init__(
        self, dist: CategoricalDistribution, seed: Optional[int] = None
    ) -> None:
        """Initializes a CategoricalSampler object.

        Args:
            dist (CategoricalDistribution): CategoricalDistribution used to draw samples
                from.
            seed (Optional[int], optional): Seed for setting random number generator
                used to sample. Defaults to None.
        """
        super().__init__(dist, seed)
        temp = list(dist.pmap.items())
        self.levels = [u[0] for u in temp]
        self.probs = [u[1] for u in temp]
        self.num_levels = len(self.levels)

    def sample(self, size: Optional[int] = None) -> Union[Any, List[Any]]:
        """Draws samples from the CategoricalSampler object.

        Args:
            size (Optional[int], optional): Number of samples to draw. If None, draws a
                single sample. Defaults to None.

        Returns:
            Union[Any, List[Any]]: List of levels if size > 1, else a single sample from
            levels with prob probs.
        """
        if size is None:
            idx = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return self.levels[idx]

        levels = self.levels
        rv = self.rng.choice(self.num_levels, p=self.probs, size=size)

        return [levels[i] for i in rv]


class CategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted counts for categorical observations.

    The sufficient statistic is a dictionary from every observed category to its
    total weight.

    Attributes:
        count_map (Dict[Any, float]): Weighted count for each observed category.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initializes a CategoricalAccumulator object.

        Args:
            name (Optional[str], optional): Name for object. Defaults to None.
            keys (Optional[str], optional): All CategoricalAccumulators with same keys
                will have suff-stats merged. Defaults to None.
        """
        self.count_map: Dict[Any, float] = {}
        self.name = name
        self.key = keys

    def update(
        self, x: Any, weight: float, estimate: Optional["CategoricalDistribution"]
    ) -> None:
        """Updates the accumulator with a new observation.

        Args:
            x (Any): Observed value.
            weight (float): Weight of the observation.
            estimate (Optional[CategoricalDistribution]): Current estimate of the
                distribution.
        """
        self.count_map[x] = self.count_map.get(x, 0.0) + weight

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initializes the accumulator with the first observation.

        Args:
            x (Any): Observed value.
            weight (float): Weight of the observation.
            rng (RandomState): Random number generator.
        """
        del rng
        self.update(x, weight, None)

    def get_seq_lambda(self) -> List:
        """Returns a list of sequence update functions.

        Returns:
            list: List containing the seq_update function.
        """
        return [self.seq_update]

    def seq_update(
        self,
        x: "CategoricalEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["CategoricalDistribution"],
    ) -> None:
        """Vectorized update of the accumulator with a sequence of encoded data.

        Args:
            x (CategoricalEncodedDataSequence): Encoded sequence of categorical data.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[CategoricalDistribution]): Current estimate of the
                distribution.
        """
        inv_key_map = x.data[1]
        bcnt = np.bincount(x.data[0], weights=weights)

        if len(self.count_map) == 0:
            self.count_map = dict(zip(inv_key_map, bcnt))

        else:
            for i, bcnt_i in enumerate(bcnt):
                self.count_map[inv_key_map[i]] += bcnt_i

    def seq_initialize(
        self,
        x: "CategoricalEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Initializes the accumulator with a sequence of encoded data.

        Args:
            x (CategoricalEncodedDataSequence): Encoded sequence of categorical data.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Random number generator.
        """
        return self.seq_update(x, weights, None)

    def combine(self, suff_stat: Dict[Any, float]) -> "CategoricalAccumulator":
        """Combines another sufficient statistic into this accumulator.

        Args:
            suff_stat (Dict[Any, float]): Sufficient statistic to combine.

        Returns:
            CategoricalAccumulator: The updated accumulator.
        """
        for k, v in suff_stat.items():
            self.count_map[k] = self.count_map.get(k, 0.0) + v

        return self

    def value(self) -> Dict[Any, float]:
        """Returns the current sufficient statistic.

        Returns:
            Dict[Any, float]: Copy of the current count map.
        """
        return self.count_map.copy()

    def from_value(self, x: Dict[Any, float]) -> "CategoricalAccumulator":
        """Sets the accumulator's value from a given sufficient statistic.

        Args:
            x (Dict[Any, float]): Sufficient statistic to set.

        Returns:
            CategoricalAccumulator: The updated accumulator.
        """
        self.count_map = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merges this accumulator into a dictionary of accumulators by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())

            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replaces this accumulator's value with the value from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "CategoricalDataEncoder":
        """Returns a CategoricalDataEncoder for this accumulator.

        Returns:
            CategoricalDataEncoder: Encoder for categorical data.
        """
        return CategoricalDataEncoder()


class CategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """CategoricalAccumulatorFactory object.

    Attributes:
        keys (Optional[str]): Key for merging sufficient statistics.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initializes a CategoricalAccumulatorFactory object.

        Args:
            name (Optional[str], optional): Name for object. Defaults to None.
            keys (Optional[str], optional): Declare keys for merging sufficient
                statistics of CategoricalAccumulators. Defaults to None.
        """
        self.name = name
        self.keys = keys

    def make(self) -> "CategoricalAccumulator":
        """Creates a new CategoricalAccumulator.

        Returns:
            CategoricalAccumulator: New accumulator instance.
        """
        return CategoricalAccumulator(name=self.name, keys=self.keys)


class CategoricalEstimator(ParameterEstimator):
    """Estimate categorical probabilities from weighted category counts.

    Without a pseudo-count, counts are normalized directly. With a pseudo-count
    but no prior map, symmetric smoothing is spread over observed categories.
    With both, the union of observed and prior categories is normalized after
    adding the weighted prior map. ``default_value=True`` also assigns unseen
    values the implementation's data-dependent default weight.

    Attributes:
        pseudo_count (Optional[float]): Inflate sufficient statistic counts by
            pseudo_count.
        suff_stat (Optional[Dict[Any, float]]): Dictionary with category labels and
            probabilities as values.
        default_value (bool): True is default value should be set.
        name (Optional[str]): Assign name to be passed to Distribution, Accumulator,
            ect.
        keys (Optional[str]): Assign key to Estimator designating all same key
            estimators to later be combined,
            in accumulation.

    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Dict[Any, float]] = None,
        default_value: bool = False,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initializes a CategoricalEstimator object.

        Args:
            pseudo_count (Optional[float], optional): Inflate sufficient statistic
                counts by pseudo_count. Defaults to None.
            suff_stat (Optional[Dict[Any, float]], optional): Dictionary with category
                labels and probabilities as values. Defaults to None.
            default_value (bool, optional): True if default value should be set.
                Defaults to False.
            name (Optional[str], optional): Assign name to be passed to Distribution,
                Accumulator, etc. Defaults to None.
            keys (Optional[str], optional): Assign key to Estimator designating all same
                key estimators to later be combined, in accumulation. Defaults to None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("CategoricalEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.default_value = default_value
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "CategoricalAccumulatorFactory":
        """Returns a CategoricalAccumulatorFactory for this estimator.

        Returns:
            CategoricalAccumulatorFactory: Factory for creating accumulators.
        """
        return CategoricalAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Dict[Any, float]
    ) -> "CategoricalDistribution":
        """Estimates a CategoricalDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with
                ParameterEstimator.estimate.
            suff_stat (Dict[Any, float]): Dict with categories as keys and counts as
                values from accumulated data.

        Returns:
            CategoricalDistribution: Estimated distribution.
        """
        stats_sum = sum(suff_stat.values())

        if self.default_value:
            if stats_sum > 0:
                default_value = 1.0 / stats_sum
                default_value *= default_value

            else:
                default_value = 0.5
        else:
            default_value = 0.0

        if self.pseudo_count is None and self.suff_stat is None:
            nobs_loc = stats_sum

            if nobs_loc == 0.0:
                p_map = {k: 1.0 / float(len(suff_stat)) for k in suff_stat.keys()}
            else:
                p_map = {k: v / nobs_loc for k, v in suff_stat.items()}

        elif self.pseudo_count is not None and self.suff_stat is None:
            nobs_loc = stats_sum
            pseudo_count_per_level = self.pseudo_count / len(suff_stat)
            adjusted_nobs = nobs_loc + self.pseudo_count

            for k, v in suff_stat.items():
                suff_stat[k] = (v + pseudo_count_per_level) / adjusted_nobs

            p_map = suff_stat

        else:
            assert self.pseudo_count is not None
            assert self.suff_stat is not None
            suff_stat_sum = sum(self.suff_stat.values())

            levels = set(suff_stat.keys()).union(self.suff_stat.keys())
            adjusted_nobs = suff_stat_sum * self.pseudo_count + stats_sum

            p_map = {
                k: (suff_stat.get(k, 0) + self.suff_stat.get(k, 0) * self.pseudo_count)
                / adjusted_nobs
                for k in levels
            }

        return CategoricalDistribution(
            pmap=p_map, default_value=default_value, name=self.name
        )


class CategoricalDataEncoder(DataSequenceEncoder):
    """Encode categorical observations for vectorized operations.

    For ``N`` observations and ``U`` unique values, the encoded representation
    contains integer inverse indices with shape ``(N,)`` and unique category
    labels with shape ``(U,)``.
    """

    def __str__(self) -> str:
        """Returns a string representation of the encoder.

        Returns:
            str: String representation.
        """
        return "CategoricalDataEncoder"

    def __eq__(self, other: Any) -> bool:
        """Checks equality with another encoder.

        Args:
            other (Any): Object to compare.

        Returns:
            bool: True if other is a CategoricalDataEncoder, False otherwise.
        """
        return isinstance(other, CategoricalDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> "CategoricalEncodedDataSequence":
        """Encodes a sequence of categorical data for use with vectorized functions.

        Args:
            x: Sequence of ``N`` category labels.

        Returns:
            Encoded sequence with data ``(inverse_indices, unique_values)``.
        """
        val_map_inv, uidx, xs = np.unique(x, return_index=True, return_inverse=True)
        val_map_inv = np.asarray([x[i] for i in uidx], dtype=object)

        return CategoricalEncodedDataSequence(data=(xs, val_map_inv))


class CategoricalEncodedDataSequence(EncodedDataSequence):
    """Store an integer-coded sequence of categorical observations.

    Attributes:
        data: Tuple of inverse indices with shape ``(N,)`` and unique values with
            shape ``(U,)``.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Initializes a CategoricalEncodedDataSequence object.

        Args:
            data (Tuple[np.ndarray, np.ndarray]): Inverse mapping of unique values,
                unique values.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Returns a string representation of the encoded data sequence.

        Returns:
            str: String representation.
        """
        return f"CategoricalEncodedDataSequence(data={self.data})"
