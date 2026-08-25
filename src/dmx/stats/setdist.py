r"""Model unordered finite sets with independent Bernoulli inclusions.

An observation is a collection of distinct, hashable values whose order has no
probabilistic meaning. For every value :math:`u` in ``pmap``, inclusion is an
independent Bernoulli event with probability :math:`p_u`. Values absent from
``pmap`` are outside the modeled support and cause evaluation to fail.

The accumulator records the weighted number of sets containing each observed value and
the total weight of all sets. Thus an estimator discovers support from observed values;
values never observed require prior sufficient statistics to appear in an estimate.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


class BernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Model an unordered set by independent element-inclusion probabilities.

    Each key of ``pmap`` defines one possible element. Duplicate values in an input are
    counted repeatedly, so callers must supply collections with unique elements.

    Attributes:
        keys (Optional[str]): Keys for object instance.
        name (Optional[str]): Name to object instance.
        pmap (Dict[Any, float]): Maps elements in support to probabilities.
        required (Set): An observation must contain this subset of elements. Else,
            return probability 0.0.
        nlog_sum (float): Normalizing term for computing numerically stable likelihood.
        log_dmap (Dict[Any, float]):Map from elements to their corrected log probability
        of inclusion in the set.
        min_prob (float): Minimum probability for elements. Corrects for prob = 0.
        num_required (int): Number of required elements in a subset. Corrected if
            min_prob was non-zero.

    """

    def __init__(
        self,
        pmap: Dict[Any, float],
        min_prob: float = 1.0e-128,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize an independent Bernoulli set distribution.

        Args:
            pmap (Dict[Any, float]): Maps values to probabilities.
            min_prob (float): Minimum probability for numerical stability in log prob
                calculations.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        super().__init__()
        self.keys = keys
        self.name = name
        self.pmap = pmap
        self.required = set()
        self.nlog_sum = 0.0
        self.log_dmap = {}

        if min_prob == 0:
            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = 0.0
                    self.required.add(k)
                elif v == 0.0:
                    self.log_dmap[k] = -np.inf
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv
            self.min_prob = 0.0
            self.num_required = len(self.required)

        else:
            min_pv = np.log(min_prob)
            min_nv = np.log1p(-min_prob)

            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = min_nv - min_pv
                    self.nlog_sum += min_pv
                elif v == 0.0:
                    self.log_dmap[k] = min_pv - min_nv
                    self.nlog_sum += min_nv
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv

            self.min_prob = min_prob
            self.num_required = 0

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s1 = repr(sorted(self.pmap.items(), key=lambda t: t[0]))
        s2 = repr(self.min_prob)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return (
            f"BernoulliSetDistribution(dict({s1}), min_prob={s2}, name={s3}, keys={s4})"
        )

    def density(self, x: Sequence[Any]) -> float:
        """Evaluate the probability of the unordered set ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[Any]) -> float:
        """Evaluate the log probability of the unordered set ``x``.

        Input order is ignored. Every value must be a key in ``pmap`` and callers must
        exclude duplicates, which the implementation would count repeatedly.
        """
        if not self.required.issubset(x):
            return -np.inf
        rv = 0.0
        for v in x:
            rv += self.log_dmap[v]

        return self.nlog_sum + rv

    def seq_log_density(self, x: "BernoulliSetEncodedDataSequence") -> np.ndarray:
        """Evaluate log probabilities for encoded set observations."""
        if not isinstance(x, BernoulliSetEncodedDataSequence):
            raise TypeError(
                "BernoulliSetEncodedDataSequence required for seq_log_density()."
            )

        sz, idx, val_map_inv, xs = x.data

        dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)

        rv = np.asarray(
            np.bincount(idx, weights=dlog_loc[xs], minlength=sz), dtype=float
        )
        rv += self.nlog_sum

        if self.num_required != 0:
            required_loc = np.isin(val_map_inv, list(self.required))
            req_cnt = np.bincount(idx, weights=required_loc[xs], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf

        return rv

    def sampler(self, seed: Optional[int] = None) -> "BernoulliSetSampler":
        """Create a sampler using independent inclusion draws."""
        return BernoulliSetSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "BernoulliSetEstimator":
        """Create an estimator, optionally centered on this distribution."""
        if pseudo_count is None:
            return BernoulliSetEstimator(
                min_prob=self.min_prob, name=self.name, keys=self.keys
            )
        return BernoulliSetEstimator(
            min_prob=self.min_prob,
            pseudo_count=pseudo_count,
            suff_stat=self.pmap,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Create an encoder for batches of unordered set observations."""
        return BernoulliSetDataEncoder()


class BernoulliSetSampler(DistributionSampler):
    """Generate unordered-set observations from a Bernoulli set distribution.

    Attributes:
        dist (BernoulliSetDistribution): Object instance to sample from.
        seed (Optional[int]): Set seed for random number generator.

    """

    def __init__(
        self, dist: BernoulliSetDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler.

        Args:
            dist (BernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Set seed for random number generator.

        """
        super().__init__(dist, seed)

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Sequence[Any], List[Sequence[Any]]]:
        """Draw one set, or ``size`` independent sets."""
        if size is not None:
            retval: List[List[Any]] = [[] for i in range(size)]
            for k, v in self.dist.pmap.items():
                for i in np.flatnonzero(self.rng.rand(size) <= v):
                    retval[i].append(k)
            return retval

        retval = []
        for k, v in self.dist.pmap.items():
            if self.rng.rand() <= v:
                retval.append(k)
        return retval


class BernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted element inclusions and total set weight.

    Attributes:
        pmap (Dict[Any, float]): Weighted inclusion count for each observed value.
        tot_sum (float): Weighted observation count.
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize an empty Bernoulli-set accumulator.

        Args:
            keys (Optional[str]): Set keys for merging sufficient statistics.
            name (Optional[str]): Name for object.

        """
        self.pmap: Dict[Any, float] = defaultdict(float)
        self.tot_sum = 0.0
        self.keys = keys
        self.name = name

    def update(
        self,
        x: Sequence[Any],
        weight: float,
        estimate: Optional[BernoulliSetDistribution],
    ) -> None:
        """Add one weighted unordered set to the sufficient statistics."""
        for u in x:
            self.pmap[u] += weight
        self.tot_sum += weight

    def initialize(
        self, x: Sequence[Any], weight: float, rng: Optional[RandomState]
    ) -> None:
        """Initialize statistics from one set; randomness is unused."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: "BernoulliSetEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[BernoulliSetDistribution],
    ) -> None:
        """Add weighted encoded sets to the sufficient statistics."""
        _sz, idx, val_map_inv, xs = x.data
        agg_cnt = np.bincount(xs, weights[idx])

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += weights.sum()

    def seq_initialize(
        self,
        x: "BernoulliSetEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize statistics from encoded sets; randomness is unused."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[Dict[Any, float], float]
    ) -> "BernoulliSetAccumulator":
        """Merge element counts and total weight into this accumulator."""
        for k, v in suff_stat[0].items():
            self.pmap[k] += v
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[Dict[Any, float], float]:
        """Return element counts and total observation weight."""
        return dict(self.pmap), self.tot_sum

    def from_value(
        self, x: Tuple[Dict[Any, float], float]
    ) -> "BernoulliSetAccumulator":
        """Replace the sufficient statistics from a serialized value."""
        self.pmap = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics into ``stats_dict`` under the configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from ``stats_dict`` when the key is present."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Create the encoder used by vectorized accumulator methods."""
        return BernoulliSetDataEncoder()


class BernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """Create consistently configured Bernoulli set accumulators.

    Attributes:
        keys (Optional[str]): Key for suff stats.
        name (Optional[str]): Name for object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """Initialize the accumulator factory.

        Args:
            keys (Optional[str]): Key for suff stats.
            name (Optional[str]): Name for object.

        """
        self.keys = keys
        self.name = name

    def make(self) -> "BernoulliSetAccumulator":
        """Create an empty Bernoulli set accumulator."""
        return BernoulliSetAccumulator(keys=self.keys, name=self.name)


class BernoulliSetEstimator(ParameterEstimator):
    """Estimate inclusion probabilities from weighted element counts.

    Attributes:
        min_prob (float): Minimum probability for elements estimated with prob = 0.
        pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
        suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to
            probability mapping.
        name (Optional[str]): Set name for object instance.
        keys (Optional[str]): Set key for merging sufficient statistics.

    """

    def __init__(
        self,
        min_prob: float = 1.0e-128,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Dict[Any, float]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize the Bernoulli set estimator.

        Args:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value
                to probability mapping.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for merging sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("BernoulliSetEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> "BernoulliSetAccumulatorFactory":
        """Create a factory for compatible sufficient-statistic accumulators."""
        return BernoulliSetAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[Dict[Any, float], float]
    ) -> "BernoulliSetDistribution":
        """Estimate a distribution from element counts and total set weight.

        ``nobs`` is ignored. With no prior statistics, each inclusion probability is
        its weighted count divided by the total weight. A pseudo-count alone adds a
        symmetric half-present, half-absent prior for each observed element.
        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            keys = set(suff_stat[0].keys())
            keys.update(self.suff_stat.keys())

            pmap = {
                k: (
                    self.suff_stat.get(k, 0.0) * self.pseudo_count
                    + suff_stat[0].get(k, 0.0)
                )
                / (self.pseudo_count + suff_stat[1])
                for k in keys
            }

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            cnt = float(p + suff_stat[1])
            pmap = {k: (v + (p / 2.0)) / cnt for k, v in suff_stat[0].items()}

        else:

            if suff_stat[1] != 0:
                pmap = {k: v / suff_stat[1] for k, v in suff_stat[0].items()}
            else:
                pmap = {k: 0.5 for k in suff_stat[0].keys()}

        return BernoulliSetDistribution(pmap, min_prob=self.min_prob, name=self.name)


class BernoulliSetDataEncoder(DataSequenceEncoder):
    """Encode batches of unordered sets into flattened value indices."""

    def __str__(self) -> str:
        """Return the encoder name."""
        return "BernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is the same stateless encoder type."""
        return isinstance(other, BernoulliSetDataEncoder)

    def seq_encode(
        self, x: Sequence[Sequence[Any]]
    ) -> "BernoulliSetEncodedDataSequence":
        """Encode sets as observation indices, unique values, and value indices.

        The flattened representation preserves duplicates if callers provide them;
        valid set observations therefore contain each value at most once.
        """
        idx: List[int] = []
        xs: List[Any] = []

        for i, x_i in enumerate(x):
            idx.extend([i] * len(x_i))
            xs.extend(x_i)

        val_map, xs_inverse = np.unique(
            np.asarray(xs, dtype=object), return_inverse=True
        )

        idx_arr = np.asarray(idx, dtype=np.int32)
        xs_arr = np.asarray(xs_inverse, dtype=np.int32)

        return BernoulliSetEncodedDataSequence(data=(len(x), idx_arr, val_map, xs_arr))


class BernoulliSetEncodedDataSequence(EncodedDataSequence):
    """Store the flattened representation of a batch of unordered sets.

    Attributes:
        data: Tuple containing the number of sets, an observation index per flattened
            element, the unique-value lookup array, and a lookup index per flattened
            element.

    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
        """Initialize an encoded batch of set observations.

        Args:
            data: Number of sets, flattened observation indices, unique values, and
                flattened indices into the unique-value array.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded tuple."""
        return f"BernoulliSetEncodedDataSequence(data={self.data})"
