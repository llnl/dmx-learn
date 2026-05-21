# pylint: disable=line-too-long
"""Create, estimate, and sample from a integer set Bernoulli distribution.

Defines the IntegerBernoulliSetDistribution, IntegerBernoulliSetSampler, IntegerBernoulliSetAccumulatorFactory,
IntegerBernoulliSetAccumulator, IntegerBernoulliSetEstimator, and the IntegerBernoulliSetDataEncoder classes for use
with pysparkplug.


Let S = {0,1,2,3...,N-1} be a set if integers. Let x_mat be a random subset of S. The Bernoulli set distribution models
random subset of S as

    p_k = p_mat(k is in x_mat) , k = 0,2,...,N-1.

The density for an observed subset of S, x=(x_1,x_2,..,x_m), for m < N) is given by
    p_mat(x) = sum_{k=0}^{K-1}( p_k*(k in x) + (1-p_k)*(k not in x)).

"""

# pylint: disable=line-too-long,too-many-positional-arguments,duplicate-code
# pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
# pylint: disable=broad-exception-raised,consider-using-f-string,no-else-return
# pylint: disable=no-else-raise,consider-using-enumerate,consider-using-generator
# pylint: disable=use-dict-literal,super-with-arguments,unnecessary-comprehension
# pylint: disable=simplifiable-if-statement,nested-min-max

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import *
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class IntegerBernoulliSetDistribution(TorchProbabilityDistribution):
    """IntegerBernoulliSetDistribution object defining a Bernoulli set distribution on integers [0,len(pvec)).

    Attributes:
        log_pvec (Tensor): Probability of integer k being in set.
        log_nvec (Tensor): Optional normalizing probability for each integer probability.
        log_dvec (Tensor): Normalized probability for each integer value.
        log_nsum (float): Sum of normalized probabilities used for easily adding unobserved (missing) integer
            values in an observation.
        key (Optional[str]): Set keys for object instance.

    """

    def __init__(
        self,
        log_pvec: Union[Sequence[float], np.ndarray],
        log_nvec: Optional[Union[Sequence[float], np.ndarray]] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerBernoulliSetDistribution object.

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Log probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            keys (Optional[str]): Set keys for object instance.
            device (Optional[tn.device]): Set device for tensor calculations.

        """
        super().__init__(device)
        num_vals = len(log_pvec)
        self.num_vals = num_vals
        self.log_pvec = vec.tensor(log_pvec, device=self._device)
        self.key = keys

        if log_nvec is None:
            log_nvec = tn.log1p(-tn.exp(self.log_pvec))
            self.log_nvec = None
            self.log_dvec = self.log_pvec - log_nvec
            self.log_nsum = tn.sum(log_nvec[tn.isfinite(log_nvec)])
        else:
            self.log_nvec = vec.tensor(log_nvec, device=self._device)
            self.log_dvec = self.log_pvec - self.log_nvec
            self.log_nsum = tn.sum(self.log_nvec[tn.isfinite(self.log_nvec)])

    def to(self, device: tn.device) -> None:
        self.log_pvec = self.log_pvec.to(device)
        self.log_nvec = self.log_nvec.to(device) if self.log_nvec is not None else None
        self.log_dvec = self.log_dvec.to(device)
        self.log_nsum = self.log_nsum.to(device)
        self._device = device

    def __repr__(self) -> str:
        s1 = repr(self.log_pvec.cpu().detach().tolist())
        s2 = repr(
            None if self.log_nvec is None else self.log_nvec.cpu().detach().tolist()
        )

        return "IntegerBernoulliSetDistribution(%s, log_nvec=%s)" % (s1, s2)

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        xx = np.asarray(x, dtype=int)
        rv = tn.sum(self.log_dvec[xx]) + self.log_nsum

        return float(rv)

    def seq_log_density(self, x: "IntegerBernoulliSetTorchSequence") -> tn.Tensor:

        if not isinstance(x, IntegerBernoulliSetTorchSequence):
            raise Exception(
                "Requires IntegerBernoulliSetTorchSequence for `seq_` calls."
            )
        sz, idx, xs = x.data
        rv = vec.zeros(sz, device=self.model_device())
        xs_dev = xs.to(device=self.log_dvec.device)
        idx_dev = idx.to(device=self.model_device())
        rv += tn.bincount(
            idx_dev,
            weights=self.log_dvec[xs_dev].to(device=self.model_device()),
            minlength=sz,
        )
        rv += self.log_nsum

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerBernoulliSetSampler":
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerBernoulliSetEstimator":
        return IntegerBernoulliSetEstimator(self.num_vals, pseudo_count=pseudo_count)

    def dist_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetSampler(DistributionSampler):
    """IntegerBernoulliSetSampler object for sampling from an IntegerBernoulliSetDistribution instance.

    Attributes:
        rng (RandomState): RandomState object with seed set if passed in args.
        log_pvec (np.ndarray): Log probs for each value.
        num_vals (int): Number of total values.

    """

    def __init__(
        self, dist: IntegerBernoulliSetDistribution, seed: Optional[int] = None
    ) -> None:
        """IntegerBernoulliSetSampler object.

        Args:
            dist (IntegerBernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = np.random.RandomState(seed)
        self.log_pvec = dist.log_pvec.cpu().detach().numpy()
        self.num_vals = dist.num_vals

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Sequence[int]], Sequence[int]]:
        if size is None:
            log_u = np.log(self.rng.rand(self.num_vals))
            return list(np.flatnonzero(log_u <= self.log_pvec))
        else:
            rv = []
            for _ in range(size):
                log_u = np.log(self.rng.rand(self.num_vals))
                rv.append(list(np.flatnonzero(log_u <= self.log_pvec)))
            return rv


class IntegerBernoulliSetAccumulator(TorchStatisticAccumulator):
    """IntegerBernoulliSetAccumulator object for accumulating sufficient statistics from observed data.

    Attributes:
        pcnt (np.ndarray): Used for aggregating weighted counts of integers.
        key (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
        num_vals (int): Number of values in integer range for the set.
        tot_sum (float): Sum of weights for observations.

    """

    def __init__(
        self,
        num_vals: int,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerBernoulliSetAccumulator object for accumulating sufficient statistics from observed data.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            device (Optional[tn.device]): Device for Tensor calculations.

        """
        super().__init__(device=device)
        self.pcnt = np.zeros(num_vals, dtype=np.float64)
        self.key = keys
        self.num_vals = num_vals
        self.tot_sum = 0.0

    def seq_update(
        self,
        x: "IntegerBernoulliSetTorchSequence",
        weights: tn.Tensor,
        estimate: Optional[IntegerBernoulliSetDistribution],
    ) -> None:
        _, idx, xs = x.data
        agg_cnt = tn.bincount(xs, weights=weights[idx]).cpu().detach().numpy()
        n = len(agg_cnt)
        self.pcnt[:n] += agg_cnt
        self.tot_sum += float(weights.sum())

    def seq_initialize(
        self,
        x: "IntegerBernoulliSetTorchSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[np.ndarray, float]:
        return self.pcnt, self.tot_sum

    def from_value(
        self, x: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

    def acc_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """IntegerBernoulliSetAccumulatorFactory for creating IntegerBernoulliSetAccumulator objects.

    Attributes:
        keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
        num_vals (int): Number of values in integer range for the set.

    """

    def __init__(self, num_vals: int, keys: Optional[str] = None) -> None:
        """IntegerBernoulliSetAccumulatorFactory object.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        """
        self.keys = keys
        self.num_vals = num_vals

    def make(
        self, device: Optional[tn.device] = None
    ) -> "IntegerBernoulliSetAccumulator":
        return IntegerBernoulliSetAccumulator(
            self.num_vals, keys=self.keys, device=device
        )


class IntegerBernoulliSetEstimator(TorchParameterEstimator):
    """IntegerBernoulliSetEstimator object for estimating integer Bernoulli set distribution from sufficient statistics.

    Attributes:
        num_vals (int): Number of values in integer range for the set.
        keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
        pseudo_count (Optional[float]): Re-weight suff stats in estimation.
        suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
        min_prob (float): Minimum probability for an integer in range of set dist.

    """

    def __init__(
        self,
        num_vals: int,
        min_prob: float = 1.0e-128,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[np.ndarray] = None,
        keys: Optional[str] = None,
    ) -> None:
        """IntegerBernoulliSetEstimator object.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in range of set dist.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        """
        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_prob = min_prob

    def accumulator_factory(self) -> "IntegerBernoulliSetAccumulatorFactory":
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Optional[np.ndarray] = None,
        device: Optional[tn.device] = None,
    ) -> "IntegerBernoulliSetDistribution":
        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = np.product(self.suff_stat, self.pseudo_count)
            p1 = np.product(np.subtract(1.0, self.suff_stat), self.pseudo_count)
            pvec = np.log(suff_stat[0] + p0)
            nvec = np.log((suff_stat[1] - suff_stat[0]) + p1)
            tsum = np.log(suff_stat[1] + self.pseudo_count)
            log_pvec = np.log(pvec) - tsum
            log_nvec = np.log(nvec) - tsum

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            log_c = np.log(suff_stat[1] + p)
            log_pvec = np.log(suff_stat[0] + (p / 2.0)) - log_c
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + (p / 2.0)) - log_c

        else:

            if suff_stat[1] == 0:
                log_pvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5
                log_nvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5

            elif self.min_prob > 0:
                log_pvec = np.log(
                    np.maximum(suff_stat[0] / suff_stat[1], self.min_prob)
                )
                log_nvec = np.log(
                    np.maximum(
                        (suff_stat[1] - suff_stat[0]) / suff_stat[1], self.min_prob
                    )
                )

            else:
                pvec = suff_stat[0] / suff_stat[1]
                nvec = (suff_stat[1] - suff_stat[0]) / suff_stat[1]

                is_zero = pvec == 0
                is_one = nvec == 0

                log_pvec = np.zeros(self.num_vals, dtype=np.float64)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64)

                log_pvec[~is_zero] = np.log(pvec[~is_zero])
                log_pvec[is_zero] = -np.inf
                log_nvec[~is_one] = np.log(nvec[~is_one])
                log_nvec[is_one] = -np.inf

        return IntegerBernoulliSetDistribution(log_pvec, log_nvec, device=device)


class IntegerBernoulliSetDataEncoder(TorchSequenceEncoder):
    """IntegerBernoulliSetDataEncoder object for encoding sequences of iid integer Bernoulli set observations."""

    def __str__(self) -> str:
        return "IntegerBernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(object, IntegerBernoulliSetDataEncoder)

    def seq_encode(
        self, x: Sequence[Sequence[int]], device: Optional[tn.device] = None
    ) -> "IntegerBernoulliSetTorchSequence":
        idx = []
        xs = []
        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx = vec.int_tensor(idx, device=device)
        xs = vec.int_tensor(xs, device=device)

        return IntegerBernoulliSetTorchSequence(data=(len(x), idx, xs), device=device)


class IntegerBernoulliSetTorchSequence(TorchEncodedSequence):

    def __init__(
        self, data: Tuple[int, tn.Tensor, tn.Tensor], device: Optional[tn.device] = None
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"IntegerBernoulliSetTorchSequence(device={repr(self.device)})"
