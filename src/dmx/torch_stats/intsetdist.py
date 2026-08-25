"""Provide torch-backed Bernoulli distributions over finite integer sets.

Defines the IntegerBernoulliSetDistribution, IntegerBernoulliSetSampler,
IntegerBernoulliSetAccumulatorFactory, IntegerBernoulliSetAccumulator,
IntegerBernoulliSetEstimator, and the IntegerBernoulliSetDataEncoder classes
for use with pysparkplug.


Let S = {0,1,2,3...,N-1} be a set of integers. Let x_mat be a random subset of
S. The Bernoulli set distribution models a random subset of S as

    p_k = p_mat(k is in x_mat) , k = 0,2,...,N-1.

The density for an observed subset of S, x=(x_1,x_2,..,x_m), for m < N) is given by
    p_mat(x) = sum_{k=0}^{K-1}( p_k*(k in x) + (1-p_k)*(k not in x)).

Each observation is a sequence of included integers from ``[0, K)``. The
encoder flattens ``N`` sets into integer tensors of observation indices and
values, both of shape ``(M,)``. Parameters move in place with ``to``; encoded
scoring returns a floating tensor of shape ``(N,)`` on the model device.
Floating tensors use the vector-helper default dtype, normally float64 and
float32 on MPS. Sampling and sufficient statistics use CPU NumPy arrays. This
mirrors ``dmx.stats.intsetdist`` without its optional names.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import exp
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
    """Represent independent inclusions on integers ``[0, K)``.

    ``log_pvec[k]`` is the log inclusion probability for integer ``k``.
    ``log_nvec[k]`` is its log absence probability when explicitly supplied.

    Attributes:
        log_pvec (Tensor): Probability of integer k being in set.
        log_nvec (Tensor): Optional normalizing probability for each integer
            probability.
        log_dvec (Tensor): Normalized probability for each integer value.
        log_nsum (float): Sum of normalized probabilities used to add
            unobserved integer values in an observation.
        key (Optional[str]): Set keys for object instance.

    """

    def __init__(
        self,
        log_pvec: Union[Sequence[float], np.ndarray],
        log_nvec: Optional[Union[Sequence[float], np.ndarray]] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize an integer Bernoulli set distribution.

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Log probability of
                integer k being in the set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional
                normalizing probability for each integer probability.
            keys (Optional[str]): Set keys for object instance.
            device (Optional[tn.device]): Set device for tensor calculations.

        """
        super().__init__(device)
        num_vals = len(log_pvec)
        self.num_vals = num_vals
        self.log_pvec: tn.Tensor = vec.tensor(log_pvec, device=self._device)
        self.key = keys

        if log_nvec is None:
            log_nvec_tensor = tn.log1p(-tn.exp(self.log_pvec))
            self.log_nvec = None
            self.log_dvec = self.log_pvec - log_nvec_tensor
            self.log_nsum = tn.sum(log_nvec_tensor[tn.isfinite(log_nvec_tensor)])
        else:
            self.log_nvec = vec.tensor(log_nvec, device=self._device)
            self.log_dvec = self.log_pvec - self.log_nvec
            self.log_nsum = tn.sum(self.log_nvec[tn.isfinite(self.log_nvec)])

    def to(self, device: vec.DeviceLike) -> "IntegerBernoulliSetDistribution":
        """Move all parameter tensors to ``device`` in place and return ``self``."""
        target_device = self._resolve_device_arg(device)
        self.log_pvec = self.log_pvec.to(target_device)
        self.log_nvec = (
            self.log_nvec.to(target_device) if self.log_nvec is not None else None
        )
        self.log_dvec = self.log_dvec.to(target_device)
        self.log_nsum = self.log_nsum.to(target_device)
        self._device = target_device
        return self

    def __repr__(self) -> str:
        """Return a constructor-like representation with CPU parameters."""
        s1 = repr(self.log_pvec.cpu().detach().tolist())
        s2 = repr(
            None if self.log_nvec is None else self.log_nvec.cpu().detach().tolist()
        )

        return f"IntegerBernoulliSetDistribution({s1}, log_nvec={s2})"

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Evaluate the probability mass of one integer set."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Evaluate the log mass of one set using integer category indices."""
        xx = np.asarray(x, dtype=int)
        rv = tn.sum(self.log_dvec[xx]) + self.log_nsum

        return float(rv)

    def seq_log_density(self, x: "IntegerBernoulliSetTorchSequence") -> tn.Tensor:
        """Return log masses for ``N`` flattened encoded set observations."""
        if not isinstance(x, IntegerBernoulliSetTorchSequence):
            raise TypeError(
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
        """Create a CPU NumPy sampler, optionally initialized with ``seed``."""
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerBernoulliSetEstimator":
        """Create an estimator initialized with this support size."""
        return IntegerBernoulliSetEstimator(self.num_vals, pseudo_count=pseudo_count)

    def dist_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Create the compatible flattened set encoder."""
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetSampler(DistributionSampler):
    """Sample from an IntegerBernoulliSetDistribution instance.

    Attributes:
        rng (RandomState): RandomState object with seed set if passed in args.
        log_pvec (np.ndarray): Log probs for each value.
        num_vals (int): Number of total values.

    """

    def __init__(
        self, dist: IntegerBernoulliSetDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize an integer Bernoulli set sampler.

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
        """Draw one integer set or a list of ``size`` sets."""
        if size is None:
            log_u = np.log(self.rng.rand(self.num_vals))
            return list(np.flatnonzero(log_u <= self.log_pvec))

        rv: List[Sequence[int]] = []
        for _ in range(size):
            log_u = np.log(self.rng.rand(self.num_vals))
            rv.append(list(np.flatnonzero(log_u <= self.log_pvec)))
        return rv


class IntegerBernoulliSetAccumulator(TorchStatisticAccumulator):
    """Accumulate sufficient statistics from observed data.

    Attributes:
        pcnt (np.ndarray): Used for aggregating weighted counts of integers.
        key (Optional[str]): Keys for merging sufficient statistics with
            matching keyed objects.
        num_vals (int): Number of values in integer range for the set.
        tot_sum (float): Sum of weights for observations.

    """

    def __init__(
        self,
        num_vals: int,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize an integer Bernoulli set accumulator.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with
                matching keyed objects.
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
        """Accumulate ``N`` encoded sets with weights of shape ``(N,)``."""
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
        """Add encoded sets during initialization; ``tng`` is unused."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        """Merge ``(positive_counts, total_weight)`` statistics."""
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[np.ndarray, float]:
        """Return CPU inclusion counts of shape ``(K,)`` and total weight."""
        return self.pcnt, self.tot_sum

    def from_value(
        self, x: Tuple[np.ndarray, float]
    ) -> "IntegerBernoulliSetAccumulator":
        """Replace the accumulator from inclusion sufficient statistics."""
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics into ``stats_dict`` under the key."""
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from the matching entry in ``stats_dict``."""
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

    def acc_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Create the compatible flattened set encoder."""
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Factory for IntegerBernoulliSetAccumulator objects.

    Attributes:
        keys (Optional[str]): Keys for merging sufficient statistics with
            matching keyed objects.
        num_vals (int): Number of values in integer range for the set.

    """

    def __init__(self, num_vals: int, keys: Optional[str] = None) -> None:
        """Initialize an integer Bernoulli set accumulator factory.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with
                matching keyed objects.
            num_vals (int): Number of values in integer range for the set.

        """
        self.keys = keys
        self.num_vals = num_vals

    def make(
        self, device: Optional[tn.device] = None
    ) -> "IntegerBernoulliSetAccumulator":
        """Create an accumulator associated with ``device``."""
        return IntegerBernoulliSetAccumulator(
            self.num_vals, keys=self.keys, device=device
        )


class IntegerBernoulliSetEstimator(TorchParameterEstimator):
    """Estimate integer Bernoulli set distributions from sufficient stats.

    Attributes:
        num_vals (int): Number of values in integer range for the set.
        keys (Optional[str]): Keys for merging sufficient statistics with
            matching keyed objects.
        pseudo_count (Optional[float]): Re-weight suff stats in estimation.
        suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
        min_prob (float): Minimum probability for an integer in the range.

    """

    def __init__(
        self,
        num_vals: int,
        min_prob: float = 1.0e-128,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[np.ndarray] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize an integer Bernoulli set estimator.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in the range.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            keys (Optional[str]): Keys for merging sufficient statistics with
                matching keyed objects.

        """
        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_prob = min_prob

    def accumulator_factory(self) -> "IntegerBernoulliSetAccumulatorFactory":
        """Create a factory for the configured support size and key."""
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Optional[np.ndarray] = None,
        device: Optional[tn.device] = None,
    ) -> "IntegerBernoulliSetDistribution":
        """Estimate inclusion probabilities on ``device``."""
        assert suff_stat is not None
        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = self.suff_stat * self.pseudo_count
            p1 = np.subtract(1.0, self.suff_stat) * self.pseudo_count
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
    """Flatten a sequence of finite integer sets for torch operations."""

    def __str__(self) -> str:
        """Return the encoder name."""
        return "IntegerBernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an integer Bernoulli set encoder."""
        return isinstance(other, IntegerBernoulliSetDataEncoder)

    def seq_encode(
        self, x: Sequence[Sequence[int]], device: Optional[tn.device] = None
    ) -> "IntegerBernoulliSetTorchSequence":
        """Encode ``N`` sets as ``(N, observation_indices, values)``.

        Both flat integer tensors have shape ``(M,)`` and are created on
        ``device``; ``M`` is the total number of included integers.
        """
        idx: List[int] = []
        xs: List[int] = []
        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx_tensor = vec.int_tensor(idx, device=device)
        xs_tensor = vec.int_tensor(xs, device=device)

        return IntegerBernoulliSetTorchSequence(
            data=(len(x), idx_tensor, xs_tensor), device=device
        )


class IntegerBernoulliSetTorchSequence(TorchEncodedSequence):
    """Store a flattened integer-set encoding and observation count."""

    data: Tuple[int, tn.Tensor, tn.Tensor]

    def __init__(
        self, data: Tuple[int, tn.Tensor, tn.Tensor], device: Optional[tn.device] = None
    ) -> None:
        """Initialize the encoded tuple and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"IntegerBernoulliSetTorchSequence(device={repr(self.device)})"
