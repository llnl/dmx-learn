"""Create, estimate, and sample from a categorical distribution on integers.

Defines the IntegerCategoricalDistribution, IntegerCategoricalSampler,
IntegerCategoricalAccumulatorFactory, IntegerCategoricalAccumulator,
IntegerCategoricalEstimator, and the IntegerCategoricalDataEncoder classes for
use with pysparkplug.

Data type (int): The integer categorical distribution is defined through
summary statistics `min_val` (int) and a probability vector `p_vec`
(`np.ndarray[float]`) that sums to 1.0. The range of values is given by
`[min_val, min_val + len(p_vec) - 1]`. The density is then,

    P(x_mat=i) = p_vec[i]

for x in {min_val,min_val+1, ..., min_val + length(p_vec) - 1}, else 0.0.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import inf
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class IntegerCategoricalDistribution(TorchProbabilityDistribution):
    """IntegerCategoricalDistribution object defining an integer category range.

    Attributes:
        p_vec (tn.Tensor): Must sum to 1.0. First probability is probability
            for p_mat(x_mat=min_val).
        min_val (int): Minimum value in support of integer categorical
        max_val (int): Maximum value in support of integer categorical set
            to min_val + length(p_vec) - 1.
        log_p_vec (tn.Tensor): Log of p_vec.
        num_vals (int): Total number of values in support of
            IntegerCategoricalDistribution instance.

    """

    def __init__(
        self,
        min_val: int,
        p_vec: Union[List[float], np.ndarray],
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerCategoricalDistribution object.

        Args:
            min_val (int): Minimum value of the integer categorical support.
            p_vec (Union[List[float], np.ndarray]): Probability vector for each
                value in range.
            device (Optional[str]): Device on which to perform computations.

        """
        super().__init__(device)
        self.p_vec = vec.tensor(p_vec, device=self._device)
        self.min_val = min_val
        self.max_val = min_val + self.p_vec.shape[0] - 1
        self.log_p_vec = tn.log(self.p_vec)
        self.num_vals = self.p_vec.shape[0]

    def __repr__(self) -> str:
        s1 = str(self.min_val)
        s2 = ",".join([str(x) for x in self.p_vec.data.cpu().tolist()])

        return f"IntegerCategoricalDistribution(min_val={s1}, p_vec=[{s2}])"

    def to(self, device: tn.device):
        self.p_vec = self.p_vec.to(device)
        self.log_p_vec = tn.log(self.p_vec)
        self._device = device

    def density(self, x: int) -> float:
        """Evaluate the density of the integer categorical at observation x.

        Notes:
            p_mat(x_mat=x) = p_vec[x] if x in support [min_val, max_val], else 0.0.

        Args:
            x (int): Integer value.

        Returns:
            float: Density at x.

        """
        return (
            0.0
            if x < self.min_val or x > self.max_val
            else float(self.p_vec[x - self.min_val])
        )

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the integer categorical at observation x.

        Notes:
            log_p(x_mat=x) = log_p_vec[x] if x is in support
            [min_val, max_val], else -np.inf.

        Args:
            x (int): Integer value.

        Returns:
            float: Log-density at x.

        """
        return (
            -inf
            if (x < self.min_val or x > self.max_val)
            else float(self.log_p_vec[x - self.min_val])
        )

    def seq_log_density(self, x: "IntegerCategoricalTorchSequence") -> tn.Tensor:
        if not isinstance(x, IntegerCategoricalTorchSequence):
            raise TypeError(
                "Requires IntegerCategoricalTorchSequence for `seq_` function calls."
            )
        v = x.data - self.min_val
        u = tn.bitwise_and(v >= 0, v < self.num_vals)
        rv = vec.zeros(len(x.data), device=self.model_device())
        rv.fill_(-tn.inf)

        v_dev = v.to(device=self.log_p_vec.device)
        u_dev = u.to(device=self.model_device())
        rv[u_dev] = self.log_p_vec[v_dev[u_dev.to(device=v_dev.device)]]

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerCategoricalSampler":
        return IntegerCategoricalSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "IntegerCategoricalEstimator":
        if pseudo_count is None:
            est = IntegerCategoricalEstimator()
        else:
            est = IntegerCategoricalEstimator(
                pseudo_count=pseudo_count,
                suff_stat=(self.min_val, self.p_vec),
            )
        return est

    def dist_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalSampler(DistributionSampler):
    """Sample from IntegerCategoricalDistribution.

    Attributes:
        p_vec (np.ndarray): Numpy array of probs for each integer value.
        min_val (int): min val of range
        max_val (int): max val of range.
        rng (RandomState): RandomState object with seed set if passed.

    """

    def __init__(
        self, dist: "IntegerCategoricalDistribution", seed: Optional[int] = None
    ) -> None:
        """IntegerCategoricalSampler object.

        Args:
            dist (IntegerCategoricalDistribution): Distribution instance to
                sample from.
            seed (Optional[int]): Set the seed for random number generator used
                to sample.

        """
        self.p_vec = dist.p_vec.cpu().numpy()
        self.min_val = dist.min_val
        self.max_val = dist.max_val
        self.rng = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw iid samples from IntegerCategoricalSampler object.

        Note: If size is None, a single sample is returned as an integer.
        If size > 0, a List of integers with length equal to size is returned.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Integer or List[int] of iid samples from IntegerCategoricalSampler
                instance.

        """
        return (
            self.rng.choice(len(self.p_vec), size=size, replace=True, p=self.p_vec)
            + self.min_val
        )


class IntegerCategoricalAccumulator(TorchStatisticAccumulator):
    """Accumulate sufficient statistics from observed data.

    Notes:
        If min_val and max_val are not provided, they are obtained from the data
        in accumulation step.

    Attributes:
        min_val (Optional[TI]): Minimum value of integer categorical range.
        max_val (Optional[TI]): Maximum value of integer categorical range.
        count_vec (Optional[tn.Tensor]): Torch tensor of floats for tracking
            probability weights for each integer value in support. Set to
            None if min_val and max_val are both not None.
        key (Optional[str]): Key for merging sufficient statistics of integer
            IntegerCategoricalAccumulator objects.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerCategoricalAccumulator object.

        Args:
            min_val (Optional[TI]): Sets the minimum value of the integer
                categorical range.
            max_val (Optional[TI]): Sets the maximum value of the integer
                categorical range.
            keys (Optional[str]): Set key for merging sufficient statistics of
                integer IntegerCategoricalAccumulator objects.
            device (Optional[str]): Device on which to perform computations.

        """
        super().__init__(device)
        self.min_val = min_val
        self.max_val = max_val

        if min_val is not None and max_val is not None:
            self.count_vec = np.zeros(max_val - min_val + 1, dtype=np.float64)

        else:
            self.count_vec = None

        self.key = keys

    def seq_initialize(
        self,
        x: "IntegerCategoricalTorchSequence",
        weights: tn.Tensor,
        tng: tn.Generator,
    ) -> None:
        return self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "IntegerCategoricalTorchSequence",
        weights: tn.Tensor,
        estimate: Optional["IntegerCategoricalDistribution"],
    ) -> None:
        min_x = int(x.data.min())
        max_x = int(x.data.max())

        loc_cnt = tn.bincount(x.data - min_x, weights=weights).data.cpu().numpy()

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1, dtype=np.float64)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1, dtype=np.float64)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def combine(
        self, suff_stat: Tuple[Optional[int], Optional[np.ndarray]]
    ) -> "IntegerCategoricalAccumulator":
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(
                suff_stat[1]
            ):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = np.zeros(max_val - min_val + 1, dtype=np.float64)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> Tuple[int, np.ndarray]:
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> "IntegerCategoricalAccumulator":
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())

            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Return IntegerCategoricalDataEncoder object for encoding sequences of
        iid integer categorical observations."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Factory for IntegerCategoricalAccumulator objects.

    Attributes:
        min_val (Optional[int]): Minimum value of integer categorical, if None
            estimated from data.
        max_val (Optional[int]): Maximum value of integer categorical, if None
            estimated from data.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
    ) -> None:
        """IntegerCategoricalAccumulatorFactory object.

        Args:
            min_val (Optional[TI]): Set minimum value of integer categorical.
            max_val (Optional[TI]): Set maximum value of integer categorical.
            keys (Optional[str]): Set keys for merging statistics of
                IntegerCategoricalAccumulator objects.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "IntegerCategoricalAccumulator":
        return IntegerCategoricalAccumulator(
            min_val=self.min_val,
            max_val=self.max_val,
            keys=self.keys,
            device=device,
        )


class IntegerCategoricalEstimator(TorchParameterEstimator):
    """Estimate IntegerCategoricalDistribution from sufficient statistics.

    Attributes:
        min_val (Optional[TI]): Minimum value of integer categorical.
        max_val (Optional[TI]): Maximum value of integer categorical.
        pseudo_count (Optional[float]): Used to re-weight suff_stat when merged
            with new aggregated data.
        suff_stat: See above for details.
        keys (Optional[str]): Keys for accumulating merging statistics of
            IntegerCategoricalAccumulator objects.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[int, np.ndarray]] = None,
        keys: Optional[str] = None,
    ) -> None:
        """IntegerCategoricalEstimator object.

        Args:
            min_val (Optional[TI]): Set minimum value of integer categorical.
            max_val (Optional[TI]): Set maximum value of integer categorical.
            pseudo_count (Optional[float]): Used to re-weight suff_stat member
                variables in merging of sufficient statistics.
            suff_stat: Set sufficient statistics. See above for details.
            keys (Optional[str]): Set keys for accumulating merging statistics
                of IntegerCategoricalAccumulator objects.

        """
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> "IntegerCategoricalAccumulatorFactory":
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        return IntegerCategoricalAccumulatorFactory(
            min_val=min_val, max_val=max_val, keys=self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Optional[Tuple[int, np.ndarray]],
        device: Optional[tn.device] = None,
    ) -> "IntegerCategoricalDistribution":
        if self.pseudo_count is not None and self.suff_stat is None:

            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count
            min_val = suff_stat[0]
            p_vec = (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs

        elif (
            self.pseudo_count is not None
            and self.min_val is not None
            and self.max_val is not None
        ):

            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = np.zeros(max_val - min_val + 1, dtype=np.float64)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            p_vec = (count_vec + pseudo_count_per_level) / adjusted_nobs

        elif self.pseudo_count is not None and self.suff_stat is not None:

            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = np.zeros(max_val - min_val + 1, dtype=np.float64)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            p_vec = count_vec / (count_vec.sum())

        else:
            min_val = suff_stat[0]
            p_vec = suff_stat[1] / np.sum(suff_stat[1])

        return IntegerCategoricalDistribution(
            min_val=min_val, p_vec=p_vec, device=device
        )


class IntegerCategoricalDataEncoder(TorchSequenceEncoder):
    """Encode sequences of iid integer categorical observations."""

    def __str__(self) -> str:
        return "IntegerCategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(
        self, x: Union[List[int], np.ndarray], device: Optional[tn.device] = None
    ) -> "IntegerCategoricalTorchSequence":
        return IntegerCategoricalTorchSequence(
            data=vec.int_tensor(x, device=device), device=device
        )


class IntegerCategoricalTorchSequence(TorchEncodedSequence):

    def __init__(self, data: tn.tensor, device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"IntegerCategoricalTorchSequence(device={repr(self.device)})"
