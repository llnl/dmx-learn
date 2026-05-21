"""Evaluate, estimate, and sample from an integer multinomial distribution.

Defines the IntegerMultinomialDistribution, IntegerMultinomialSampler,
IntegerMultinomialAccumulatorFactory, IntegerMultinomialAccumulator,
IntegerMultinomialEstimator, and the IntegerMultinomialDataEncoder classes for
use with pysparkplug.

Data type: Sequence[Tuple[int, float]]: Consider an observation of a
multinomial consisting of integer-category counts of the form
`x = (x_0,..,x_K)`, where K is the number of integers in the range
`[min_val, max_val]`. The IntegerMultinomialDistribution with support
`[min_val, max_value]`, number of trials `N`, and success probabilities
`p = (p_0, ..., p_k)` is given by

    log(P(x,N|p)) = -log(n!)
        - sum_{k=1}^{K} (x_k * log(p_k) + log(x_k!))
        + log(P_len(N))

where P_len(N) is a distribution for the number of trials in the multinomial.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import inf, maxrandint
from dmx.torch_stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

SS = TypeVar("SS")
D = Sequence[Tuple[int, float]]
E0 = TypeVar("E0")
E = Tuple[int, tn.Tensor, tn.Tensor, tn.Tensor, Optional[E0]]


class IntegerMultinomialDistribution(TorchProbabilityDistribution):
    """Declare a multinomial distribution on a set of integers.

    Attributes:
        p_vec (tn.tensor): Probability of each integer category for a trial.
        min_val (int): Smallest integer value for category range. Defaults to 0.
        max_val (int): Largest value of category range. Set by
            `min_val + len(p_vec) - 1`.
        log_p_vec (tn.tensor): Log of p_vec member instance.
        num_vals (int): Total number of integer valued categories.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution for
            number of trials. Set to NullDistribution if None.
        keys (Optional[str]): Keys for distribution passed when
            ParameterEstimator is created.

    """

    def __init__(
        self,
        min_val: int = 0,
        p_vec: List[float] = None,
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Create IntegerMultinomialDistribution object.

        Args:
            min_val (int): Set the minimum value on range of values.
            p_vec (Union[List[float],np.ndarray): Probabilities for values.
                Length determines number of categories.
            len_dist (Optional[TorchProbabilityDistribution]): Optional length
                distribution for the number of trials.
            keys (Optional[str]): Set key for distribution.

        """
        super().__init__(device)
        self.p_vec = vec.tensor(p_vec, device=self._device)
        self.min_val = min_val
        self.max_val = min_val + self.p_vec.shape[0] - 1
        self.log_p_vec = tn.log(self.p_vec)
        self.num_vals = self.p_vec.shape[0]
        self.len_dist = (
            len_dist if len_dist is not None else NullDistribution(device=self._device)
        )
        self.keys = keys

    def to(self, device: tn.device) -> None:
        self.p_vec = self.p_vec.to(device)
        self.log_p_vec = tn.log(self.p_vec)
        self.len_dist.to(device)
        self._device = device

    def __repr__(self) -> str:
        s1 = repr(self.min_val)
        s2 = repr(self.p_vec.data.cpu().tolist())
        s3 = repr(self.len_dist)

        return f"IntegerMultinomialDistribution({s1}, {s2}, len_dist={s3})"

    def density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the density of IntegerMultinomialDistribution at observed value x.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of tuples containing the
                integer category value and number of successes.

        Returns:
            float: Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the log-density at observed value x.

        Notes:
            Log-density given by,

            log(p_mat(x)) = log(n!) - sum_k x_k*log(p_k) - log(x_k!), for x
            having k integer categories.

            n is the total number of trials in observation x and x has k
            integer values. p_k denotes the probability of success for
            integer-category x_k.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of tuples containing the
                integer category value and number of successes.

        Returns:
            float: Log-density at x.

        """
        rv = 0.0
        xc = 0
        for xx, cnt in x:
            rv += (
                -inf
                if (xx < self.min_val or xx > self.max_val)
                else self.log_p_vec[xx - self.min_val]
            ) * cnt
            xc += cnt

        rv += self.len_dist.log_density(xc)

        return float(rv)

    def seq_log_density(self, x: "IntegerMultinomialTorchSequence") -> tn.Tensor:
        if not isinstance(x, IntegerMultinomialTorchSequence):
            raise TypeError(
                "Requires IntegerMultinomialTorchSequence for `seq_` function calls."
            )
        sz, idx, cnt, val, tcnt = x.data

        val_dev = val.to(device=self.log_p_vec.device)
        idx_dev = idx.to(device=self.model_device())
        cnt_dev = cnt.to(device=self.model_device())
        v = val_dev - self.min_val
        u = tn.bitwise_and(v >= 0, v < self.num_vals)
        rv = vec.zeros(len(v), device=self.model_device())
        rv.fill_(-tn.inf)
        rv[u] = self.log_p_vec[v[u]]
        rv[u] *= cnt_dev[u.to(device=cnt_dev.device)]
        ll = tn.bincount(
            idx_dev, weights=rv.to(device=self.model_device()), minlength=sz
        )

        if tcnt is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll

    def sampler(self, seed: Optional[int] = None) -> "IntegerMultinomialSampler":
        if isinstance(self.len_dist, NullDistribution):
            raise ValueError(
                "IntegerMultinomialDistribution must have len_dist set to "
                "distribution with support on "
                "non-negative integers."
            )
        return IntegerMultinomialSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[int] = None
    ) -> "IntegerMultinomialEstimator":
        len_est = (
            NullEstimator()
            if self.len_dist is None
            else self.len_dist.estimator(pseudo_count=pseudo_count)
        )

        if pseudo_count is None:
            return IntegerMultinomialEstimator(len_estimator=len_est)

        return IntegerMultinomialEstimator(
            min_val=self.min_val,
            max_val=self.max_val,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            suff_stat=(self.min_val, self.p_vec),
        )

    def dist_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialSampler(DistributionSampler):
    """Sample from IntegerMultinomialDistribution.

    Attributes:
        p_vec (np.ndarray): Probability for each value.
        min_val (int): Min value of multinomial
        max_val (int): Max value of multinomial
        rng (RandomState): RandomState set with seed if passed.
        len_sampler (DistributionSampler): DistributionSampler for the number
            of trials.

    """

    def __init__(
        self, dist: IntegerMultinomialDistribution, seed: Optional[int] = None
    ) -> None:
        """IntegerMultinomialSampler object.

        Args:
            dist (IntegerMultinomialDistribution): Distribution instance to
                sample from.
            seed (Optional[int]): Optional seed for random number generator.

        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self.p_vec = dist.p_vec.data.cpu().numpy()
        self.min_val = dist.min_val
        self.max_val = dist.max_val
        self.len_sampler = self.dist.len_dist.sampler(
            seed=self.rng.randint(0, maxrandint)
        )

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """Draw independent samples from an integer multinomial distribution.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            List of length `size` containing `List[Tuple[int, float]]`. If
            size is None, returns one sample `List[Tuple[int, float]]`.

        """
        if size is None:
            cnt = self.len_sampler.sample()
            entry = self.rng.multinomial(cnt, self.p_vec)
            rrv = []
            for j in np.flatnonzero(entry):
                rrv.append((j + self.min_val, entry[j]))
            return rrv

        cnt = self.len_sampler.sample(size=size)
        rv = []

        for i in range(size):
            rrv = []
            entry = self.rng.multinomial(cnt[i], self.p_vec)
            for j in np.flatnonzero(entry):
                rrv.append((j + self.min_val, entry[j]))
            rv.append(rrv)
        return rv


class IntegerMultinomialAccumulator(TorchStatisticAccumulator):
    """Accumulate sufficient statistics from observed data.

    Attributes:
        min_val (Optional[int]): Minimum value for integer multinomial.
        max_val (Optional[int]): Maximum value for integer multinomial.
        len_accumulator (Optional[TorchStatisticAccumulator]): Optional
            accumulator for integer multinomial trial counts. Set to
            NullAccumulator() if None.
        count_vec (Optional[ndarray]): Counter for the number of values in the
            integer multinomial range. Initialized if `min_val` and `max_val`
            are passed; otherwise set to None.
        key (Optional[str]): Keys for merging sufficient stats with other
            objects containing a matching key.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
        len_accumulator: Optional[TorchStatisticAccumulator] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerMultinomialAccumulator object.

        Args:
            min_val (Optional[int]): Set minimum value for integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            keys (Optional[str]): Set keys for merging sufficient stats with
                other objects containing matching keys.
            len_accumulator (Optional[TorchStatisticAccumulator]): Optional
                accumulator for integer multinomial trial counts.
            device: Device for tensor calculations

        """
        super().__init__(device=device)
        self.min_val = min_val
        self.max_val = max_val
        self.len_accumulator = (
            len_accumulator
            if len_accumulator is not None
            else NullAccumulator(device=self._device)
        )

        if min_val is not None and max_val is not None:
            self.count_vec = np.zeros(max_val - min_val + 1, dtype=np.float64)
        else:
            self.count_vec = None

        self.key = keys

    def seq_update(
        self,
        x: "IntegerMultinomialTorchSequence",
        weights: tn.Tensor,
        estimate: Optional[IntegerMultinomialDistribution],
    ) -> None:
        _, idx, cnt, val, tenc = x.data

        min_x = int(val.min())
        max_x = int(val.max())

        loc_cnt = (
            tn.bincount(val - min_x, weights=cnt * weights[idx]).data.cpu().numpy()
        )

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

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.seq_update(tenc, weights, None)
            else:
                self.len_accumulator.seq_update(tenc, weights, estimate.len_dist)

    def seq_initialize(
        self,
        x: "IntegerMultinomialTorchSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        del tng
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[int, np.ndarray, Optional[SS]]
    ) -> "IntegerMultinomialAccumulator":
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

                count_vec = np.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[int, np.ndarray, Optional[Any]]:
        return self.min_val, self.count_vec, self.len_accumulator.value()

    def from_value(
        self, x: Tuple[int, np.ndarray, Optional[SS]]
    ) -> "IntegerMultinomialAccumulator":
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        len_encoder = self.len_accumulator.acc_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Factory for IntegerMultinomialAccumulator objects.

    Attributes:
        min_val (Optional[int]): Optional minimum value for
            IntegerMultinomialAccumulator.
        max_val (Optional[int]): Optional maximum value for
            IntegerMultinomialAccumulator.
        keys (Optional[str]): Optional keys for merging sufficient statistics
            of the object instance.
        len_factory (Optional[StatisticAccumulatorFactory]): Optional factory
            for the number-of-trials accumulator. Defaults to
            NullAccumulatorFactory().

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        keys: Optional[str] = None,
        len_factory: Optional[
            TorchStatisticAccumulatorFactory
        ] = NullAccumulatorFactory(),
    ) -> None:
        """IntegerMultinomialAccumulatorFactory object.

        Args:
            min_val (Optional[int]): Optional minimum value for
                IntegerMultinomialAccumulator.
            max_val (Optional[int]): Optional maximum value for
                IntegerMultinomialAccumulator.
            keys (Optional[str]): Optional keys for merging sufficient
                statistics of the object instance.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional
                factory for creating the number-of-trials accumulator.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "IntegerMultinomialAccumulator":
        len_acc = self.len_factory.make(device=device)
        return IntegerMultinomialAccumulator(
            min_val=self.min_val,
            max_val=self.max_val,
            keys=self.keys,
            len_accumulator=len_acc,
            device=device,
        )


class IntegerMultinomialEstimator(TorchParameterEstimator):
    """Estimate integer multinomial distributions from aggregated data.

    Attributes:
        min_val (Optional[int]): Set minimum value integer multinomial.
        max_val (Optional[int]): Set maximum value for integer multinomial.
        len_estimator (TorchParameterEstimator): ParameterEstimator for number
            of trials, default `NullEstimator()`.
        len_dist (Optional[TorchProbabilityDistribution]): Optional
            TorchProbabilityDistribution for fixing distribution on number of trials.
        name (Optional[str]): Set name for object instance.
        pseudo_count (Optional[float]): Used to re-weight sufficient statistics
            if suff_stat is passed.
        suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and
            counts for categories. If `min_val` and `max_val` are both not
            None, this is ignored in estimation.
        keys (Optional[str]): Set key for merging sufficient statistics of
            objects with matching keys.

    """

    def __init__(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        len_dist: Optional[TorchProbabilityDistribution] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[int, np.ndarray]] = None,
        keys: Optional[str] = None,
    ) -> None:
        """IntegerMultinomialEstimator object.

        Args:
            min_val (Optional[int]): Set minimum value integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            len_estimator (Optional[ParameterEstimator]): Optional
                ParameterEstimator for number of trials.
            len_dist (Optional[TorchProbabilityDistribution]): Optional
                TorchProbabilityDistribution for fixing the distribution on the
                number of trials.
            pseudo_count (Optional[float]): Used to re-weight sufficient
                statistics if suff_stat is passed.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and
                counts for categories.
            keys (Optional[str]): Set key for merging sufficient statistics of
                objects with matching keys.

        """
        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.len_dist = len_dist
        self.keys = keys

    def accumulator_factory(self) -> "IntegerMultinomialAccumulatorFactory":
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        len_factory = self.len_estimator.accumulator_factory()
        return IntegerMultinomialAccumulatorFactory(
            min_val=min_val,
            max_val=max_val,
            keys=self.keys,
            len_factory=len_factory,
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[int, np.ndarray, Optional[SS]],
        device: Optional[tn.device] = None,
    ) -> "IntegerMultinomialDistribution":

        len_dist = (
            self.len_dist.to(device)
            if self.len_dist is not None
            else self.len_estimator.estimate(nobs, suff_stat[2], device=device)
        )

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(
                suff_stat[0],
                (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs,
                len_dist=len_dist,
                keys=self.keys,
                device=device,
            )

        if (
            self.pseudo_count is not None
            and self.min_val is not None
            and self.max_val is not None
        ):
            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(
                min_val,
                (count_vec + pseudo_count_per_level) / adjusted_nobs,
                len_dist=len_dist,
                keys=self.keys,
                device=device,
            )

        if self.pseudo_count is not None and self.suff_stat is not None:
            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            return IntegerMultinomialDistribution(
                min_val,
                count_vec / (count_vec.sum()),
                len_dist=len_dist,
                keys=self.keys,
                device=device,
            )
        return IntegerMultinomialDistribution(
            suff_stat[0],
            suff_stat[1] / (suff_stat[1].sum()),
            len_dist=len_dist,
            keys=self.keys,
            device=device,
        )


class IntegerMultinomialDataEncoder(TorchSequenceEncoder):
    """Encode sequences of iid integer multinomial observations.

    Attributes:
        len_encoder (TorchSequenceEncoder): TorchSequenceEncoder for encoding
            the number of trials in each iid integer multinomial observation.
            Defaults to `NullDataEncoder()` if None is passed.

    """

    def __init__(
        self, len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder()
    ) -> None:
        """IntegerMultinomialDataEncoder object.

        Args:
            len_encoder (Optional[TorchSequenceEncoder]): Optional sequence
                encoder for the number of trials in each iid integer
                multinomial observation.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        return (
            "IntegerMultinomialDataEncoder(len_encoder=" + str(self.len_encoder) + ")"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerMultinomialDataEncoder):
            return self.len_encoder == other.len_encoder

        return False

    def seq_encode(
        self,
        x: Sequence[Sequence[Tuple[int, float]]],
        device: Optional[tn.device] = None,
    ) -> "IntegerMultinomialTorchSequence":
        idx = []
        cnt = []
        val = []
        tcnt = []

        for i, y in enumerate(x):
            cc = 0
            for z in y:
                idx.append(i)
                cnt.append(z[1])
                val.append(z[0])
                cc += z[1]
            tcnt.append(cc)

        sz = len(x)
        idx = vec.int_tensor(idx, device=device)
        cnt = vec.tensor(cnt, device=device)
        val = vec.int_tensor(val, device=device)

        tcnt = self.len_encoder.seq_encode(tcnt, device=device)

        return IntegerMultinomialTorchSequence(
            data=(sz, idx, cnt, val, tcnt), device=device
        )


class IntegerMultinomialTorchSequence(TorchEncodedSequence):

    def __init__(
        self,
        data: Tuple[int, tn.tensor, tn.tensor, tn.tensor, TorchEncodedSequence],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"IntegerMultinomialTorchSequence(device={repr(self.device)})"
