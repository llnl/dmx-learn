"""
Create, estimate, and sample from a heterogeneous mixture distribution.

Defines the HeterogeneousMixtureDistribution, HeterogeneousMixtureSampler,
HeterogeneousMixtureAccumulatorFactory,
HeterogeneousMixtureAccumulator, HeterogeneousMixtureEstimator, and the
HeterogeneousMixtureDataEncoder classes for use
with pysparkplug.

HeterogeneousMixtureDistribution with data type T, is defined by the density of the
form,

p_mat(Y) = sum_{k=1}^{K} p_mat(Y|Z=k)*p_mat(Z=k),

where p_mat(Z=k) is a mixture weight, and p_mat(Y|Z=k) is defined as a the k^{th}
component distribution. Note that
the component distributions p_mat(Y|Z=k) must only be compatible in data type T.

Example: A heterogeneous mixture with weights [0.5, 0.5] and component distribution
Exponential(beta) and Gamma(k,theta),
has form
    p_mat(x_mat) = 0.5*P_0(x; beta) + 0.5*P_1(x; k, theta), for x > 0.0,
where
    P_0(x;beta) is an exponential density and P_1(x; k, theta) is a Gamma density.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

T = TypeVar("T")  # Type of Mixture component data.
T1 = TypeVar("T1")  # Type of encoded data.
T2 = TypeVar("T2")  # Type of component suff_stat

key_type = Union[Tuple[str, str], Tuple[None, None]]


def _sample_dirichlet_like(alpha: tn.Tensor, size: int, tng: tn.Generator) -> tn.Tensor:
    return vec.sample_dirichlet(alpha=alpha, size=size, tng=tng)


class HeterogeneousMixtureDistribution(TorchProbabilityDistribution):
    """
    HeterogeneousMixtureDistribution object for defining a mixture with heterogeneous
    components.

        Attributes:
            w (tn.tensor): Weights for the mixture.
            zw (tn.tensor): Boolean tensor, true if a weight is 0.
            log_w (tn.tensor): Log of the mixture weights.
            components (Sequence[TorchProbabilityDistribution]): Mixture components.
            num_components (int): Number of mixture components.

    """

    def __init__(
        self,
        components: Sequence[TorchProbabilityDistribution],
        w: Union[np.ndarray, List[float], tn.Tensor],
        device: Optional[tn.device] = None,
    ) -> None:
        """
        HeterogeneousMixtureDistribution object.

                Args:
                    components (Sequence[TorchProbabilityDistribution]):
                    Mixture components.
                    w (tn.tensor): Weights for the mixture.
                    device (Optional[tn.device]): Device to declare model on.

        """

        super().__init__(device)

        self.w = vec.tensor(w, device=self._device)
        self.zw = self.w == 0.0
        self.log_w = tn.log(self.w + self.zw)
        self.log_w[self.zw] = -tn.inf
        self.components = components
        self.num_components = len(components)

    def to(self, device: tn.device) -> None:
        self._device = device
        self.w = self.w.to(device)
        self.zw = self.w == 0.0
        self.log_w = tn.log(self.w + self.zw)
        self.log_w[self.zw] = -tn.inf

        for comp in self.components:
            comp.to(device)

    def __repr__(self) -> str:
        s1 = ",".join([str(u) for u in self.components])
        s2 = repr(self.w.data.cpu().tolist())

        return f"HeterogeneousMixtureDistribution([{s1}], {s2})"

    def density(self, x: T) -> float:
        """
        Evaluate density of Heterogeneous Mixture distribution at observation x.

                Args:
                    x: (T): Single observation from mixture distribution. T is
                    data type of components.

                Returns:
                    float: Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """
        Evaluate log-density of heterogeneous mixture distribution at observation x.

                Args:
                    x: (T): Single observation from heterogeneous mixture
                    distribution. T is data type of components.

                Returns:
                    float: Log-density at x.

        """
        rv = tn.logsumexp(
            vec.tensor([u.log_density(x) for u in self.components], device=self._device)
            + self.log_w,
            dim=0,
        )
        return float(rv)

    def component_log_density(self, x: T) -> tn.Tensor:
        return vec.tensor(
            [m.log_density(x) for m in self.components], device=self._device
        )

    def posterior(self, x: T) -> tn.Tensor:

        comp_log_density = vec.tensor(
            [m.log_density(x) for m in self.components], device=self._device
        )
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -tn.inf

        rv = tn.logsumexp(comp_log_density, dim=0)
        if tn.isinf(rv):
            return self.w
        comp_log_density -= rv
        tn.exp(comp_log_density, out=comp_log_density)
        return comp_log_density

    def seq_component_log_density(
        self, x: "HeterogeneousMixtureTorchSequence"
    ) -> tn.Tensor:
        """
        Vectorized evaluation of component-wise log-density for encoded sequence x.

                Args:
                    x (HeterogeneousMixtureTorchSequence): TorchEncodedSequence for
                    HeterogeneousMixture.

                Returns:
                    tn.tensor: log-density of mixture components evaluated at
                    each observation.

        """
        if not isinstance(x, HeterogeneousMixtureTorchSequence):
            raise TypeError(
                "Requires HeterogeneousMixtureTorchSequence for `seq_` function calls."
            )

        tag_list, enc_data = x.data
        ll_mat: Optional[tn.Tensor] = None
        device = tn.device(self._device)

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if ll_mat is None:
                        ll_mat = vec.zeros(
                            (len(temp), self.num_components), device=device
                        )
                        ll_mat += -np.inf
                    ll_mat[:, i] = temp

        if ll_mat is None:
            return vec.zeros((0, self.num_components), device=device)

        return ll_mat

    def seq_log_density(self, x: "HeterogeneousMixtureTorchSequence") -> tn.Tensor:
        if not isinstance(x, HeterogeneousMixtureTorchSequence):
            raise TypeError(
                "Requires HeterogeneousMixtureTorchSequence for `seq_` function calls."
            )

        tag_list, enc_data = x.data
        ll_mat: Optional[tn.Tensor] = None

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if ll_mat is None:
                        ll_mat = vec.zeros(
                            (len(temp), self.num_components), device=self.model_device()
                        )
                        ll_mat += -np.inf
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += self.log_w[i]

        if ll_mat is None:
            return vec.zeros(0, device=self.model_device())

        assert ll_mat is not None

        ll_max, _ = tn.max(ll_mat, dim=1, keepdim=True)
        good_rows = tn.isfinite(ll_max.flatten())

        if tn.all(good_rows):
            ll_mat -= ll_max
            tn.exp(ll_mat, out=ll_mat)
            ll_sum = tn.sum(ll_mat, dim=1, keepdim=True)
            tn.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        ll_mat = ll_mat[good_rows, :]
        ll_max = ll_max[good_rows]
        ll_mat -= ll_max
        tn.exp(ll_mat, out=ll_mat)

        ll_sum = tn.sum(ll_mat, dim=1, keepdim=True)
        tn.log(ll_sum, out=ll_sum)
        ll_sum += ll_max

        rv = vec.zeros(good_rows.shape, device=self._device)
        rv[good_rows] = ll_sum.flatten()
        rv[~good_rows] = -tn.inf

        return rv

    def seq_posterior(self, x: "HeterogeneousMixtureTorchSequence") -> tn.Tensor:
        """
        Vectorized evaluation of posterior of
        HeterogeneousMixtureDistribution for encoded sequence x.

                Args:
                    x (HeterogeneousMixtureTorchSequence): TorchEncodedSequence for
                    HeterogeneousMixture.

                Returns:
                    tn.tensor: Tensor containing the posterior of each
                    observation in encoded sequence.

        """
        if not isinstance(x, HeterogeneousMixtureTorchSequence):
            raise TypeError(
                "Requires HeterogeneousMixtureTorchSequence for `seq_` function calls."
            )

        tag_list, enc_data = x.data
        ll_mat: Optional[tn.Tensor] = None

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if ll_mat is None:
                        ll_mat = vec.zeros(
                            (len(temp), self.num_components), device=self.model_device()
                        )
                        ll_mat += -tn.inf
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += self.log_w[i]

        if ll_mat is None:
            return vec.zeros((0, self.num_components), device=self.model_device())

        assert ll_mat is not None

        ll_max, _ = ll_mat.max(dim=1, keepdim=True)
        bad_rows = tn.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows] = tn.max(self.log_w)
        ll_mat -= ll_max

        tn.exp(ll_mat, out=ll_mat)
        tn.sum(ll_mat, dim=1, keepdim=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> "HeterogeneousMixtureSampler":

        return HeterogeneousMixtureSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "HeterogeneousMixtureEstimator":
        if pseudo_count is not None:
            return HeterogeneousMixtureEstimator(
                [
                    u.estimator(pseudo_count=1.0 / self.num_components)
                    for u in self.components
                ],
                pseudo_count=pseudo_count,
            )
        return HeterogeneousMixtureEstimator([u.estimator() for u in self.components])

    def dist_to_encoder(self) -> "HeterogeneousMixtureDataEncoder":
        encoders = [comp.dist_to_encoder() for comp in self.components]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureSampler(DistributionSampler):
    """
    HeterogeneousMixtureSampler used to generate samples from instance of
    HeterogeneousMixtureDistribution.

        Attributes:
            rng (RandomState): Seeded RandomState for sampling.
            w (np.ndarray): Weights for the mixture components.
            ncomps (int): Number of mixture components.
            comp_samplers (List[DistributionSamplers]): List of
            DistributionSampler objects for each mixture component.

    """

    def __init__(
        self, dist: HeterogeneousMixtureDistribution, seed: Optional[int] = None
    ):
        """
        HeterogeneousMixtureSampler object.

                Args:
                    dist (HeterogeneousMixtureDistribution): Assign
                    HeterogeneousMixtureDistribution to draw samples from.
                    seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.w = dist.w.data.cpu().numpy()
        self.ncomps = len(self.w)
        self.comp_samplers = [
            d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in dist.components
        ]

    def sample(self, size: Optional[int] = None) -> Union[Any, List[Any]]:
        """
        Draw iid samples from a heterogeneous mixture distribution.

                The data type drawn from 'comp_samplers' is type T,
                corresponding to the data type of the mixture components.

                If size is None, a single sample (of data type T) is drawn
                and returned. If size is not None, 'size'-iid
                heterogeneous mixture samples are drawn and returned as a
                List with data type List[T].

                Args:
                    size (Optional[int]): Number of iid samples to draw.

                Returns:
                    Data type T or List[T].

        """
        comp_state = self.rng.choice(
            range(0, self.ncomps), size=size, replace=True, p=self.w
        )

        if size is None:
            return self.comp_samplers[comp_state].sample()
        return [self.comp_samplers[i].sample() for i in comp_state]


class HeterogeneousMixtureAccumulator(TorchStatisticAccumulator):

    def __init__(
        self,
        accumulators: Sequence[TorchStatisticAccumulator],
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        device: Optional[tn.device] = None,
    ):
        super().__init__()
        self._device = tn.device("cpu") if device is None else device
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

        self.comp_counts = np.zeros(self.num_components, dtype=np.float64)

    def seq_update(
        self,
        x: "HeterogeneousMixtureTorchSequence",
        weights: tn.Tensor,
        estimate: "HeterogeneousMixtureDistribution",
    ) -> None:
        tag_list, enc_data = x.data
        ll_mat: Optional[tn.Tensor] = None
        device = tn.device(self._device)

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not estimate.zw[i]:
                    temp = estimate.components[i].seq_log_density(enc_data[tag])
                    if ll_mat is None:
                        ll_mat = vec.zeros(
                            (len(temp), self.num_components), device=device
                        )
                        ll_mat += -tn.inf
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += estimate.log_w[i]

        if ll_mat is None:
            return

        assert ll_mat is not None

        ll_max, _ = tn.max(ll_mat, dim=1, keepdim=True)

        bad_rows = tn.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w

        if tn.any(bad_rows):
            ll_max[bad_rows] = tn.max(estimate.log_w)

        ll_mat -= ll_max
        tn.exp(ll_mat, out=ll_mat)
        tn.sum(ll_mat, dim=1, keepdim=True, out=ll_max)
        tn.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                w_loc = ll_mat[:, i]
                self.comp_counts[i] += float(w_loc.sum())
                self.accumulators[i].seq_update(
                    enc_data[tag], w_loc, estimate.components[i]
                )

    def seq_initialize(
        self,
        x: "HeterogeneousMixtureTorchSequence",
        weights: tn.Tensor,
        tng: tn.Generator,
    ) -> None:
        tag_list, enc_data = x.data
        device = tn.device(self._device)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = tn.count_nonzero(keep_idx)
        ww = vec.zeros((sz, self.num_components), device=device)

        if keep_len > 0:
            alpha = (
                vec.ones(self.num_components, device=device) / self.num_components**2
            )
            ww[keep_idx, :] += _sample_dirichlet_like(
                alpha=alpha, size=int(keep_len), tng=tng
            )

        ww *= tn.reshape(weights, (sz, 1))

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                self.accumulators[i].seq_initialize(enc_data[tag], ww[:, i], tng)
                self.comp_counts[i] += float(tn.sum(ww[:, i]))

    def combine(
        self, suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]
    ) -> "HeterogeneousMixtureAccumulator":
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        return self.comp_counts, tuple(u.value() for u in self.accumulators)

    def from_value(
        self, x: Tuple[np.ndarray, Tuple[Any, ...]]
    ) -> "HeterogeneousMixtureAccumulator":
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i, acc_item in enumerate(acc):
                    acc[i] = acc_item.combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HeterogeneousMixtureDataEncoder":
        encoders = [comp.acc_to_encoder() for comp in self.accumulators]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """
    HeterogeneousMixtureAccumulatorFactory object for creating
    HeterogeneousMixtureAccumulator objects.

        Attributes:
            factories (Sequence[TorchStatisticAccumulatorFactory]): Factories for the
            mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.

    """

    def __init__(
        self,
        factories: Sequence[TorchStatisticAccumulatorFactory],
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
    ):
        """
        HeterogeneousMixtureAccumulatorFactory object.

                Args:
                    factories (Sequence[TorchStatisticAccumulatorFactory]):
                    Factories for the mixture components.
                    keys (Tuple[Optional[str], Optional[str]]): Keys for
                    weights and components.

        """
        self.factories = factories
        self.keys = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "HeterogeneousMixtureAccumulator":
        if device is not None:
            factories = [factory.make(device=device) for factory in self.factories]
            return HeterogeneousMixtureAccumulator(
                factories, keys=self.keys, device=device
            )

        factories = [factory.make(device=device) for factory in self.factories]
        return HeterogeneousMixtureAccumulator(factories, keys=self.keys)


class HeterogeneousMixtureEstimator(TorchParameterEstimator):
    """
    HeterogeneousMixtureEstimator object used to estimate
    HeterogeneousMixtureDistribution from aggregated sufficient statistics.

        Attributes:
            estimators (Sequence[TorchParameterEstimator]): Sequence of
            TorchParameterEstimator objects for the mixture
                components.
            fixed_weights (Optional[np.ndarray]): Treat mixture weights as fixed values.
            Must sum to 1.0.
            suff_stat (Optional[np.ndarray]): Weights of the mixture. Must sum to 1.0.
            pseudo_count (Optional[float]): Used to re-weight the member
            variable sufficient statistics in estimation.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the weights
            and component distributions.

    """

    def __init__(
        self,
        estimators: Sequence[TorchParameterEstimator],
        fixed_weights: Optional[Union[List[float], tn.Tensor]] = None,
        suff_stat: Optional[np.ndarray] = None,
        pseudo_count: Optional[float] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
    ) -> None:
        """
        HeterogeneousMixtureEstimator object.

                Args:
                    estimators (Sequence[TorchParameterEstimator]): Sequence of
                    TorchParameterEstimator objects for the mixture
                        components.
                    fixed_weights (Optional[Union[List[float], np.ndarray]]):
                    Set fixed values for mixture weights.
                    suff_stat (Optional[np.ndarray]): Numpy array of floats
                    with length equal to length of estimators.
                    pseudo_count (Optional[float]): Used to re-weight the
                    member variable sufficient statistics in estimation.
                    keys (Tuple[Optional[str], Optional[str]]): Set keys for
                    the weights and component distributions.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

        self.fixed_weights = (
            np.asarray(fixed_weights) if fixed_weights is not None else None
        )

    def accumulator_factory(self) -> "HeterogeneousMixtureAccumulatorFactory":
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return HeterogeneousMixtureAccumulatorFactory(est_factories, keys=self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, Tuple[Any, ...]],
        device: Optional[tn.device] = None,
    ) -> "HeterogeneousMixtureDistribution":

        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [
            self.estimators[i].estimate(counts[i], comp_suff_stats[i], device=device)
            for i in range(num_components)
        ]

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (
                counts.sum() + self.pseudo_count
            )

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return HeterogeneousMixtureDistribution(components, w, device=device)


class HeterogeneousMixtureDataEncoder(TorchSequenceEncoder):
    """
    HeterogeneousMixtureDataEncoder used for sequence encoding data for use
    with vectorized 'seq_' functions.

        Attributes:
            encoder_dict (Dict[DataSequenceEncoder, List[int]]): Dictionary of distinct
            DataSequenceEncoder objects
                found in encoders list. Value of encoder_dict is a list of ids for the
                components that are encoded by
                'encoder_dict key.

    """

    def __init__(self, encoders: List[TorchSequenceEncoder]) -> None:
        """
        HeterogeneousMixtureDataEncoder object.

                Args:
                    encoders (List[DataSequenceEncoder]): List of
                    DataSequenceEncoder objects for each heterogeneous mixture
                        component.

        """
        encoder_dict: Dict[str, TorchSequenceEncoder] = {}
        idx_dict: Dict[str, List[int]] = {}

        for encoder_idx, encoder in enumerate(encoders):
            enc_str = str(encoder)
            if enc_str not in encoder_dict:
                encoder_dict[enc_str] = encoder
                idx_dict[enc_str] = []
            idx_dict[enc_str].append(encoder_idx)

        self.encoder_dict: Dict[str, TorchSequenceEncoder] = encoder_dict
        self.idx_dict: Dict[str, List[int]] = idx_dict

    def __str__(self) -> str:
        s = "HeterogeneousMixtureDataEncoder(["
        item_list = list(self.idx_dict.items())
        for enc_str, comp_list in item_list[:-1]:
            s += enc_str + ",comps=" + str(comp_list) + ","

        s += item_list[-1][0] + ",comps=" + str(item_list[-1][1]) + "])"

        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeterogeneousMixtureDataEncoder):
            return False
        for encoder, comp_list in self.encoder_dict.items():
            if other.idx_dict[encoder] != comp_list:
                return False
        return True

    def seq_encode(
        self, x: Sequence[T], device: Optional[tn.device] = None
    ) -> "HeterogeneousMixtureTorchSequence":
        enc_data = []
        tag_list = []

        for enc_str, encoder_idx in self.idx_dict.items():
            tag_list.append(np.asarray(encoder_idx, dtype=int))
            enc_data.append(self.encoder_dict[enc_str].seq_encode(x, device=device))

        return HeterogeneousMixtureTorchSequence(
            data=(tag_list, enc_data), device=device
        )


class HeterogeneousMixtureTorchSequence(TorchEncodedSequence):

    def __init__(
        self,
        data: Tuple[List[np.ndarray], List[TorchEncodedSequence]],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:

        return f"HeterogeneousMixtureTorchSequence(device={repr(self.device)})"
