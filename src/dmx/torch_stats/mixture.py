"""Provide torch-backed finite mixtures with homogeneous components.

A latent component ``Z`` is drawn from ``K`` mixture weights and the observed
value is drawn from ``components[Z]``. All components accept the same data type
and share one encoded observation sequence. Encoded component log densities
and posterior responsibilities have shape ``(N, K)``; mixture log densities
have shape ``(N,)``. Parameters and children move with ``to``. Floating tensors
use the vector-helper dtype, normally float64 and float32 on MPS, while sampled
weights and accumulated component counts use CPU NumPy arrays. This mirrors
``dmx.stats.mixture`` without its optional name.
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

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

key_type = Union[Tuple[str, str], Tuple[None, None]]


def _sample_dirichlet_like(alpha: tn.Tensor, size: int, tng: tn.Generator) -> tn.Tensor:
    return vec.sample_dirichlet(alpha=alpha, size=size, tng=tng)


class MixtureDistribution(TorchProbabilityDistribution):
    """Represent a finite mixture of ``K`` homogeneous child distributions."""

    def __init__(
        self,
        components: Sequence[TorchProbabilityDistribution],
        w: Union[np.ndarray, List[float], tn.Tensor],
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize components and a length-``K`` mixture-weight tensor."""
        super().__init__(device)

        self.w = vec.tensor(w, device=self._device)
        self.zw = self.w == 0.0
        self.log_w = tn.log(self.w + self.zw)
        self.log_w[self.zw] = -tn.inf
        self.components = components
        self.num_components = len(components)

    def to(self, device: vec.DeviceLike) -> "MixtureDistribution":
        """Move weights and all component models to ``device`` in place."""
        target_device = self._resolve_device_arg(device)
        self._device = target_device
        self.w = self.w.to(target_device)
        self.zw = self.w == 0.0
        self.log_w = tn.log(self.w + self.zw)
        self.log_w[self.zw] = -tn.inf

        for comp in self.components:
            comp.to(target_device)
        return self

    def __str__(self) -> str:
        """Return a constructor-like representation using CPU weights."""
        s1 = ",".join([str(u) for u in self.components])
        s2 = repr(self.w.data.cpu().tolist())

        return f"MixtureDistribution([{s1}], {s2})"

    def density(self, x: T) -> float:
        """Evaluate the marginal density of one observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: T) -> float:
        """Marginalize the latent component with a torch log-sum-exp."""
        rv = tn.logsumexp(
            vec.tensor([u.log_density(x) for u in self.components], device=self._device)
            + self.log_w,
            dim=0,
        )
        return float(rv)

    def component_log_density(self, x: T) -> tn.Tensor:
        """Return ``K`` unweighted component log densities for one observation."""
        return vec.tensor(
            [m.log_density(x) for m in self.components], device=self._device
        )

    def posterior(self, x: T) -> tn.Tensor:
        """Return length-``K`` posterior component responsibilities."""
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

    def seq_component_log_density(self, x: "MixtureTorchEncodedSequence") -> tn.Tensor:
        """Return unweighted component log densities with shape ``(N, K)``."""
        if not isinstance(x, MixtureTorchEncodedSequence):
            raise TypeError(
                "MixtureTorchEncodedSequence required for `seq_` function calls."
            )

        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros(
                        (len(temp), self.num_components), device=self._device
                    )
                    ll_mat += -tn.inf
                    ll_mat_init = True
                ll_mat[:, i] = temp

        return ll_mat

    def seq_log_density(self, x: "MixtureTorchEncodedSequence") -> tn.Tensor:
        """Return marginal mixture log densities with shape ``(N,)``."""
        if not isinstance(x, MixtureTorchEncodedSequence):
            raise TypeError(
                "MixtureTorchEncodedSequence required for `seq_` function calls."
            )

        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros(
                        (len(temp), self.num_components), device=self._device
                    )
                    ll_mat += -tn.inf
                    ll_mat_init = True
                ll_mat[:, i] = temp
                ll_mat[:, i] += self.log_w[i]

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

    def seq_posterior(self, x: "MixtureTorchEncodedSequence") -> tn.Tensor:
        """Return posterior responsibilities with shape ``(N, K)``."""
        if not isinstance(x, MixtureTorchEncodedSequence):
            raise TypeError(
                "MixtureTorchEncodedSequence required for `seq_` function calls."
            )

        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros(
                        (len(temp), self.num_components), device=self._device
                    )
                    ll_mat += -tn.inf
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += self.log_w[i]

        ll_max, _ = ll_mat.max(dim=1, keepdim=True)
        bad_rows = tn.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows] = tn.max(self.log_w)
        ll_mat -= ll_max

        tn.exp(ll_mat, out=ll_mat)
        tn.sum(ll_mat, dim=1, keepdim=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> "MixtureSampler":
        """Create a sampler for latent components and their observations."""
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "MixtureEstimator":
        """Create component estimators and optional weight smoothing."""
        if pseudo_count is not None:
            return MixtureEstimator(
                [
                    u.estimator(pseudo_count=1.0 / self.num_components)
                    for u in self.components
                ],
                pseudo_count=pseudo_count,
            )
        return MixtureEstimator([u.estimator() for u in self.components])

    def dist_to_encoder(self) -> "MixtureDataEncoder":
        """Create the shared encoder from the first homogeneous component."""
        dist_encoder = self.components[0].dist_to_encoder()
        return MixtureDataEncoder(encoder=dist_encoder)


class MixtureSampler(DistributionSampler):
    """Draw a latent component and then an observation from that component."""

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None) -> None:
        """Initialize CPU component selection and independently seeded samplers."""
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.w = dist.w.data.cpu().numpy()
        self.comp_samplers = [
            d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in dist.components
        ]
        self.num_components = len(self.comp_samplers)

    def sample(self, size: Optional[int] = None) -> Union[List[Any], Any]:
        """Draw one observation or a list of ``size`` observations."""
        comp_state = self.rng.choice(
            range(0, self.num_components), size=size, replace=True, p=self.w
        )

        if size is None:
            return self.comp_samplers[comp_state].sample()
        return [self.comp_samplers[i].sample() for i in comp_state]


class MixtureAccumulator(TorchStatisticAccumulator):
    """Accumulate component counts and responsibility-weighted child statistics."""

    def __init__(
        self,
        accumulators: Sequence[TorchStatisticAccumulator],
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        device: Optional[tn.device] = None,
    ):
        """Initialize ``K`` child accumulators and CPU component counts."""
        super().__init__(device)
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

        self.comp_counts = np.zeros(self.num_components, dtype=np.float64)

    def seq_update(
        self,
        x: "MixtureTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: "MixtureDistribution",
    ) -> None:
        """Update children using an ``(N, K)`` responsibility matrix."""
        ll_mat_init = False

        for i in range(estimate.num_components):

            if not estimate.zw[i]:

                temp = estimate.components[i].seq_log_density(x.data)

                if not ll_mat_init:
                    ll_mat = vec.zeros(
                        (len(temp), self.num_components), device=self._device
                    )
                    ll_mat += -tn.inf
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += estimate.log_w[i]

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

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]
            self.comp_counts[i] += float(w_loc.sum())
            self.accumulators[i].seq_update(x.data, w_loc, estimate.components[i])

    def seq_initialize(
        self, x: "MixtureTorchEncodedSequence", weights: tn.Tensor, tng: tn.Generator
    ) -> None:
        """Randomly split ``(N,)`` weights across components with Dirichlet draws."""
        sz = len(weights)
        keep_idx = weights > 0
        keep_len = tn.count_nonzero(keep_idx)
        ww = vec.zeros((sz, self.num_components), device=self._device)

        if keep_len > 0:
            alpha = (
                vec.ones(self.num_components, device=self._device)
                / self.num_components**2
            )
            ww[keep_idx, :] += _sample_dirichlet_like(
                alpha=alpha, size=int(keep_len), tng=tng
            )

        ww *= tn.reshape(weights, (sz, 1))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x.data, ww[:, i], tng)
            self.comp_counts[i] += float(tn.sum(ww[:, i]))

    def combine(
        self, suff_stat: Tuple[np.ndarray, Tuple[T2, ...]]
    ) -> "MixtureAccumulator":
        """Merge component counts and the length-``K`` child-statistic tuple."""
        self.comp_counts += suff_stat[0]

        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        """Return CPU component counts and child sufficient statistics."""
        return self.comp_counts, tuple(u.value() for u in self.accumulators)

    def from_value(self, x: Tuple[np.ndarray, Tuple[T2, ...]]) -> "MixtureAccumulator":
        """Replace component counts and child sufficient statistics."""
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge weight, component-list, and recursive child keys."""
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
        """Replace weight, component-list, and recursive child keys."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "MixtureDataEncoder":
        """Create the shared encoder from the first child accumulator."""
        acc_encoder = self.accumulators[0].acc_to_encoder()
        return MixtureDataEncoder(encoder=acc_encoder)


class MixtureAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create mixture accumulators from ``K`` component factories."""

    def __init__(
        self,
        factories: Sequence[TorchStatisticAccumulatorFactory],
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
    ):
        """Initialize component factories and weight/component merge keys."""
        self.factories = factories
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "MixtureAccumulator":
        """Create component accumulators on ``device`` when it is explicit."""
        if device is not None:
            factories = [factory.make(device=device) for factory in self.factories]
            return MixtureAccumulator(factories, keys=self.keys, device=device)

        factories = [factory.make() for factory in self.factories]
        return MixtureAccumulator(factories, keys=self.keys)


class MixtureEstimator(TorchParameterEstimator):
    """Estimate component models and categorical mixture weights."""

    def __init__(
        self,
        estimators: Sequence[TorchParameterEstimator],
        fixed_weights: Optional[Union[List[float], tn.Tensor]] = None,
        suff_stat: Optional[np.ndarray] = None,
        pseudo_count: Optional[float] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
    ) -> None:
        """Initialize component estimators and weight-estimation controls.

        Args:
            estimators: One estimator for each latent component.
            fixed_weights: Optional fixed length-``K`` weight vector.
            suff_stat: Optional length-``K`` prior proportions.
            pseudo_count: Strength of uniform or supplied weight smoothing.
            keys: Separate keys for weights and the component accumulator list.
        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

        self.fixed_weights = (
            np.asarray(fixed_weights) if fixed_weights is not None else None
        )

    def accumulator_factory(self) -> "MixtureAccumulatorFactory":
        """Create a factory from the component estimator factories."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureAccumulatorFactory(est_factories, keys=self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, Tuple[Any, ...]],
        device: Optional[tn.device] = None,
    ) -> "MixtureDistribution":
        """Estimate components from responsibilities and normalize weights."""
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

        return MixtureDistribution(components, w, device=device)


class MixtureDataEncoder(TorchSequenceEncoder):
    """Encode observations once for all homogeneous mixture components."""

    def __init__(self, encoder: TorchSequenceEncoder) -> None:
        """Initialize the shared component encoder."""
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a representation of the shared child encoder."""
        return "MixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether the wrapped encoders compare equal."""
        if not isinstance(other, MixtureDataEncoder):
            return self.encoder == other
        return other.encoder == self.encoder

    def seq_encode(
        self, x: Sequence[T], device: Optional[tn.device] = None
    ) -> "MixtureTorchEncodedSequence":
        """Encode ``N`` observations once on ``device`` for all components."""
        return MixtureTorchEncodedSequence(
            data=self.encoder.seq_encode(x, device=device), device=device
        )


class MixtureTorchEncodedSequence(TorchEncodedSequence):
    """Wrap one child encoding shared by every homogeneous component."""

    def __init__(self, data: TorchEncodedSequence, device: Optional[tn.device] = None):
        """Initialize the shared child encoding and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"MixtureTorchEncodedSequence(device={repr(self.device)})"
