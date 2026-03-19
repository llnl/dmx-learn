"""Create, estimate, and sample from a mixture distribution with homogenous components.

Defines the MixtureDistribution, MixtureSampler, MixtureAccumulatorFactory, MixtureAccumulator,
MixtureEstimator, and the MixtureDataEncoder classes for use with pysparkplug.

MixtureDistribution is defined by the density of the form,

P(Y) = sum_{k=1}^{K} P(Y|Z=k)*P(Z=k),

where P(Z=k) is a mixture weight for component k, and P(Y|Z=k) is defined as a the k^{th} component distribution.

If component distribution P(Y|Z=k) has data type (T), then the Mixture distribution has data type (T) as well.

"""
import torch as tn
import numpy as np
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict


T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')

key_type = Union[Tuple[str, str], Tuple[None, None]]


def _sample_dirichlet_like(alpha: tn.Tensor, size: int, tng: tn.Generator) -> tn.Tensor:
    return vec.sample_dirichlet(alpha=alpha, size=size, tng=tng)


class MixtureDistribution(TorchProbabilityDistribution):
    """MixtureDistribution object defining mixture distribution with torch tensors.

    Attributes:
        w (tn.tensor): Mixture weights.
        zw (tn.tensor): True where weights are 0.0.
        log_w (tn.tensor): Log of mixture weights.
        components (Sequence[TorchProbabilityDistribution]): Mixture components, all TorchProbabilityDistributions of
            the same type.
        num_components (int): Number of mixture comps.

    """
    def __init__(self, components: Sequence[TorchProbabilityDistribution],
                 w: Union[np.ndarray, List[float], tn.Tensor],
                 device: Optional[tn.device] = None) -> None:
        """MixtureDistribution object.

        Args:
            components (Sequence[TorchProbabilityDistribution]): Mixture components, all TorchProbabilityDistributions of
                the same type.
            w (Union[np.ndarray, List[float], tn.Tensor]): Mixture weights.
            device (Optional[tn.device]): Device to declare tensors on.

        """
        super().__init__(device)

        self.w = vec.tensor(w, device=self._device)
        self.zw = (self.w == 0.0)
        self.log_w = tn.log(self.w + self.zw)
        self.log_w[self.zw] = -tn.inf
        self.components = components
        self.num_components = len(components)

    def to(self, device: tn.device) -> None:
        self._device = device
        self.w = self.w.to(device)

        for comp in self.components:
            comp.to(device)

    def __str__(self) -> str:
        s1 = ','.join([str(u) for u in self.components])
        s2 = repr(self.w.data.cpu().tolist())

        return 'MixtureDistribution([%s], %s)' % (s1, s2)

    def density(self, x: T) -> float:
        """Evaluate density of Mixture distribution at observation x.

        Notes:
            See log_density() for details.

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            float: Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate log-density of Mixture distribution at observation x.

        Notes:
            A K-component Mixture has log-density,

            log(P(x)) = log(sum_{z=k}^{K} P(x|z=k)*P(z=k)),

            where P(x|z=k) is component-k log-density at x, and P(z=k) = w[k]. A log-sum-exp is used to evaluate the
            sum inside the log of the right-hand side above. (See dmx.utils.vector.log_sum() for details).

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            float: Log-density at x.

        """
        rv = tn.logsumexp(vec.tensor([u.log_density(x) for u in self.components], device=self._device) + self.log_w, dim=0)
        return float(rv)

    def component_log_density(self, x: T) -> tn.Tensor:
        return vec.tensor([m.log_density(x) for m in self.components], device=self._device)

    def posterior(self, x: T) -> tn.Tensor:
        
        comp_log_density = vec.tensor([m.log_density(x) for m in self.components], device=self._device)
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -tn.inf

        rv = tn.logsumexp(comp_log_density, dim=0)
        if tn.isinf(rv):
            return self.w
        else:
            comp_log_density -= rv
            tn.exp(comp_log_density, out=comp_log_density)
            return comp_log_density

    def seq_component_log_density(self, x: 'MixtureTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, MixtureTorchEncodedSequence):
            raise Exception('MixtureTorchEncodedSequence required for `seq_` function calls.')
        
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros((len(temp), self.num_components), device=self._device)
                    ll_mat += -tn.inf
                    ll_mat_init = True
                ll_mat[:, i] = temp

        return ll_mat

    def seq_log_density(self, x: 'MixtureTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, MixtureTorchEncodedSequence):
            raise Exception('MixtureTorchEncodedSequence required for `seq_` function calls.')

        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros((len(temp), self.num_components), device=self._device)
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

        else:

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

    def seq_posterior(self, x: 'MixtureTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, MixtureTorchEncodedSequence):
            raise Exception('MixtureTorchEncodedSequence required for `seq_` function calls.')

        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(x.data)
                if not ll_mat_init:
                    ll_mat = vec.zeros((len(temp), self.num_components), device=self._device)
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

    def sampler(self, seed: Optional[int] = None) -> 'MixtureSampler':
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'MixtureEstimator':
        if pseudo_count is not None:
            return MixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count)
        else:
            return MixtureEstimator([u.estimator() for u in self.components])

    def dist_to_encoder(self) -> 'MixtureDataEncoder':
        dist_encoder = self.components[0].dist_to_encoder()
        return MixtureDataEncoder(encoder=dist_encoder)


class MixtureSampler(DistributionSampler):
    """MixtureSampler used to generate samples from instance of MixtureDistribution.

    Attributes:
        rng (RandomState): Seeded RandomState for sampling.
        w (np.ndarray): Mixture weights.
        comp_samplers (List[DistributionSamplers]): List of DistributionSampler objects for each mixture component.
        num_components (int): Number of mixture components.

    """

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None) -> None:
        """MixtureSampler object.

        Args:
            dist (MixtureDistribution): Assign MixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.w = dist.w.data.cpu().numpy()
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in dist.components]
        self.num_components = len(self.comp_samplers)

    def sample(self, size: Optional[int] = None) -> Union[List[Any], Any]:
        """Draw iid samples from a mixture distribution.

        The data type drawn from 'comp_samplers' is type T, corresponding to the data type of the mixture components.

        If size is None, a single sample (of data type T) is drawn and returned. If size is not None, 'size'-iid
        mixture samples are drawn and returned as a List with data type List[T].

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Data type T or List[T].

        """
        comp_state = self.rng.choice(range(0, self.num_components), size=size, replace=True, p=self.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class MixtureAccumulator(TorchStatisticAccumulator):

    def __init__(self,
                 accumulators: Sequence[TorchStatisticAccumulator],
                 keys: Tuple[Optional[str], Optional[str]] = (None, None),
                 device: Optional[tn.device] = None):
        super().__init__(device)
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

        self.comp_counts = np.zeros(self.num_components, dtype=np.float64)

    def seq_update(self, x: 'MixtureTorchEncodedSequence', weights: tn.Tensor, estimate: 'MixtureDistribution') -> None:

        ll_mat_init = False

        for i in range(estimate.num_components):

            if not estimate.zw[i]:

                temp = estimate.components[i].seq_log_density(x.data)

                if not ll_mat_init:
                    ll_mat = vec.zeros((len(temp), self.num_components), device=self._device)
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

    def seq_initialize(self, x: 'MixtureTorchEncodedSequence', weights: tn.Tensor, tng: tn.Generator) -> None:
        sz = len(weights)
        keep_idx = weights > 0
        keep_len = tn.count_nonzero(keep_idx)
        ww = vec.zeros((sz, self.num_components), device=self._device)

        if keep_len > 0:
            alpha = vec.ones(self.num_components, device=self._device) / self.num_components**2
            ww[keep_idx, :] += _sample_dirichlet_like(alpha=alpha, size=int(keep_len), tng=tng)

        ww *= tn.reshape(weights, (sz, 1))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x.data, ww[:, i], tng)
            self.comp_counts[i] += float(tn.sum(ww[:, i]))

    def combine(self, suff_stat: Tuple[np.ndarray, Tuple[T2, ...]]) -> 'MixtureAccumulator':
        self.comp_counts += suff_stat[0]
        
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: Tuple[np.ndarray, Tuple[T2, ...]]) -> 'MixtureAccumulator':
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
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
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

    def acc_to_encoder(self) -> 'MixtureDataEncoder':
        acc_encoder = self.accumulators[0].acc_to_encoder()
        return MixtureDataEncoder(encoder=acc_encoder)


class MixtureAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """MixtureAccumulatorFactory object for creating MixtureAccumulator objects.

    Attributes:
        factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
            components.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.

    """

    def __init__(self,
                 factories: Sequence[TorchStatisticAccumulatorFactory],
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)):
        """MixtureAccumulatorFactory object for creating MixtureAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
                components.
            keys (Tuple[Optional[str], Optional[str]]): Assign keys for weights and component aggregations.

        """
        self.factories = factories
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'MixtureAccumulator':
        if device is not None:
            factories = [factory.make(device=device) for factory in self.factories]
            return MixtureAccumulator(factories, keys=self.keys, device=device)
            
        else:
            factories = [factory.make() for factory in self.factories]
            return MixtureAccumulator(factories, keys=self.keys)
            

class MixtureEstimator(TorchParameterEstimator):
    """MixtureEstimator object used to estimate MixtureDistribution from aggregated sufficient statistics.

    Attributes:
        estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the mixture
            components.
        fixed_weights (Optional[np.ndarray]): Treat mixture weights as fixed values. Must sum to 1.0.
        suff_stat (Optional[np.ndarray]): Weights of the mixture. Must sum to 1.0.
        pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component distributions.

    """

    def __init__(self,
                 estimators: Sequence[TorchParameterEstimator],
                 fixed_weights: Optional[Union[List[float], tn.Tensor]] = None,
                 suff_stat: Optional[np.ndarray] = None,
                 pseudo_count: Optional[float] = None,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """MixtureEstimator object used to estimate MixtureDistribution from aggregated sufficient statistics.

        Args:
            estimators (Sequence[TorchParameterEstimator]): Sequence of TorchParameterEstimator objects for the mixture
                components.
            fixed_weights (Optional[Union[List[float], np.ndarray]]): Set fixed values for mixture weights.
            suff_stat (Optional[np.ndarray]): Numpy array of floats with length equal to length of estimators.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

        self.fixed_weights = np.asarray(fixed_weights) if fixed_weights is not None else None

    def accumulator_factory(self) -> 'MixtureAccumulatorFactory':
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureAccumulatorFactory(est_factories, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, Tuple[Any, ...]], device: Optional[tn.device] = None) -> 'MixtureDistribution':
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i], device=device) for i in range(num_components)]

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return MixtureDistribution(components, w, device=device)


class MixtureDataEncoder(TorchSequenceEncoder):
    """MixtureDataEncoder object for creating MixtureTorchEncodedSequence instances.

    Attributes:
        encoder (TorchSequenceEncoder): TorchSequenceEncoder for data compatible with each mixture component.

    """

    def __init__(self, encoder: TorchSequenceEncoder) -> None:
        """MixtureDataEncoder object.

        Args:
            encoder (TorchSequenceEncoder): TorchSequenceEncoder for data compatible with each mixture component.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        return 'MixtureDataEncoder(' + str(self.encoder) + ')'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixtureDataEncoder):
            return self.encoder == other
        else:
            if other.encoder == self.encoder:
                return True
            else:
                return False

    def seq_encode(self, x: Sequence[T], device: Optional[tn.device] = None) -> 'MixtureTorchEncodedSequence':
        return MixtureTorchEncodedSequence(data=self.encoder.seq_encode(x, device=device), device=device)


class MixtureTorchEncodedSequence(TorchEncodedSequence):
    """MixtureTorchEncodedSequence object for use with vectorized `seq_` function calls.

    Attributes:
        data (TorchEncodedSequence): TorchEncodedSequence for data of mixture components.
        device (tn.device): Device tensors of instance are declared on.

    """

    def __init__(self, data: TorchEncodedSequence, device: Optional[tn.device] = None):
        """MixtureTorchEncodedSequence object.

        Args:
            data (TorchEncodedSequence): TorchEncodedSequence for data of mixture components.
            device (Optional[tn.device]): Device tensors of instance are declared on.

        """
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'MixtureTorchEncodedSequence(device={repr(self.device)})'
