"""Create, estimate, and sample from a mixture distribution with homogenous components.

Defines the MixtureDistribution, MixtureSampler, MixtureAccumulatorFactory, MixtureAccumulator,
MixtureEstimator, and the MixtureDataEncoder classes for use with dmx-learn.

MixtureDistribution is defined by the density of the form,

P(Y) = sum_{k=1}^{K} P(Y|Z=k)*P(Z=k),

where P(Z=k) is a mixture weight for component k, and P(Y|Z=k) is defined as a the k^{th} component distribution.

If component distribution P(Y|Z=k) has data type (T), then the Mixture distribution has data type (T) as well.

"""
import numpy as np
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import maxrandint
from typing import Sequence, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict


T = TypeVar('T')   ### Type of Mixture component data.
T1 = TypeVar('T1') ### Type of encoded data.
T2 = TypeVar('T2') ### Type of component suff_stat
key_type = Union[Tuple[str, str], Tuple[None, None]]


class MixtureDistribution(SequenceEncodableProbabilityDistribution):
    """MixtureDistribution object defined by component distributions and weights.

    Attributes:
        components (Sequence[SequenceEncodableProbabilityDistribution]): List of component distributions (data type T).
        w (ndarray[float]): Mixture weights assigned from args (w).
        zw (ndarray[bool]): True if a weight is 0.0, else False.
        log_w (ndarray[float]): Log of weights (w). set to -np.inf, where zw is True.
        num_components (int): Number of components in MixtureDistribution instance.
        name (Optional[str]): String name to MixtureDistribution object.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.

    """

    def __init__(self,
                 components: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[np.ndarray, Sequence[float]],
                 name: Optional[str] = None,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """MixtureDistribution object.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Component distributions.
            w (ndarray[float]): Mixture weights, must sum to 1.0.
            name (Optional[str]): Assign string name to MixtureDistribution object.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.

        """
        if isinstance(w, np.ndarray):
            self.w = w
        else:
            self.w = np.asarray(w, dtype=float)

        self.zw = (self.w == 0.0)
        self.log_w = np.log(w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.components = components
        self.num_components = len(components)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s1 = ','.join([str(u) for u in self.components])
        s2 = repr(self.w.tolist())
        s3 = repr(self.name)
        s4 = repr(self.keys)

        return 'MixtureDistribution([%s], %s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: T) -> float:
        """Evaluate density of Mixture distribution at observation x.
        
        See log_density for details. 

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            float: Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate log-density of Mixture distribution at observation x.
        
        .. math::

           \\log{f(x)} = \\log{\\left(\\sum_{k=1}^{K} f_k(x) \\pi_k\\right)}.

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            float: Log-density at x.

        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def component_log_density(self, x: T) -> np.ndarray:
        """Evaluate component-wise log-density of Mixture distribution at observation x.
        
        Returns *num_components* dim array with :math:`\\log{\\left(f_k(x)\\right)}` in each entry.

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            np.ndarray: Component-wise log-density at x.

        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: T) -> np.ndarray:
        """Obtain the posterior distribution for each mixture component at observation x.
        
        .. math::

           f(z=k \\vert x ) = \\frac{f_k(x) \\pi_k}{\\sum_{k=1}^{K} f_k(x) \\pi_k}

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            np.ndarray: Posterior distribution at observation x.

        """
        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()

            return comp_log_density

    def seq_component_log_density(self, x: 'MixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of component_log_density. 

        Args:
            x (MixtureEncodedDataSequence): EncodedDataSequence for mixture component.

        Returns:
            np.ndarray: 2-d numpy array of floats having shape (sz,K), where sz is the number of iid obs in encoded
            sequence x, and K is the number of mixture components.

        """

        if not isinstance(x, MixtureEncodedDataSequence):
            raise Exception('MixtureEncodedDataSequence required for seq_component_log_density().')

        enc_data = x.data
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True
                ll_mat[:, i] = temp

        return ll_mat

    def seq_log_density(self, x: 'MixtureEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, MixtureEncodedDataSequence):
            raise Exception('MixtureEncodedDataSequence required for seq_log_density().')

        enc_data = x.data
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True
                ll_mat[:, i] = temp
                ll_mat[:, i] += self.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:

            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)

            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv

    def seq_posterior(self, x: T1) -> np.ndarray:
        """Vectorized evaluation of posterior.

        Args:
            x (MixtureEncodedDataSequence): EncodedDataSequence for mixture component.

        Returns:
            np.ndarray: Posterior of each observation in encoded sequence.

        """

        if not isinstance(x, MixtureEncodedDataSequence):
            raise Exception('MixtureEncodedDataSequence required for seq_posterior().')

        enc_data = x.data
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += self.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows] = np.max(self.log_w)
        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> 'MixtureSampler':
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'MixtureEstimator':
        if pseudo_count is not None:
            return MixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count, name=self.name, keys=self.keys)
        else:
            return MixtureEstimator([u.estimator() for u in self.components], name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'MixtureDataEncoder':
        dist_encoder = self.components[0].dist_to_encoder()
        return MixtureDataEncoder(encoder=dist_encoder)


class MixtureSampler(DistributionSampler):
    """MixtureSampler used to generate samples from instance of MixtureDistribution.

    Attributes:
        dist (MixtureDistribution): MixtureDistribution to draw samples from.
        rng (RandomState): Seeded RandomState for sampling.
        comp_samplers (Sequence[DistributionSamplers]): List of DistributionSampler objects for each mixture component.

    """

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None) -> None:
        """MixtureSampler object.

        Args:
            dist (MixtureDistribution): Assign MixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], Any]:
        """Draw iid samples from a mixture distribution.

        The data type drawn from 'comp_samplers' is type T, corresponding to the data type of the mixture components.

        If size is None, a single sample (of data type T) is drawn and returned. If size is not None, 'size'-iid
        mixture samples are drawn and returned as a Sequence with data type List[T].

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Data type T or Sequence[T].

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class MixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """MixtureAccumulator object used to aggregate the sufficient statistics of observed data.

    Attributes:
        accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
            SequenceEncodableStatisticAccumulator objects for the components of the mixture.
        num_components (int): Total number of mixture components (length of accumulators).
        comp_counts (np.ndarray[float]): Numpy array of floats for accumulating component weights.
        weight_key (Optional[str]): Key for weights of mixture.
        comp_key (Optional[str]): Key for components of mixture.
        _init_rng (bool): False if rng for accumulators has not been set.
        _w_rng (Optional[RandomState]): RandomState for generating weights in init.
        _acc_rng (Optional[Sequence[RandomState]]): List of RandomState obejcts for setting seed on accumulator
            initialization.

    """

    def __init__(self,
                 accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None) -> None:
        """MixtureAccumulator object.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator objects for the components of the mixture.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and mixture components.

        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.name = name

        # Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[Sequence[RandomState]] = None

    def seq_update(self, x: 'MixtureEncodedDataSequence', weights: np.ndarray, estimate: 'MixtureDistribution') -> None:
        enc_data = x.data
        ll_mat_init = False

        for i in range(estimate.num_components):

            if not estimate.zw[i]:

                temp = estimate.components[i].seq_log_density(enc_data)

                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components), dtype=np.float64)
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += estimate.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def update(self, x: T, weight: float, estimate: 'MixtureDistribution') -> None:
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _rng_acc member variable.

        Returns:
            None.

        """
        seeds = rng.randint(2 ** 31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = self._w_rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
        else:
            ww = np.zeros(self.num_components)

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, self._acc_rng[i])
            self.comp_counts[i] += w

    def seq_initialize(self, x: 'MixtureEncodedDataSequence', weights: np.ndarray, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = np.count_nonzero(keep_idx)
        ww = np.zeros((sz, self.num_components))

        if keep_len > 0:
            ww[keep_idx, :] = self._w_rng.dirichlet(alpha=np.ones(self.num_components) / (self.num_components ** 2),
                                            size=keep_len)
        ww *= np.reshape(weights, (sz, 1))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x.data, ww[:, i], self._acc_rng[i])
            self.comp_counts[i] += np.sum(ww[:, i])

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


class MixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """MixtureAccumulatorFactory object for creating MixtureAccumulator objects.

    Attributes:
        factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
            components.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.
        name (Optional[str]): Name for object.

    """

    def __init__(self,
                 factories: Sequence[StatisticAccumulatorFactory],
                 keys: Tuple[Optional[str], Optional[str]] = (None, None),
                 name: Optional[str] = None) -> None:
        """MixtureAccumulatorFactory object.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
                components.
            keys (Tuple[Optional[str], Optional[str]]): Assign keys for weights and component aggregations.
            name (Optional[str]): Name for object.

        """
        self.factories = factories
        self.keys = keys
        self.name = name

    def make(self) -> 'MixtureAccumulator':
        return MixtureAccumulator([factory.make() for factory in self.factories], keys=self.keys, name=self.name)


class MixtureEstimator(ParameterEstimator):
    """MixtureEstimator object used to estimate MixtureDistribution from aggregated sufficient statistics.

    Attributes:
        estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the mixture
            components.
        fixed_weights (Optional[np.ndarray]): Treat mixture weights as fixed values. Must sum to 1.0.
        suff_stat (Optional[np.ndarray]): Weights of the mixture. Must sum to 1.0.
        pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
        name (Optional[str]): Name for MixtureEstimator object.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component distributions.

    """

    def __init__(self,
                 estimators: Sequence[ParameterEstimator],
                 fixed_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                 suff_stat: Optional[np.ndarray] = None,
                 pseudo_count: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """MixtureEstimator object.

        Args:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the mixture
                components.
            fixed_weights (Optional[Union[Sequence[float], np.ndarray]]): Set fixed values for mixture weights.
            suff_stat (Optional[np.ndarray]): Numpy array of floats with length equal to length of estimators.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            name (Optional[str]): Set a name to the MixtureEstimator object.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.

        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError("MixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")
        
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.fixed_weights = np.asarray(fixed_weights) if fixed_weights is not None else None

    def accumulator_factory(self) -> 'MixtureAccumulatorFactory':
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureAccumulatorFactory(est_factories, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'MixtureDistribution':
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

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

        return MixtureDistribution(components, w, name=self.name)


class MixtureDataEncoder(DataSequenceEncoder):
    """MixtureDataEncoder used for sequence encoding data for use with vectorized 'seq_' functions.

    Notes:
        Data must be type T, that matches the data type of each Mixture component.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for encoding sequence of iid data.

    """

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """MixtureDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the component Distributions.

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

    def seq_encode(self, x: Sequence[T]) -> 'MixtureEncodedDataSequence':
        """Sequence encoder a sequence of iid observations that match the data type of 'encoder' member variable.

        Note: MixtureDataEncoder attribute 'encoder' is an encoder for the components of the MixtureDistribution.
        The data type for 'encoder' is T.

        Args:
            x (Sequence[T]): A Sequence of iid observations drawn from a mixture distribution with component
                distributions consistent with 'encoder'.

        Returns:
            MixtureEncodedDataSequence

        """
        return MixtureEncodedDataSequence(data=self.encoder.seq_encode(x))


class MixtureEncodedDataSequence(EncodedDataSequence):
    """MixtureEncodedDataSequence object for vectorized function calls.

     Attributes:
         data (EncodedDataSequence): EncodedDataSequence for the mixture component distribution type.

     """

    def __init__(self, data: EncodedDataSequence):
        """MixtureEncodedDataSequence object.

         Args:
             data (EncodedDataSequence): EncodedDataSequence for the mixture component distribution type.

         """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'MixtureEncodedDataSequence(data={self.data})'

