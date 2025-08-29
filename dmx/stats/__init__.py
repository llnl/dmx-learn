"""Initialization module for the stats package.

This module initializes the stats subpackage.
"""
__all__ = [
    "initialize",
    "estimate",
    "seq_encode",
    "seq_log_density",
    "seq_log_density_sum",
    "seq_estimate",
    "seq_initialize",
    "BinomialDistribution",
    "BinomialEstimator",
    "CategoricalDistribution",
    "CategoricalEstimator",
    "MultinomialDistribution",
    "MultinomialEstimator",
    "CompositeDistribution",
    "CompositeEstimator",
    "ConditionalDistribution",
    "ConditionalDistributionEstimator",
    "DiracMixtureDistribution",
    "DiracMixtureEstimator",
    "DirichletDistribution",
    "DirichletEstimator",
    "DiagonalGaussianMixtureDistribution",
    "DiagonalGaussianMixtureEstimator",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianEstimator",
    "ExponentialDistribution",
    "ExponentialEstimator",
    "GammaDistribution",
    "GammaEstimator",
    "GaussianDistribution",
    "GaussianEstimator",
    "GeometricDistribution",
    "GeometricEstimator",
    "GaussianMixtureDistribution",
    "GaussianMixtureEstimator",
    "HeterogeneousMixtureDistribution",
    "HeterogeneousMixtureEstimator",
    "HiddenAssociationDistribution",
    "HiddenAssociationEstimator",
    "HiddenMarkovModelDistribution",
    "HiddenMarkovEstimator",
    "HierarchicalMixtureDistribution",
    "HierarchicalMixtureEstimator",
    "ICLTreeDistribution",
    "ICLTreeEstimator",
    "IgnoredDistribution",
    "IgnoredEstimator",
    "IntegerBernoulliEditDistribution",
    "IntegerBernoulliEditEstimator",
    "IntegerStepBernoulliEditDistribution",
    "IntegerStepBernoulliEditEstimator",
    "IntegerHiddenAssociationDistribution",
    "IntegerHiddenAssociationEstimator",
    "IntegerMarkovChainDistribution",
    "IntegerMarkovChainEstimator",
    "IntegerPLSIDistribution",
    "IntegerPLSIEstimator",
    "SpikeAndSlabDistribution",
    "SpikeAndSlabEstimator",
    "IntegerMultinomialDistribution",
    "IntegerMultinomialEstimator",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalEstimator",
    "IntegerBernoulliSetDistribution",
    "IntegerBernoulliSetEstimator",
    "JointMixtureDistribution",
    "JointMixtureEstimator",
    "LDADistribution",
    "LDAEstimator",
    "LogGaussianDistribution",
    "LogGaussianEstimator",
    "LookbackHiddenMarkovDistribution",
    "LookbackHiddenMarkovEstimator",
    "MarkovChainDistribution",
    "MarkovChainEstimator",
    "MixtureDistribution",
    "MixtureEstimator",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    "NullDistribution",
    "NullEstimator",
    "OptionalDistribution",
    "OptionalEstimator",
    "PoissonDistribution",
    "PoissonEstimator",
    "SequenceDistribution",
    "SequenceEstimator",
    "BernoulliSetDistribution",
    "BernoulliSetEstimator",
    "SparseMarkovAssociationDistribution",
    "SparseMarkovAssociationEstimator",
    "SpearmanRankingDistribution",
    "SpearmanRankingEstimator",
    "SemiSupervisedMixtureDistribution",
    "SemiSupervisedMixtureEstimator",
    "TreeHiddenMarkovModelDistribution",
    "TreeHiddenMarkovEstimator",
    "VonMisesFisherDistribution",
    "VonMisesFisherEstimator",
    "WeightedDistribution",
    "WeightedEstimator"
]

# Abstract Classes
import pyspark.rdd
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DataSequenceEncoder, \
    EncodedDataSequence

# Discrete base distributions
from dmx.stats.binomial import BinomialDistribution, BinomialEstimator
from dmx.stats.categorical import CategoricalDistribution, CategoricalEstimator
from dmx.stats.poisson import PoissonDistribution, PoissonEstimator
from dmx.stats.geometric import GeometricDistribution, GeometricEstimator
from dmx.stats.int_spike import SpikeAndSlabDistribution, SpikeAndSlabEstimator
from dmx.stats.intrange import IntegerCategoricalDistribution, IntegerCategoricalEstimator
from dmx.stats.catmultinomial import MultinomialDistribution, MultinomialEstimator
from dmx.stats.intmultinomial import IntegerMultinomialDistribution, IntegerMultinomialEstimator

# Continuous base distributions
from dmx.stats.exponential import ExponentialDistribution, ExponentialEstimator
from dmx.stats.gamma import GammaDistribution, GammaEstimator
from dmx.stats.gaussian import GaussianDistribution, GaussianEstimator
from dmx.stats.dirichlet import DirichletDistribution, DirichletEstimator
from dmx.stats.vmf import VonMisesFisherDistribution, VonMisesFisherEstimator
from dmx.stats.log_gaussian import LogGaussianDistribution, LogGaussianEstimator
from dmx.stats.gmm import GaussianMixtureDistribution, GaussianMixtureEstimator


# combinators distributions
from dmx.stats.composite import CompositeDistribution, CompositeEstimator
from dmx.stats.conditional import ConditionalDistribution, ConditionalDistributionEstimator
from dmx.stats.sequence import SequenceDistribution, SequenceEstimator
from dmx.stats.ignored import IgnoredDistribution, IgnoredEstimator
from dmx.stats.optional import OptionalDistribution, OptionalEstimator
from dmx.stats.weighted import WeightedDistribution, WeightedEstimator

# Generic Distributions
from dmx.stats.categorical import CategoricalDistribution, CategoricalEstimator
from dmx.stats.mixture import MixtureDistribution, MixtureEstimator
from dmx.stats.heterogeneous_mixture import HeterogeneousMixtureDistribution, HeterogeneousMixtureEstimator
from dmx.stats.markovchain import MarkovChainDistribution, MarkovChainEstimator
from dmx.stats.null_dist import NullDistribution, NullEstimator
from dmx.stats.hidden_association import HiddenAssociationDistribution, HiddenAssociationEstimator
from dmx.stats.hidden_markov import HiddenMarkovModelDistribution, HiddenMarkovEstimator
from dmx.stats.jmixture import JointMixtureDistribution, JointMixtureEstimator
from dmx.stats.tree_hmm import TreeHiddenMarkovModelDistribution, TreeHiddenMarkovEstimator
# from dmx.stats.lda import LDADistribution, LDAEstimator
from dmx.stats.markovchain import MarkovChainDistribution, MarkovChainEstimator
from dmx.stats.setdist import BernoulliSetDistribution, BernoulliSetEstimator
from dmx.stats.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator
from dmx.stats.look_back_hmm import LookbackHiddenMarkovDistribution, LookbackHiddenMarkovEstimator
from dmx.stats.lda import LDADistribution, LDAEstimator

# Reduced Generic Distributions
from dmx.stats.hmixture import HierarchicalMixtureDistribution, HierarchicalMixtureEstimator
from dmx.stats.int_edit_setdist import IntegerBernoulliEditDistribution, IntegerBernoulliEditEstimator
from dmx.stats.int_edit_stepsetdist import IntegerStepBernoulliEditDistribution, IntegerStepBernoulliEditEstimator
from dmx.stats.int_markovchain import IntegerMarkovChainDistribution, IntegerMarkovChainEstimator
from dmx.stats.int_plsi import IntegerPLSIDistribution, IntegerPLSIEstimator
from dmx.stats.icltree import ICLTreeDistribution, ICLTreeEstimator
from dmx.stats.intsetdist import IntegerBernoulliSetDistribution, IntegerBernoulliSetEstimator
from dmx.stats.mvn import MultivariateGaussianDistribution, MultivariateGaussianEstimator
from dmx.stats.int_hidden_association import IntegerHiddenAssociationDistribution, IntegerHiddenAssociationEstimator
from dmx.stats.sparse_markov_transform import SparseMarkovAssociationDistribution, SparseMarkovAssociationEstimator
from dmx.stats.dmvn import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from dmx.stats.ss_mixture import SemiSupervisedMixtureDistribution, SemiSupervisedMixtureEstimator
from dmx.stats.dirac_length import DiracMixtureDistribution, DiracMixtureEstimator
from dmx.stats.dmvn_mixture import DiagonalGaussianMixtureDistribution, DiagonalGaussianMixtureEstimator


### imports
import numpy as np
import pickle
from numpy.random import RandomState

from typing import Optional, TypeVar, List, Tuple, Any, Union, Sequence

T = TypeVar('T')
T_D = TypeVar('T_D', bound=SequenceEncodableProbabilityDistribution)


def load_models(x: str) -> SequenceEncodableProbabilityDistribution:
    """Load a model from a string representation.

    Args:
        x (str): String representation of the model.

    Returns:
        SequenceEncodableProbabilityDistribution: Loaded model.
    """
    return eval(x)


def dump_models(x: SequenceEncodableProbabilityDistribution) -> str:
    """Dump a model to its string representation.

    Args:
        x (SequenceEncodableProbabilityDistribution): Model to dump.

    Returns:
        str: String representation of the model.
    """
    return str(x)


def initialize(
    data: Union[Sequence[T], pyspark.rdd.RDD],
    estimator: ParameterEstimator,
    rng: np.random.RandomState,
    p: float = 0.1
) -> SequenceEncodableProbabilityDistribution:
    """Randomly initialize a model corresponding to ParameterEstimator for iid observations data.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Set of iid observations compatible with 'estimator'.
        estimator (ParameterEstimator): ParameterEstimator object for desired model to be estimated from data.
        rng (np.random.RandomState): RandomState object for setting seed.
        p (float, optional): Proportion of data to randomly sample for initializing model. Defaults to 0.1.

    Returns:
        SequenceEncodableProbabilityDistribution: Initialized model.
    """
    if isinstance(data, pyspark.rdd.RDD):
        factory = estimator.accumulator_factory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(2 ** 31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(seeds)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_w = np.random.RandomState(seed=rng_loc.randint(2 ** 31))

            for x in itr:
                w = rng.binomial(n=1, p=p)
                counts_for_split += w
                accumulator_for_split.initialize(x, w, rng_loc)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2 ** 31))

        for i, x in enumerate(idata):
            w = rng_w.binomial(n=1, p=p)
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


def estimate(
    data: Union[Sequence[T], pyspark.rdd.RDD],
    estimator: ParameterEstimator,
    prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None
) -> SequenceEncodableProbabilityDistribution:
    """Perform E-step in EM algorithm by iterating over all observations in 'data'.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'estimator' and/or 'prev_estimate'.
        estimator (ParameterEstimator): Model to be estimated from 'data'.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution], optional): Previous estimate of EM algorithm. Must
            be included for distributions that require initialization. Defaults to None.

    Returns:
        SequenceEncodableProbabilityDistribution: Next iteration of EM algorithm.
    """
    if isinstance(data, pyspark.rdd.RDD):
        sc = data.context
        factory = estimator.accumulator_factory()
        estimator_broadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimate_b = sc.broadcast(temp_estimate)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            loc_prev_estimate = pickle.loads(temp_estimate_b.value)

            for x in itr:
                counts_for_split = counts_for_split + 1.0
                accumulator_for_split.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for x in idata:
            nobs += 1.0
            accumulator.update(x, 1.0, estimate=prev_estimate)

        return estimator.estimate(nobs, accumulator.value())


def seq_encode(
    data: Union[Sequence[T], pyspark.rdd.RDD],
    encoder: Optional[DataSequenceEncoder] = None,
    estimator: Optional[ParameterEstimator] = None,
    model: Optional[SequenceEncodableProbabilityDistribution] = None,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None
) -> Union['pyspark.rdd.RDD', List[Tuple[int, EncodedDataSequence]]]:
    """Sequence encode a sequence of iid observations from a distribution corresponding to 'encoder'.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'encoder'.
        encoder (Optional[DataSequenceEncoder], optional): A DataSequenceEncoder object for sequence encoding iid sequences.
        estimator (Optional[ParameterEstimator], optional): An estimator to create DataSequenceEncoder from.
        model (Optional[SequenceEncodableProbabilityDistribution], optional): A distribution to create DataSequenceEncoder from.
        num_chunks (int, optional): Number of chunks to split the data into. Defaults to 1.
        chunk_size (Optional[int], optional): Approximate size of chunks to determine num_chunks above.

    Returns:
        Union[pyspark.rdd.RDD, List[Tuple[int, EncodedDataSequence]]]: Encoded data.
    """
    if encoder is None:
        if model is not None:
            encoder = model.dist_to_encoder()
        elif estimator is not None:
            encoder = estimator.accumulator_factory().make().acc_to_encoder()
        else:
            raise Exception('At least one arg: encoder, estimator, or dist must be passed.')

    if isinstance(data, pyspark.rdd.RDD):
        sc = data.context
        temp_encoder = pickle.dumps(encoder, protocol=0)
        encoder_broadcast = sc.broadcast(temp_encoder)

        enc_data = (
            data.glom()
            .map(lambda x: list(x))
            .map(lambda x: (len(x), pickle.loads(encoder_broadcast.value).seq_encode(x)))
        )

        return enc_data

    else:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        rv = []
        for i in range(num_chunks_loc):
            data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
            enc_data = encoder.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(enc_data: Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD'],
                        estimate: SequenceEncodableProbabilityDistribution) -> Tuple[float, float]:
    """Vectorized evaluation of the sum of log_density values for a given SequenceEncodableProbabilityDistribution
        over encoded data.

    Notes:
        Returns a Tuple containing the sum of all observations in enc_data, and the sum of the log_density evaluated at
        all encoded data observations in enc_data. This is a fully vectorized evaluation.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations. Must
            be consistent with enc_data.

    Returns:
        Tuple[float, float]

    """
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context
        estimate_broadcast = sc.broadcast(pickle.dumps(estimate, protocol=0))

        def acc(itr):

            rv = 0.0
            cnt = 0.0

            estimate_loc = pickle.loads(estimate_broadcast.value)

            for sz, x in itr:
                rv += estimate_loc.seq_log_density(x).sum()
                cnt += sz

            return [(cnt, rv)]

        return enc_data.mapPartitions(acc).reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        )

    else:

        return sum([u[0] for u in enc_data]), sum(
            [estimate.seq_log_density(u[1]).sum() for u in enc_data]
        )


def seq_log_density(enc_data: Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD'],
                    estimate: Union[Sequence[SequenceEncodableProbabilityDistribution],
                                     SequenceEncodableProbabilityDistribution]) -> List[np.ndarray]:
    """Vectorized evaluation of 'estimate' log-density for each observation in enc_data.

    Notes:
        If 'estimate' is input as a List of numpy arrays. Each list entry corresponds to the seq_log_density calls of all
        the encoded data for each List entry of estimate.

        If 'estimate' is a single SequenceEncodableProbabilityDistribution instance. The log_density of every observation
        in the 'enc_data' data set is returned as a list.

        E = Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD']

    Args:
        enc_data (E): Sequence encoded data of format matching output of seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations.

    Returns:
        Union[List[np.ndarray[float]], List[float]]

    """
    is_list = issubclass(type(estimate), Sequence)

    if isinstance(enc_data, pyspark.rdd.RDD):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimate_broadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimate_broadcast.value)
            if is_list:
                return [
                    np.asarray([ee.seq_log_density(x) for ee in loc_estimate])
                    for sz, x in itr
                ]
            else:
                return [loc_estimate.seq_log_density(x) for sz, x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:

        if is_list:
            return [
                np.asarray([ee.seq_log_density(u[1]) for ee in estimate])
                for u in enc_data
            ]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]


def seq_estimate(enc_data: Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD'],
                 estimator: ParameterEstimator,
                 prev_estimate: T_D) -> T_D:
    """Perform vectorized E-step in EM algorithm for encoded sequence of observations in 'enc_data'.

    Notes:
        Arg estimator must be consistent with prev_estimate. That is, prev_estimate must be an estimate that could be
        obtained from estimator.

        Arg enc_data must type consistent with estimator and prev_estimate (result of seq_encode() call).

        Returns the next iteration of EM algorithm with vectorized calls to "seq_update()" of the corresponding
        SequenceEncodableStatsiticAccumulator objects.

        E = Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD']

    Args:
        enc_data (E): Sequence encoded data of format matching output of seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        prev_estimate (SequenceEncodableProbabilityDistribution): Previous estimate of EM algorithm.

    Returns:
        SequenceEncodableProbabilityDistribution

    """
    if isinstance(enc_data, pyspark.rdd.RDD):
        sc = enc_data.context

        estimator_broadcast = sc.broadcast(estimator)
        estimate_broadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            local_estimate = pickle.loads(estimate_broadcast.value)

            for sz, x in itr:
                counts_for_split = counts_for_split + sz
                accumulator_for_split.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)

            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        estimate_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


def seq_initialize(enc_data: Union[List[Tuple[int, T]], 'pyspark.rdd.RDD'],
                   estimator: ParameterEstimator,
                   rng: np.random.RandomState,
                   p: float = 0.1) -> 'SequenceEncodableProbabilityDistribution':
    """Vectorized initialization of a model corresponding to ParameterEstimator for encoded sequences of iid data
        observations.

    Notes:
        Arg enc_data must type consistent with estimator (result of seq_encode() call).
        Arg estimator must be of data type consistent with encoded sequence data type in 'enc_data'.

        Vectorized initialization of SequenceEncodableProbabilityDistribution corresponding to 'estimator' from enc_data.
        Observations in the encoded sequence enc_data are kept with probability p.

        This functions relies on calls to SequenceEncodableStatisticAccumulator.seq_initialize(), which is a vectorized
        initialization of the SequenceEncodableStatisticAccumulator object.

        This method should produce the same initialized model as a call to initialize() if the data sets are the same.

        E = Union[List[Tuple[int, EncodedDataSequence]], 'pyspark.rdd.RDD']

    Args:
        enc_data (E): Sequence encoded data of format matching output of seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        rng (RandomState): RandomState object for setting seed.
        p (float): Proportion of data to randomly sample for initializing model.

    Returns:
        SequenceEncodableProbabilityDistribution

    """

    if isinstance(enc_data, pyspark.rdd.RDD):
        sc = enc_data.context
        num_partitions = enc_data.getNumPartitions()
        seeds = rng.randint(2 ** 31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(pickle.dumps(seeds, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_loc_w = np.random.RandomState(seed=rng_loc.randint(2 ** 31))

            for sz, x in itr:
                w = np.zeros(sz, dtype=float)
                w_1 = rng_loc_w.rand(sz) <= p
                w[w_1] = 1.0

                counts_for_split += np.sum(w)
                accumulator_for_split.seq_initialize(x, w, rng_loc)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        seeds_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31-1))

        for sz, enc_x in enc_data:
            w = rng_w.binomial(n=1, p=p, size=sz).astype(dtype=np.float64)
            accumulator.seq_initialize(enc_x, w, rng)
            nobs += sz

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

