__all__ = ['BernoulliDistribution', 'BernoulliEstimator', 'BernoulliSampler',
           'BernoulliSetDistribution', 'BernoulliSetEstimator', 'BernoulliSetSampler',
           'BetaDistribution', 'BetaSampler',
           'CategoricalDistribution', 'CategoricalEstimator', 'CategoricalSampler',
           'CompositeDistribution', 'CompositeEstimator', 'CompositeSampler',
           'DiagonalGaussianDistribution', 'DiagonalGaussianEstimator', 'DiagonalGaussianSampler',
           'DictDirichletDistribution',
           'DirichletDistribution', 'DirichletEstimator', 'DirichletSampler',
           'DirichletProcessMixtureDistribution', 'DirichletProcessMixtureEstimator', 'DirichletProcessMixtureSampler',
           'ExponentialDistribution', 'ExponentialEstimator', 'ExponentialSampler',
           'GaussianDistribution', 'GaussianEstimator', 'GaussianSampler',
           'GammaDistribution', 'GammaEstimator', 'GammaSampler',
           'GeometricDistribution', 'GeometricEstimator', 'GeometricSampler',
           'IgnoredDistribution', 'IgnoredEstimator', 'IgnoredSampler',
           'IntegerCategoricalDistribution', 'IntegerCategoricalEstimator', 'IntegerCategoricalSampler',
           'MixtureDistribution', 'MixtureEstimator', 'MixtureSampler',
           'MultivariateNormalGammaDistribution', 'MultivariateNormalGammaSampler',
           'NullDistribution', 'NullEstimator', 'NullSampler',
           'OptionalDistribution', 'OptionalEstimator', 'OptionalSampler',
           'PoissonDistribution', 'PoissonEstimator', 'PoissonSampler',
           'SequenceDistribution', 'SequenceEstimator', 'SequenceSampler',
           'estimate', 'seq_estimate', 'initialize', 'seq_log_density_sum', 'seq_encode', 'seq_log_density']

from typing import Any, Tuple, Sequence, Dict, Optional, Union, List

from dmx.arithmetic import *

from dmx.bstats.pdist import ParameterEstimator, DataSequenceEncoder, EncodedDataSequence, ProbabilityDistribution

from dmx.bstats.beta         import BetaDistribution, BetaSampler
from dmx.bstats.bernoulli    import BernoulliDistribution, BernoulliEstimator, BernoulliSampler
from dmx.bstats.categorical  import CategoricalDistribution, CategoricalEstimator, CategoricalSampler
from dmx.bstats.composite    import CompositeDistribution, CompositeEstimator, CompositeSampler
from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.dirichlet    import DirichletDistribution, DirichletEstimator, DirichletSampler
from dmx.bstats.dmvn         import DiagonalGaussianDistribution, DiagonalGaussianEstimator, DiagonalGaussianSampler
from dmx.bstats.exponential  import ExponentialDistribution, ExponentialEstimator, ExponentialSampler
from dmx.bstats.gaussian     import GaussianDistribution, GaussianEstimator, GaussianSampler
from dmx.bstats.gamma        import GammaDistribution, GammaEstimator, GammaSampler
from dmx.bstats.geometric    import GeometricDistribution, GeometricEstimator, GeometricSampler
from dmx.bstats.ignored      import IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from dmx.bstats.intrange     import IntegerCategoricalDistribution, IntegerCategoricalEstimator, IntegerCategoricalSampler
from dmx.bstats.mixture      import MixtureDistribution, MixtureEstimator, MixtureSampler
from dmx.bstats.mvngamma     import MultivariateNormalGammaDistribution, MultivariateNormalGammaSampler
from dmx.bstats.nulldist     import NullDistribution, NullEstimator, NullSampler
from dmx.bstats.optional     import OptionalDistribution, OptionalEstimator, OptionalSampler
from dmx.bstats.poisson      import PoissonDistribution, PoissonEstimator, PoissonSampler
from dmx.bstats.sequence     import SequenceDistribution, SequenceEstimator, SequenceSampler
from dmx.bstats.setdist      import BernoulliSetDistribution, BernoulliSetEstimator, BernoulliSetSampler

from dmx.bstats.dpm import DirichletProcessMixtureDistribution, DirichletProcessMixtureEstimator, DirichletProcessMixtureSampler


import numpy as np
from numpy.random import RandomState
from pyspark import RDD
import pickle
import pandas as pd

def load_models(x: str) -> ProbabilityDistribution:
    """Read in ProbabilityDistribution from string representaiton.

    Args:
        x (str): String representation of ProbabilityDistribution.

    Returns:
        ProbabilityDistribution
    """
    return eval(x)

def dump_models(x: ProbabilityDistribution) -> str:
    """Serialize Probability Distribution. 

    Args:
        x (ProbabilityDistribution): distribution to serialize.

    Returns:
        str
    """
    return str(x)



def _local_estimate(
    data: Sequence[Any],
    estimator: ParameterEstimator,
    prev_estimate: Optional[ProbabilityDistribution] = None
) -> ProbabilityDistribution:
    """
    Perform a local estimation of parameters using the provided data and estimator.

    Args:
        data (Sequence[Any]): The input sequence of data points to be used for estimation.
        estimator (ParameterEstimator): The estimator object that provides methods for creating accumulators 
            and estimating parameters.
        prev_estimate (Optional[ProbabilityDistribution]): An optional previous probability distribution 
            estimate to guide the current estimation process. Defaults to None.

    Returns:
        Any: The result of the estimation process, as determined by the estimator's `estimate` method.
    """
    idata = iter(data)
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0

    for x in idata:
        nobs += 1.0
        accumulator.update(x, 1.0, estimate=prev_estimate)

    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)

    return estimator.estimate(accumulator.value())


def estimate(
    data: Union[RDD, pd.DataFrame, Sequence[Any]],
    estimator: ParameterEstimator,
    prev_estimate: Optional[ProbabilityDistribution] = None
) -> ProbabilityDistribution:
    """
    Estimate parameters based on the input data, using the given estimator and optional previous estimate.

    This function supports data in multiple formats, including PySpark RDDs, Pandas DataFrames, and iterables.

    Args:
        data (Union[RDD, pd.DataFrame, Sequence[Any]]): The input data for estimation. It can be:
            - A PySpark RDD (`pyspark.rdd.RDD`).
            - A Pandas DataFrame (`pandas.core.frame.DataFrame`).
            - Any iterable object.
        estimator (ParameterEstimator): The estimator object that provides methods for creating accumulators 
            and estimating parameters.
        prev_estimate (Optional[ProbabilityDistribution]): An optional previous probability distribution 
            estimate to guide the current estimation process. Defaults to None.

    Returns:
        ProbabilityDistribution: Variational inference update
    """
    if 'pyspark.rdd' in str(type(data)):
        sc = data.context
        factory = estimator.accumulatorFactory()
        estimatorBroadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimateB = sc.broadcast(temp_estimate)

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = 0.0
            loc_prev_estimate = pickle.loads(temp_estimateB.value)

            for x in itr:
                countsForSplit += 1.0
                accumulatorForSplit.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs += nobsForSplit
            accumulator.combine(statsForSplit)

        return estimator.estimate(nobs, accumulator.value())

    elif 'pandas.core.frame.DataFrame' in str(type(data)):
        accumulator = estimator.accumulatorFactory().make()
        accumulator.df_update(data, np.ones(len(data)), estimate=prev_estimate)
        return estimator.estimate(None, accumulator.value())

    elif hasattr(data, '__iter__'):
        return _local_estimate(data, estimator, prev_estimate)


def initialize(
    data: Union[Sequence[Any], RDD, pd.DataFrame],
    estimator: ParameterEstimator,
    rng: RandomState,
    p: float
) -> Any:
    """
    Initialize parameters based on the input data, using the given estimator, random state, and sampling probability.

    This function supports data in multiple formats, including PySpark RDDs, Pandas DataFrames, and iterables.

    Args:
        data (Union[Sequence[Any], RDD, pd.DataFrame]): The input data for initialization. It can be:
            - A PySpark RDD (`pyspark.rdd.RDD`).
            - A Pandas DataFrame (`pandas.core.frame.DataFrame`).
            - Any iterable object.
        estimator (ParameterEstimator): The estimator object that provides methods for creating accumulators 
            and estimating parameters.
        rng (RandomState): A NumPy random state object used for random sampling.
        p (float): The probability for random sampling.

    Returns:
        Any: The result of the initialization process, as determined by the estimator's `estimate` method.
    """
    if 'pyspark.rdd' in str(type(data)):
        factory = estimator.accumulator_factory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(np.iinfo(np.int32).max, size=num_partitions)

        estimatorBroadcast = sc.broadcast(estimator)
        seedsBroadcast = sc.broadcast(seeds)

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulator_factory().make()
            countsForSplit = 0.0
            rng_loc = np.random.RandomState(seedsBroadcast.value[splitIndex])

            for x in itr:
                w = 1.0 if rng_loc.rand() <= p else 0.0
                countsForSplit += w
                accumulatorForSplit.initialize(x, w, rng_loc)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs += nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

    elif 'pandas.core.frame.DataFrame' in str(type(data)):
        accumulator = estimator.accumulator_factory().make()
        accumulator.df_initialize(data, rng.rand(len(data)) * p, rng)
        return estimator.estimate(None, accumulator.value())

    elif hasattr(data, '__iter__'):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for x in idata:
            w = 1.0 if rng.rand() <= p else 0.0
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)
  
        return estimator.estimate(accumulator.value())
    

def seq_encode(
    data: Union[Sequence[Any], RDD],
    model: ProbabilityDistribution,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None
) -> Union[RDD, List[Tuple[int, Any]]]:
    """
    Encodes a sequence of data using a given probability distribution model.

    Args:
        data (Union[Sequence[Any], RDD]): The input data to encode. This can be a standard Python sequence 
            or a PySpark RDD.
        model (ProbabilityDistribution): The model used for encoding the data. It must implement a `seq_encode` method.
        num_chunks (int, optional): The number of chunks to divide the data into for encoding. Defaults to 1.
        chunk_size (Optional[int], optional): The size of each chunk. If provided, it overrides `num_chunks`. Defaults to None.

    Returns:
        Union[RDD, List[Tuple[int, Any]]]: 
            - If `data` is a PySpark RDD, returns an RDD where each element is a tuple containing the length of the chunk 
              and the encoded data.
            - If `data` is a sequence, returns a list of tuples, where each tuple contains the length of the chunk 
              and the encoded data.
    """
    if 'pyspark.rdd' in str(type(data)):
        sc = data.context

        temp_model = pickle.dumps(model, protocol=0)
        modelBroadcast = sc.broadcast(temp_model)

        enc_data = data.glom().map(lambda x: list(x)).map(
            lambda x: (len(x), pickle.loads(modelBroadcast.value).seq_encode(x))
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
            data_loc = [data[j] for j in range(i, sz, num_chunks_loc)]
            enc_data = model.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv

def seq_estimate(
    enc_data: Union[RDD, Sequence[tuple[int, Any]]],
    estimator: ParameterEstimator,
    prev_estimate: Any
) -> Any:
    """
    Sequentially estimate parameters based on encoded data, using the given estimator and previous estimate.

    This function supports data in multiple formats, including PySpark RDDs and sequences of tuples.

    Args:
        enc_data (Union[RDD, Sequence[tuple[int, Any]]]): The encoded data for estimation. It can be:
            - A PySpark RDD (`pyspark.rdd.RDD`) containing tuples of size and data.
            - A sequence of tuples, where each tuple contains an integer size and associated data.
        estimator (ParameterEstimator): The estimator object that provides methods for creating accumulators 
            and estimating parameters.
        prev_estimate (Any): The previous estimate to guide the current estimation process.

    Returns:
        Any: The result of the sequential estimation process, as determined by the estimator's `estimate` method.
    """

    if 'pyspark.rdd' in str(type(enc_data)):
        sc = enc_data.context

        estimatorBroadcast = sc.broadcast(estimator)
        estimateBroadcast  = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = zero
            local_estimate = pickle.loads(estimateBroadcast.value)

            for sz, x in itr:
                countsForSplit = countsForSplit + sz
                accumulatorForSplit.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((countsForSplit, accumulatorForSplit.value()), protocol=0)
            #return [(countsForSplit, accumulatorForSplit.value())]
            return [rv]

        def red(x, y):
            """Reduce function to combine accros partitions."""
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimatorBroadcast.value.accumulatorFactory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))
            #return (nobs, vals)
            return rv


        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = zero
        accumulator = estimator.accumulatorFactory().make()

        for stuff in temp.collect():
            nobsForSplit, statsForSplit = pickle.loads(stuff)
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)


        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)


        estimateBroadcast.destroy()
        estimatorBroadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return(estimator.estimate(nobs, accumulator.value()))

    else:

        accumulator = estimator.accumulator_factory().make()
        nobs        = 0.0

        data_update = []

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)
            #x_update = accumulator.seq_update(x, np.ones(sz), prev_estimate)
            #data_update.append((sz, x_update))

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(accumulator.value())


def seq_log_density(
    enc_data: Union[RDD, Sequence[Tuple[int, EncodedDataSequence]]],
    estimate: Union[ProbabilityDistribution, List[ProbabilityDistribution]],
    is_list: bool = False
) -> List[np.ndarray]:
    """
    Compute the sequential log density for encoded data using the given estimate.

    This function supports data in multiple formats, including PySpark RDDs and sequences of tuples.

    Args:
        enc_data (Union[RDD, Sequence[Tuple[int, EncodedDataSequence]]]): The encoded data for density computation. It can be:
            - A PySpark RDD (`pyspark.rdd.RDD`) containing tuples of size and data.
            - A sequence of tuples, where each tuple contains an integer size and associated EncodedDataSequence.
        estimate (Union[ProbabilityDistribution, List[ProbabilityDistribution]]): The estimate object or a list of estimates used for log density computation.
        is_list (bool): Whether the `estimate` is a list of estimates. Defaults to `False`.

    Returns:
        List[np.ndarray]: A list of log density values computed for the encoded data.
    """

    if 'pyspark.rdd' in str(type(enc_data)):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimateBroadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimateBroadcast.value)
            if is_list:
                return [np.asarray([ee.seq_log_density(x) for ee in loc_estimate]) for sz, x in itr]
            else:
                return [loc_estimate.seq_log_density(x) for sz,x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:

        if is_list:
            return [np.asarray([ee.seq_log_density(u[1]) for ee in estimate]) for u in enc_data]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]



def seq_log_density_sum(
    enc_data: Union[RDD, Sequence[Tuple[int, EncodedDataSequence]]],
    estimate: ProbabilityDistribution
) -> Tuple[float, float]:
    """
    Compute the sum of sequential log densities for encoded data using the given estimate.

    This function supports data in multiple formats, including PySpark RDDs and sequences of tuples.

    Args:
        enc_data (Union[RDD, Sequence[Tuple[int, EncodedDataSequence]]]): The encoded data for density computation. 
        It can be:
            - A PySpark RDD (`pyspark.rdd.RDD`) containing tuples of size and data.
            - A sequence of tuples, where each tuple contains an integer size and associated data.
        estimate (ProbabilityDistribution): The ProbabilityDistribution used for log density computation.

    Returns:
        Tuple[float, float]: A tuple containing:
            - The total count of observations (`cnt`).
            - The sum of log densities (`rv`).
    """

    if 'pyspark.rdd' in str(type(enc_data)):
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

        return enc_data.mapPartitions(acc).reduce(lambda a,b : (a[0]+b[0], a[1]+b[1]))

    else:

        return sum([u[0] for u in enc_data]), sum([estimate.seq_log_density(u[1]).sum() for u in enc_data])