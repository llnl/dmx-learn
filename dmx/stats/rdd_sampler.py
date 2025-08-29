from pyspark import SparkContext
from numpy.random import RandomState
from dmx.arithmetic import *
import numpy as np
import pickle
from dmx.arithmetic import maxrandint
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution
from typing import Optional


def take_sample(
    rdd: SparkContext.parallelize,
    with_replacement: bool,
    n: int,
    seed: Optional[int] = None
) -> list:
    """
    Takes a sample from an RDD.

    Args:
        rdd (SparkContext.parallelize): The input RDD to sample from.
        with_replacement (bool): Whether sampling is done with replacement.
        n (int): Number of samples to take.
        seed (Optional[int]): Seed for random number generation. Defaults to None.

    Returns:
        list: A list of samples from the RDD.
    """
    rng = RandomState(seed)
    sample = rdd.zipWithUniqueId().takeSample(with_replacement, n, rng.randint(0, maxrandint))
    sidx = np.argsort([u[1] for u in sample])
    sample = [sample[i][0] for i in sidx]
    sidx = np.argsort(rng.uniform(size=n))
    return [sample[i] for i in sidx]


def sample_seq_as_rdd(
    sc: SparkContext,
    dist: SequenceEncodableProbabilityDistribution,
    seq_len: int,
    count_per_split: int,
    num_splits: int,
    seed: Optional[int] = None
) -> SparkContext.parallelize:
    """
    Samples sequences from a distribution and returns them as an RDD.

    Args:
        sc (SparkContext): The Spark context.
        dist (SequenceEncodableProbabilityDistribution): The distribution object providing the sampler.
        seq_len (int): Length of each sequence to sample.
        count_per_split (int): Number of samples per split.
        num_splits (int): Number of splits in the RDD.
        seed (Optional[int]): Seed for random number generation. Defaults to None.

    Returns:
        SparkContext.parallelize: An RDD containing sampled sequences.
    """
    distB = sc.broadcast(dist)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = distB.value
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample_seq(seq_len, size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)


def sample_rdd(
    sc: SparkContext,
    dist: SequenceEncodableProbabilityDistribution,
    count_per_split: int,
    num_splits: int,
    seed: Optional[int] = None
) -> SparkContext.parallelize:
    """
    Samples data from a distribution and returns it as an RDD.

    Args:
        sc (SparkContext): The Spark context.
        dist (SequenceEncodableProbabilityDistribution): The distribution object providing the sampler.
        count_per_split (int): Number of samples per split.
        num_splits (int): Number of splits in the RDD.
        seed (Optional[int]): Seed for random number generation. Defaults to None.

    Returns:
        SparkContext.parallelize: An RDD containing sampled data.
    """
    dd = pickle.dumps(dist, protocol=0)
    distB = sc.broadcast(dd)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = pickle.loads(distB.value)
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample(size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)
