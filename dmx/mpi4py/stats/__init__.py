"""
MPI-based sequence encoding and estimation utilities.

This module provides functions for distributed sequence encoding, parameter
initialization, estimation, and log density computation using MPI. It is
designed to work with sequence-encodable probability distributions and
related estimators/encoders from the dml.stats.pdist module.

Functions:
    seq_encode_mpi: Distributes and encodes sequence data across MPI workers.
    seq_initialize_mpi: Initializes model parameters in parallel.
    seq_estimate_mpi: Estimates model parameters in parallel.
    seq_log_density_mpi: Computes log densities of encoded data in parallel.
    seq_log_density_sum_mpi: Computes sum of log densities and total observations in parallel.
"""

__all__ = [
    "seq_encode_mpi", 
    "seq_initialize_mpi", 
    "seq_estimate_mpi", 
    "seq_log_density_mpi", 
    "seq_log_density_sum_mpi",
]

from typing import Optional, Sequence, List, Tuple, Any
import numpy as np
import pandas as pd
from numpy.random import RandomState
from mpi4py import MPI

from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DataSequenceEncoder, \
    EncodedDataSequence


def seq_encode_mpi(
    data: Optional[Sequence[Any]],
    encoder: Optional[DataSequenceEncoder] = None,
    estimator: Optional[ParameterEstimator] = None,
    model: Optional[SequenceEncodableProbabilityDistribution] = None,
    num_chunks: int = 1, 
    chunk_size: Optional[int] = None
) -> List[Tuple[int, Any]]:
    """Distributes and encodes sequence data across MPI workers.

    Args:
        data (Optional[Sequence[Any]]): The sequence data to encode.
        encoder (Optional[DataSequenceEncoder], optional): Encoder to use for encoding data.
        estimator (Optional[ParameterEstimator], optional): Estimator to derive encoder if not provided.
        model (Optional[SequenceEncodableProbabilityDistribution], optional): Model to derive encoder if not provided.
        num_chunks (int, optional): Number of data chunks to split for parallel processing. Defaults to 1.
        chunk_size (Optional[int], optional): Size of each data chunk. If specified, overrides num_chunks.

    Returns:
        List[Tuple[int, Any]]: List of tuples containing chunk size and encoded data for each chunk.

    Raises:
        Exception: If neither encoder, estimator, nor model is provided on rank 0.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        # world rank 0 needs at least one of the following
        if encoder is None:
            if model is not None:
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
            else:
                raise Exception('At least one arg: encoder, estimator, or dist must be passed to rank 0.')

        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        # Distribute data across processes
        data_loc = [[data[i] for i in range(r, sz, world_size)] for r in range(world_size)]
        encoder_loc = encoder
    else:
        num_chunks_loc = None
        data_loc = None
        encoder_loc = None

    # Broadcast the model and number of chunks to all processes
    encoder_loc = comm.bcast(encoder_loc, root=0)
    data_loc = comm.scatter(data_loc, root=0)
    num_chunks_loc = comm.bcast(num_chunks_loc, root=0)

    nn = len(data_loc)
    rv = []

    # Process data chunks locally and return on worker
    for i in range(num_chunks_loc):
        data_chunk = [data_loc[j] for j in range(i, nn, num_chunks_loc)]
        enc_data = encoder_loc.seq_encode(data_chunk)
        rv.append((len(data_chunk), enc_data))

    return rv


def seq_initialize_mpi(
    enc_data: List[Tuple[int, EncodedDataSequence]], 
    estimator: ParameterEstimator, 
    rng: RandomState, 
    p: float
) -> Optional[SequenceEncodableProbabilityDistribution]:
    """Initializes model parameters in parallel using encoded data.

    Args:
        enc_data (List[Tuple[int, EncodedDataSequence]]): List of tuples containing chunk size and encoded data.
        estimator (ParameterEstimator): Parameter estimator for the model.
        rng (RandomState): Random number generator for initialization.
        p (float): Probability for subsampling data during initialization.

    Returns:
        Optional[SequenceEncodableProbabilityDistribution]: Estimated model parameters (on rank 0), None otherwise.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Broadcast StatisticAccumulatorFactory and random seeds 
    if world_rank == 0:
        factory = estimator.accumulator_factory()
        seeds = rng.randint(low=0, high=2**31-1, size=world_size).tolist()
    else:
        seeds = None
        factory = None
    
    seed = comm.scatter(seeds, root=0)
    factory = comm.bcast(factory, root=0)

    # Define local accumulator and initialize locally on chunks
    rng_loc = np.random.RandomState(seed)
    local_accumulator = factory.make() 
    nobs = 0.0

    for sz, x in enc_data:
        w = np.where(rng_loc.rand(sz) <= p, 1.0, 0.0)
        nobs += np.sum(w)
        local_accumulator.seq_initialize(x, w, rng)

    # Merge keys for local accumulators 
    stats_dict = dict()
    local_accumulator.key_merge(stats_dict)
    local_accumulator.key_replace(stats_dict)
    suff_stats = comm.gather((nobs, local_accumulator.value()), root=0)

    # Merge all accumulators on master
    if world_rank == 0:
        total_obs = 0.0
        accumulator = factory.make()
        for nn, ss in suff_stats:
            accumulator.combine(ss)
            total_obs += nn

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(total_obs, accumulator.value())

    else:
        return None


def seq_estimate_mpi(
    enc_data: List[Tuple[int, EncodedDataSequence]], 
    estimator: Optional[ParameterEstimator] = None, 
    prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None
) -> Optional[SequenceEncodableProbabilityDistribution]:
    """Estimates model parameters in parallel using encoded data and a previous estimate.

    Args:
        enc_data (List[Tuple[int, EncodedDataSequence]]): List of tuples containing chunk size and encoded data.
        estimator (Optional[ParameterEstimator], optional): Parameter estimator for the model.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution], optional): Previous model estimate.

    Returns:
        Optional[SequenceEncodableProbabilityDistribution]: Estimated model parameters (on rank 0), None otherwise.

    Raises:
        Exception: If estimator or prev_estimate is not provided on rank 0.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        if estimator is None:
            raise Exception("Rank 0 must have estimator for seq_estimate_mpi.")
        
        if prev_estimate is None:
            raise Exception("Rank 0 must have prev_estimate for seq_estimate_mpi.")

        factory = estimator.accumulator_factory()
        
    else:
        factory = None
    
    # Broadcast prev_estimate and factory
    factory = comm.bcast(factory, root=0)
    prev_estimate = comm.bcast(prev_estimate, root=0)
    local_accumulator = factory.make()
    nobs = 0.0

    # update locally
    for sz, x in enc_data:
        nobs += sz
        local_accumulator.seq_update(x, np.ones(sz), prev_estimate)

    # Merge keys for local accumulators and gather suff_stats to master
    stats_dict = dict()
    local_accumulator.key_merge(stats_dict)
    local_accumulator.key_replace(stats_dict)
    suff_stats = comm.gather((nobs, local_accumulator.value()), root=0)

    # aggregate on master
    if world_rank == 0:
        total_obs = 0.0
        accumulator = factory.make()
        for nn, ss in suff_stats:
            total_obs += nn
            accumulator.combine(ss)
        
        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(total_obs, accumulator.value())
    else:
        return None
    

def seq_log_density_mpi(
    enc_data: Sequence[Tuple[int, EncodedDataSequence]],
    estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
    is_list: bool = False
) -> List[np.ndarray]:
    """Computes log densities of encoded data in parallel.

    Args:
        enc_data (Sequence[Tuple[int, EncodedDataSequence]]): Encoded data to compute log densities for.
        estimate (Optional[SequenceEncodableProbabilityDistribution], optional): Model estimate for log density computation.
        is_list (bool, optional): If True, computes log densities for a list of estimates.

    Returns:
        List[np.ndarray]: List of arrays containing log densities for each data chunk.

    Raises:
        Exception: If estimate is not provided on rank 0.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0 and estimate is None:
        raise Exception("Rank 0 must have estimate for seq_log_density_mpi.")
    
    # broadcast estimate to each worker
    estimate = comm.bcast(estimate, root=0)

    # evaluate likelihood on worker data chunks, might rearrange later for use
    if is_list:
        return [np.asarray([ee.seq_log_density(u[1]) for ee in estimate]) for u in enc_data]
    else:
        return [estimate.seq_log_density(u[1]) for u in enc_data]
    

def seq_log_density_sum_mpi(
    enc_data: Sequence[Tuple[int, EncodedDataSequence]],
    estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
) -> List[np.ndarray]:
    """Computes sum of log densities and total number of observations in parallel.

    Args:
        enc_data (Sequence[Tuple[int, EncodedDataSequence]]): Encoded data to compute log densities for.
        estimate (Optional[SequenceEncodableProbabilityDistribution], optional): Model estimate for log density computation.

    Returns:
        Tuple[int, float]: Total number of observations and sum of log densities across all data.

    Raises:
        Exception: If estimate is not provided on rank 0.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0 and estimate is None:
        raise Exception("Rank 0 must have estimate for seq_log_density_mpi.")
    
    estimate = comm.bcast(estimate, root=0)
    rv0 = sum([u[0] for u in enc_data])
    rv1 = sum([estimate.seq_log_density(u[1]).sum() for u in enc_data])

    # return nobs and ll sum back to all workers.
    nobs = comm.allreduce(rv0)
    ll = comm.allreduce(rv1)
    
    return nobs, ll