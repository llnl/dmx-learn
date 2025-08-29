"""Contains estimation tools for bstats with mpi4py use."""
__all__ = [
    'seq_estimate_mpi', 
    'initialize_mpi', 
    'seq_log_density_sum_mpi', 
    'seq_encode_mpi', 
    'seq_log_density_mpi'
    ]

from typing import Any, Tuple, Sequence, Optional, List
import numpy as np
from numpy.random import RandomState
from mpi4py import MPI
from dmx.bstats import *
from dmx.arithmetic import *
from dmx.bstats.pdist import ParameterEstimator, EncodedDataSequence, ProbabilityDistribution


def seq_encode_mpi(
    data: Sequence[Any],
    model: ProbabilityDistribution,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None
) -> List[Tuple[int, Any]]:
    """
    Encode data sequentially using MPI for parallel processing.

    This function distributes data across MPI processes, performs encoding on each process, 
    and collects the encoded results.

    Args:
        data (Sequence[Any]): The input data to be encoded.
        model (ProbabilityDistribution): The model object that provides the `seq_encode` method for encoding.
        num_chunks (int, optional): The number of chunks to divide the data into. Defaults to 1.
        chunk_size (Optional[int], optional): The size of each chunk. If provided, it overrides `num_chunks`.

    Returns:
        List[Tuple[int, Any]]: A list of tuples, where each tuple contains:
            - The size of the data chunk.
            - The encoded data for that chunk.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        # Distribute data across processes
        data_loc = [[data[i] for i in range(r, sz, world_size)] for r in range(world_size)]
        model_loc = model
    else:
        num_chunks_loc = None
        data_loc = None
        model_loc = None

    # Broadcast the model and number of chunks to all processes
    model_loc = comm.bcast(model_loc, root=0)
    data_loc = comm.scatter(data_loc, root=0)
    num_chunks_loc = comm.bcast(num_chunks_loc, root=0)

    nn = len(data_loc)
    rv = []

    # Process data chunks locally
    for i in range(num_chunks_loc):
        data_chunk = [data_loc[j] for j in range(i, nn, num_chunks_loc)]
        enc_data = model_loc.seq_encode(data_chunk)
        rv.append((len(data_chunk), enc_data))

    return rv


def initialize_mpi(data: Sequence[Any], estimator: ParameterEstimator, rng: RandomState, p: float) -> Optional[ProbabilityDistribution]:
    """
    Initialize MPI-based parallel data processing and estimate parameters.

    This function distributes data processing across MPI processes, computes
    sufficient statistics locally, and combines them at the root process to
    estimate parameters using the provided estimator.

    Args:
        data (Sequence[Any]): The input data to be processed.
        estimator (ParameterEstimator): The estimator object for parameter estimation.
        rng (RandomState): The random number generator for sampling.
        p (float): The probability threshold for weighting observations.

    Returns:
        Optional[ProbabilityDistribution]: The estimated model if called from the root process (rank 0),
        otherwise `None`.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        # factory = estimator.accumulator_factory()
        est = estimator
        seeds = rng.randint(low=0, high=2**31-1, size=world_size).tolist()
        data_loc = [[data[i] for i in range(r, len(data), world_size)] for r in range(world_size)]

    else:
        seeds = None
        est = None
        data_loc = None
    
    seed = comm.scatter(seeds, root=0)
    est = comm.bcast(est, root=0) # this should be factory cast
    factory = est.accumulator_factory()
    data_loc = comm.scatter(data_loc, root=0)
    rng_loc = np.random.RandomState(seed)

    idata = iter(data_loc)
    local_accumulator = factory.make() 
    nobs = 0.0

    for x in idata:
        w = 1.0 if rng_loc.rand() <= p else 0.0
        nobs += w
        local_accumulator.initialize(x, w, rng)


    stats_dict = dict()
    local_accumulator.key_merge(stats_dict)
    local_accumulator.key_replace(stats_dict)
    suff_stats = comm.gather((nobs, local_accumulator.value()), root=0)

    if world_rank == 0:
        total_obs = 0.0
        accumulator = factory.make()
        for nn, ss in suff_stats:
            accumulator.combine(ss)
            total_obs += nn

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(accumulator.value())

    else:
        return None


def seq_estimate_mpi(
    enc_data: List[Tuple[int, EncodedDataSequence]],
    estimator: ParameterEstimator,
    prev_estimate: ProbabilityDistribution
) -> Optional[ProbabilityDistribution]:
    """
    Estimate parameters using MPI-based parallel processing.

    This function distributes encoded data across MPI processes, updates accumulators
    locally, and combines sufficient statistics at the root process to compute the final
    estimate.

    Args:
        enc_data (List[Tuple[int, EncodedDataSequence]]): Encoded data, where each tuple contains:
            - The size of the data chunk (`int`).
            - The encoded data (`EncodedDataSequence`).
        estimator (ParameterEstimator): The estimator object for parameter estimation.
        prev_estimate (ProbabilityDistribution): The previous estimate to be used during updates.

    Returns:
        Optional[ProbabilityDistribution]: The estimated model if called from the root process (rank 0),
        otherwise `None`.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        est = estimator
        
    else:
        est = None
        prev_estimate = None
    
    est = comm.bcast(est, root=0)
    factory = est.accumulator_factory()
    prev_estimate = comm.bcast(prev_estimate, root=0)
    local_accumulator = factory.make()
    nobs = 0.0

    for sz, x in enc_data:
        nobs += sz
        local_accumulator.seq_update(x, np.ones(sz), prev_estimate)
    
    suff_stats = comm.gather((nobs, local_accumulator.value()), root=0)
    if world_rank == 0:
        total_obs = 0.0
        accumulator = factory.make()
        for nn, ss in suff_stats:
            total_obs += nn
            accumulator.combine(ss)
        
        stats_dict = ()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(accumulator.value())
    else:
        return None



def seq_log_density_mpi(
    enc_data: Sequence[Tuple[int, EncodedDataSequence]],
    estimate: ProbabilityDistribution,
    is_list: bool = False
) -> List[np.ndarray]:
    """
    Compute log densities for encoded data sequences in parallel using MPI.

    Args:
        enc_data (Sequence[Tuple[int, EncodedDataSequence]]): Encoded data sequences, where
            each tuple contains:
            - The size of the data chunk (`int`).
            - The encoded data sequence (`EncodedDataSequence`).
        estimate (ProbabilityDistribution): The probability distribution object used to compute log densities.
        is_list (bool, optional): Whether to compute log densities for multiple probability distributions.
            Defaults to `False`.

    Returns:
        List[np.ndarray]: A list of log densities for the encoded data sequences.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:
        loc_estimate = estimate
    else:
        loc_estimate = None
    loc_estimate = comm.bcast(loc_estimate, root=0)
    if is_list:
        return [np.asarray([ee.seq_log_density(u[1]) for ee in loc_estimate]) for u in enc_data]
    else:
        return [loc_estimate.seq_log_density(u[1]) for u in enc_data]



def seq_log_density_sum_mpi(
    enc_data: Sequence[Tuple[int, EncodedDataSequence]],
    estimate: ProbabilityDistribution
) -> Tuple[int, float]:
    """
    Compute the total number of observations and the sum of log densities in parallel using MPI.

    Args:
        enc_data (Sequence[Tuple[int, EncodedDataSequence]]): Encoded data sequences, where
            each tuple contains:
            - The size of the data chunk (`int`).
            - The encoded data sequence (`EncodedDataSequence`).
        estimate (ProbabilityDistribution): The probability distribution object used to compute log densities.

    Returns:
        Tuple[int, float]: A tuple containing:
            - `nobs` (int): The total number of observations.
            - `ll` (float): The sum of log densities for all encoded data sequences.
    """
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    if world_rank == 0:
        loc_estimate = estimate
    else:
        loc_estimate = None
    loc_estimate = comm.bcast(loc_estimate, root=0)
    rv0 = sum([u[0] for u in enc_data])
    rv1 = sum([loc_estimate.seq_log_density(u[1]).sum() for u in enc_data])

    nobs = comm.allreduce(rv0)
    ll = comm.allreduce(rv1)
    
    return nobs, ll
