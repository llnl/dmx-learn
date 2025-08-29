"""Estimation functions for bstats with mpi4py support."""
import sys
from typing import Any, Optional, Tuple, Sequence, List, IO, TypeVar

import numpy as np
from numpy.random import RandomState
from mpi4py import MPI

from dmx.bstats.pdist import ParameterEstimator, ProbabilityDistribution
from dmx.mpi4py.bstats import * 

DATUM_TYPE = TypeVar('DATUM_TYPE')
ENCODED_SEQ = TypeVar('ENCODED_SEQ')

def optimize_mpi(
          data: Optional[Sequence[DATUM_TYPE]], 
          estimator: ParameterEstimator, 
          max_its: int = 10,
          delta: Optional[float] = 1.0e-9,
          init_estimator: Optional[ParameterEstimator] = None, 
          init_p: float = 0.1,
          rng: RandomState = RandomState(), 
          prev_estimate: Optional[ProbabilityDistribution] = None,
          vdata: Optional[Sequence[DATUM_TYPE]] = None,
          enc_data: Optional[List[Tuple[int, ENCODED_SEQ]]] = None,
          enc_vdata: Optional[List[Tuple[int, ENCODED_SEQ]]] = None,
          out: IO = sys.stdout,
          print_iter: int = 1, 
          num_chunks: int = 1) -> ProbabilityDistribution:
    """Estimation of 'estimator' via EM algorithm for max_its iterations or until
        new_loglikelihood - old_loglikelihood < delta.

    Args:
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator): ParameterEstimator used to specify to-be-estimated distribution for observed
            data.
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            |old_loglikelihood - new_loglikelihood| < delta or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        rng (RandomState): RandomState used to set seed for initializing EM algorithm.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[SeqeuenceEncodableProbabilityDistribution]): Optional model estimate used from prior
            fitting. Must be consistent with estimator.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (IO): IO stream to write out iterations of EM algorithm.
        print_iter (int): Print iterations (i.e. log-likelihood difference) every print_iter-iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # check if encoded data is already on each worker
    enc_data_exists = enc_data is not None
    enc_data_exists_all = comm.allreduce(enc_data_exists, op=MPI.LAND)
    if world_rank == 0:
        data_exception = data is None
    else:
        data_exception = None

    # enc_data_exists_all = comm.bcast(enc_data_exists_all, root=0)
    data_exception = comm.bcast(data_exception, root=0)

    if data_exception and not enc_data_exists_all:
        raise Exception('Optimization called with empty data one rank 0 and encoded data does not exist.')

    est = estimator if init_estimator is None else init_estimator

    if world_rank == 0:
        if prev_estimate:
            # data_encoder = prev_estimate.dist_to_encoder()
            mm = prev_estimate
            skip_init = True
        else:
            # data_encoder = est.accumulator_factory().make().acc_to_encoder()
            mm = None 
            skip_init = False
    else:
        # data_encoder = None
        mm = None
        skip_init = None
    
    # if previous estimate not passed, initialize the model
    skip_init = comm.bcast(skip_init, root=0)
    if not skip_init:
        if init_p <= 0.0:
            p = 0.10
        else:
            p = min(max(init_p, 0.0), 1.0)
        mm = initialize_mpi(data, estimator=est, rng=rng, p=p)

    # the data has not been encoded on all workers yet
    if not enc_data_exists_all:
        enc_data = seq_encode_mpi(data=data, model=mm, num_chunks=num_chunks)

    _, old_ll = seq_log_density_sum_mpi(enc_data=enc_data, estimate=mm)

    # check if validation data is passed
    # check if encoded data is already on each worker
    enc_vdata_exists = enc_vdata is not None
    enc_vdata_exists_all = comm.allreduce(enc_vdata_exists, op=MPI.LAND)
    if world_rank == 0:
        vdata_exists = vdata is not None
    else:
        vdata_exists = None

    vdata_exists = comm.bcast(vdata_exists, root=0)

    if not enc_vdata_exists_all and vdata_exists:
        enc_vdata = seq_encode_mpi(vdata, model=mm, num_chunks=num_chunks)
        enc_vdata_exists_all = True
    
    if enc_vdata_exists_all:
        _, old_vll = seq_log_density_sum_mpi(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate_mpi(enc_data=enc_data, estimator=est, prev_estimate=mm)
        cnt, ll = seq_log_density_sum_mpi(enc_data=enc_data, estimate=mm_next)

        if enc_vdata_exists_all:
            _, vll = seq_log_density_sum_mpi(enc_data=enc_vdata, estimate=mm_next)
        else:
            vll = ll
        
        dll = ll - old_ll

        if (dll >= 0) or (delta is None):
            mm = mm_next

        # converged in delta tolerance
        if (delta is not None) and (dll < delta):
            if world_rank == 0:
                if enc_vdata_exists_all:
                    out.write(
                        'Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, '
                        'ln[p_mat(Valid Data|Model)]=%e\n' % (
                        i + 1, ll, dll, vll))
                else:
                    out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n' %
                            (i + 1, ll, dll))
                    
            break
            
        if world_rank == 0:
            if (i + 1) % print_iter == 0:
                if enc_vdata_exists_all:
                    out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, '
                            'ln[p_mat(Valid Data|Model)]=%e\n' % (i + 1, ll, dll, vll))
                else:
                    out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n' %
                            (i + 1, ll, dll))


        old_ll = ll

        if best_vll < vll:
            best_vll = vll
            best_model = mm

    return best_model