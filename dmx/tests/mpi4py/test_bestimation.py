"""Tests for mpi4py functionality on bstats estimation functions. 

All tests are run with mpiexec -n 4 pytest test_bestimation

"""
import os
import pickle
from dmx.bstats import *
from dmx.mpi4py.bstats import *
from dmx.mpi4py.bstats import *
from dmx.mpi4py.utils.bestimation import optimize_mpi
import numpy as np
from mpi4py import MPI

DATA_DIR = "dmx/tests/data"
ANSWER_DIR = "dmx/tests/answerkeys"

def test_initialize_mpi() -> None:
    """Test initialize with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_rank == 0:

        with open(os.path.join(DATA_DIR, 'testInput_optimize_mpi_n4.pkl'), 'rb') as f:
            data = pickle.load(f)

        with open(os.path.join(DATA_DIR, 'testInput_bstats_estimator.pkl'), 'rb') as f:
            est = pickle.load(f)

        print(f"RANK {world_rank}: {est}")

        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_initialize_mpi_n4.pkl'), 'rb') as f:
            answer = pickle.load(f)


    else:
        data = None
        est = None

    rng = np.random.RandomState(1)
    est = comm.bcast(est, root=0)
    prev_mm = initialize_mpi(data, estimator=est, rng=rng, p=0.10)

    if world_rank == 0:
        assert str(prev_mm) == str(answer)
    

def test_seq_encode_mpi() -> None:
    """Test sequence encoding with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:

        with open(os.path.join(DATA_DIR, 'testInput_optimize_mpi_n4.pkl'), 'rb') as f:
            data = pickle.load(f)

        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_initialize_mpi_n4.pkl'), 'rb') as f:
            prev_estimate = pickle.load(f)

    else:
        data = None
        prev_estimate = None


    # load the answers for each of the 4 workers
    with open(os.path.join(ANSWER_DIR, f'testOutput_bstats_seq_encode_mpi_n4_rank{world_rank}.pkl'), 'rb') as f:
        answer = pickle.load(f)


    enc_data = seq_encode_mpi(data=data, model=prev_estimate)

    assert str(enc_data) == str(answer)

    
def test_seq_estimate_mpi() -> None:
    """Test sequence estimation with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:

        with open(os.path.join(DATA_DIR, 'testInput_bstats_estimator.pkl'), 'rb') as f:
            est = pickle.load(f)

        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_initialize_mpi_n4.pkl'), 'rb') as f:
            prev_estimate = pickle.load(f)

    else:
        est = None
        prev_estimate = None


    # load the answers for each of the 4 workers
    with open(os.path.join(ANSWER_DIR, f'testOutput_bstats_seq_encode_mpi_n4_rank{world_rank}.pkl'), 'rb') as f:
        enc_data = pickle.load(f)

    next_mm = seq_estimate_mpi(enc_data=enc_data, estimator=est, prev_estimate=prev_estimate)

    if world_rank == 0:
        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_seq_estimate_mpi_n4.pkl'), 'rb') as f:
            answer = pickle.load(f)

        assert str(next_mm) == str(answer)


def test_seq_log_density_mpi() -> None:
    """Test sequence log density with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_seq_estimate_mpi_n4.pkl'), 'rb') as f:
            prev_estimate = pickle.load(f)

    else:
        prev_estimate = None


    # load the answers for each of the 4 workers
    with open(os.path.join(ANSWER_DIR, f'testOutput_bstats_seq_encode_mpi_n4_rank{world_rank}.pkl'), 'rb') as f:
        enc_data = pickle.load(f)

    ll = seq_log_density_mpi(enc_data=enc_data, estimate=prev_estimate)

    if world_rank == 0:
        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_seq_log_density_mpi_n4.pkl'), 'rb') as f:
            answer = pickle.load(f)

        print(answer) 
        assert np.all(ll == answer[0])


def test_bestimation_optimize_mpi() -> None:
    """Test bstats optimize mpi call with mpi4py using 4 cores."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    if world_rank == 0:
        with open(os.path.join(DATA_DIR, 'testInput_optimize_mpi_n4.pkl'), 'rb') as f:
            data = pickle.load(f)

    else:
        data = None

    with open(os.path.join(DATA_DIR, 'testInput_bstats_estimator.pkl'), 'rb') as f:
        est = pickle.load(f)

    rng = np.random.RandomState(1)
    model = optimize_mpi(data, estimator=est, rng=rng)

    if world_rank == 0:
        with open(os.path.join(ANSWER_DIR, 'testOutput_bstats_optimize_mpi_n4.pkl'), 'rb') as f:
            answer = pickle.load(f)

        assert str(model) == str(answer)