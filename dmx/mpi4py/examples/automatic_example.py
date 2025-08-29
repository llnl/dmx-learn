"""Example on how to automatically select and estimator and fit a DPM with mpi4py. 

In general models are launched with calls like mpiexec -n 4 python3 automatic_example.py

"""
import os
import pickle
os.environ['NUMBA_DISABLE_JIT'] =  '1'

from mpi4py import MPI
from numpy.random import RandomState

from dmx.mpi4py.utils.automatic import get_dpm_mixture_mpi, get_estimator
from dmx.mpi4py.utils.optsutil import pickle_on_master


comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

PATH_TO_DATA = "dmx/mpi4py/examples/data"

if __name__ == "__main__":

    # Load the data on the master node
    if world_rank == 0: 

        with open(os.path.join(PATH_TO_DATA, "sample_data.pkl"), "rb") as f:
            data = pickle.load(f)

    else:
        data = None
    
    rng = RandomState(1)
    # automatically select and estimator
    est = get_estimator(data=data, use_bstats=True)

    # fit a DPM using mpi4py, note the model is returned on all nodes in this case
    fit = get_dpm_mixture_mpi(data=data, estimator=est, rng=rng)

    # if we want to save the model
    pickle_on_master(fit, "mpi4py_dpm_fit.pkl")


    


