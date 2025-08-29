"""Example on how to fit dmx/stats models with mpi4py. 

In general models are launched with calls like mpiexec -n 4 python3 estimation_example.py

"""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'

from mpi4py import MPI
from numpy.random import RandomState
import pickle
from dmx.stats import *
from dmx.mpi4py.stats import *
from dmx.mpi4py.utils.estimation import optimize_mpi
from dmx.mpi4py.utils.optsutil import pickle_on_master


comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()


if __name__ == "__main__":

    # First we can simulate some data on the master node
    # in general you can replace this with just reading in data here
    if world_rank == 0: 
        d00 = GaussianDistribution(mu=0.0, sigma2=1.0)
        d01 = CategoricalDistribution({'a': 0.3, 'b': 0.7})
        d0 = CompositeDistribution([d00, d01])

        d10 = GaussianDistribution(mu=3.0, sigma2=1.0)
        d11 = CategoricalDistribution({'a': 0.7, 'b': 0.3})
        d1 = CompositeDistribution([d10, d11])

        dist = MixtureDistribution([d0, d1], w=[0.25, 0.75])

        data = dist.sampler(seed=1).sample(1000)

    else:
        data = None


    # Now we define the estimator
    e0 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
    est = MixtureEstimator([e0]*2)

    # We can just directly call optimize as usual 
    rng = RandomState(1)
    fit = optimize_mpi(data=data, estimator=est, rng=rng)

    # note that the model is living on the master node
    print(f"Rank {world_rank}: Model is None == {fit is None}")

    # if we want to save the model
    pickle_on_master(fit, "mpi4py_model_fit.pkl")

    if world_rank == 0:
        print(f"Wrote file ./mpi4py_model_fit.pkl")


    


