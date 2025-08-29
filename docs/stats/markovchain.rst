Markov Chain 
=====================

A Markov chain is a stochastic process that undergoes transitions from one state to another within a finite or countable number of possible states. It is characterized by the property that the future state depends only on the current state and not on the sequence of events that preceded it. This property is known as the Markov property.

Mathematically, a Markov chain can be defined as follows:

- Let `S`  be a finite set of states.
- Let `P` be the transition matrix, where :math:`P_{ij}` represents the probability of moving from state `i` to state `j`.

The transition probabilities must satisfy the following conditions:

.. math::

   \sum_{j \in S} P_{ij} = 1 \quad \text{for all } i \in S

Likelihood Evaluation
-------------------------
Assume we have observed a sequence :math:`\boldsymbol{x} = (x_1, ..., x_{n})` of length :math:`n`. The likelihood is given by 

.. math:: 

   p(\boldsymbol{x}_i) = p(N=n)p(x_1) \prod_{k=1}^{n}p(x_k \vert x_{k-1}) 

In the above equation, :math:`p(x_k \vert x_{k-1})` is defined by the transition matrix, :math:`p(x_1)` is defined by the intial state vector :math:`\pi`, and :math:`p(N=n)` is a length distribution on the integers. 


MarkovChainDistribution
--------------------------

.. autoclass:: dmx.stats.markovchain.MarkovChainDistribution
   :members:
   :special-members: __init__

MarkovChainEstimator
-----------------------

.. autoclass:: dmx.stats.markovchain.MarkovChainEstimator
   :members:
   :special-members: __init__

MarkovChainSampler
---------------------

.. autoclass:: dmx.stats.markovchain.MarkovChainSampler
   :members:

