.. _stats_hidden_markov:
Hidden Markov Distribution
===============================
Hidden Markov Models (HMMs) are statistical models used to represent systems that are assumed to be a Markov process with hidden (unobserved) states. They are particularly useful in scenarios where the system being modeled is not directly observable, but can be inferred through observable outputs.

.. list-table:: Summary of Hidden Markov Models
   :header-rows: 1
   
   * - Feature
     - Symbol
     - Description
   * - Initial States
     - :math:`\boldsymbol{\pi}`
     - Finite set of initial hidden states representing possible initial conditions of the system.
   * - Observations
     - :math:`\boldsymbol{Y}_i = (y_i(0), ..., y_{i}(t_i - 1))`
     - Outputs produced by hidden states according to a probability distribution.
   * - Transition Probabilities
     - :math:`\boldsymbol{\tau}`, and S by S matrix with entries :math:`P(Z(t)=j \vert Z(t-1)=i)`
     - Probabilities associated with transitioning from one hidden state to another.
   * - Emission Probabilities
     - :math:`f_k(y(t) \vert Z(t)=k)`
     - Likelihood of producing each possible observation from hidden states. 

The generative process for the Hidden Markov model is described as follows, for the initial value

.. math::

   Z(0) &\sim \pi \\
   y(0) &\sim f_{Z(0)}

for time points 1,2, ..., t-1,

.. math::

   Z(t) \vert Z(t-1) &\sim \boldsymbol{\tau}_{Z(t)} \\
   Y(t) \vert Z(t) &\sim f_{Z(t)}(\cdot) 

HiddenMarkovModelDistribution
---------------------------------

.. autoclass:: dmx.stats.hidden_markov.HiddenMarkovModelDistribution
   :members:
   :special-members: __init__

HiddenMarkovEstimator
-----------------------------

.. autoclass:: dmx.stats.hidden_markov.HiddenMarkovEstimator
   :members:
   :special-members: __init__

HiddenMarkovSampler
--------------------------

.. autoclass:: dmx.stats.hidden_markov.HiddenMarkovSampler
   :members:


