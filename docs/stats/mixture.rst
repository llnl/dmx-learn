.. _stats_mixture:
Mixture Distribution
=========================
Mixture distributions are useful when a statistical population contains two or more subpopulations that are unobserved. *dmx-learn* allows for the specification of any model for the *components* of the mixture distribution. Assuming we have an observation *x* of data type *T* (any heterogenous form), the data generating process for a K-component mixture model is given by 

.. math::

   \begin{array}{ll}
   z &\sim \boldsymbol{\pi} \\
   x \vert z &\sim f_k(x \vert \theta_k)
  \end{array} 

where :math:`\pi_k` representing the probability of *x* being drawn from component distribution :math:`f_k(x \vert \theta_k)`. For more details see `Mixture Distribution <https://en.wikipedia.org/wiki/Mixture_distribution_>`__.

MixtureDistribution
---------------------------------

.. autoclass:: dmx.stats.mixture.MixtureDistribution
   :members:
   :special-members: __init__

MixtureEstimator
-----------------------------

.. autoclass:: dmx.stats.mixture.MixtureEstimator
   :members:
   :special-members: __init__

MixtureSampler
--------------------------

.. autoclass:: dmx.stats.mixture.MixtureSampler
   :members:


