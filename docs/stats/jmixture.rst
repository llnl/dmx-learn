Joint Mixture Distribution
===================================
The Joint Mixture Distribution is a mixture of mixtures (see :ref:`stats_mixture`). This model is particularly useful when observations can belong to multiple latent groups simultaneously. This model can capture mutli-level clustering and dependencies. For :math:`K_1 = K_2`, this model can be viewed as a single-step :ref:`stats_hidden_markov`. The generative process for a Joint Mixture Model with :math:`K_1` outer-states and :math:`K_2` inner-states is described as

.. math::

   \begin{array}{ll}
   z_1 &\sim \boldsymbol{\pi} \\
   z_2 \vert z_1 = k_1 &\sim \boldsymbol{\tau_{k_1}} \\
   x \vert z_2 = k_2 &\sim f_k(x \vert \theta_{k_2})
  \end{array} 

where the initial group membership is drawn :math:`P(Z_1 = k_1) = \pi_{k_1}` and transition probability is given by :math:`P(Z_2 = k_2 \vert Z_1 = k_1) = \tau_{k_1, k_2}`.  



JointMixtureDistribution
---------------------------------

.. autoclass:: dmx.stats.jmixture.JointMixtureDistribution
   :members:
   :special-members: __init__

JointMixtureEstimator
--------------------------------

.. autoclass:: dmx.stats.jmixture.JointMixtureEstimator
   :members:
   :special-members: __init__

JointMixtureSampler
-------------------------------

.. autoclass:: dmx.stats.jmixture.JointMixtureSampler
   :members:

