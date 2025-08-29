Hierarchical Mixture Distribution
===================================
A Hierarchical Mixture Model is a statistical model that combines multiple layers of mixture models to capture complex data distributions. It is particularly useful in scenarios where data can be grouped into subpopulations, each of which may follow its own mixture distribution. This model is often referred to as a "mixture of mixtures." The data format required for using the Hierarchical mixture requries a sequence of observations of the form :math:`\boldsymbol{X}_i = (X_{i,1}, ..., X_{i, n_i})`. This can be thought of as independent draws from a topic, where the topic distribution is also a mixture model. 

.. list-table:: Hierarchical Mixture Model Features
   :header-rows: 1

   * - Feature
     - Symbol
     - Description
   * - Outer-State Distribution
     - :math:`\boldsymbol{\pi}` 
     - Represents the distribution over the :math:`K_1` outer states.
   * - Inner-State Probabilities
     - :math:`\boldsymbol{\tau}_{k}`
     - Distribution over the :math:`K_2` inner-states.
   * - Component Distributions
     - :math:`f_{k}(x)`
     - The probability distribution of the observed data given a specific inner state.
   * - Length Distribution
     - :math:`g(\cdot)`
     - Distribution for the lengths of each observation.


The generative process for data :math:`\boldsymbol{X}_i = (X_{i,1}, ..., X_{i, n_i})` is as follows,

.. math::

   \begin{array}{ll}
   n_i &\sim g(\cdot) \\
   Z_i & \sim \boldsymbol{\pi} \\
   U_{i, j} \vert Z_i  & \sim \boldsymbol{\tau}_{Z_i} \\
   X_{i, j} \vert U_{i, j} & \sim f_{U_{i, j}}(\cdot)
   \end{array}

   


HierarchicalMixtureDistribution
---------------------------------

.. autoclass:: dmx.stats.hmixture.HierarchicalMixtureDistribution
   :members:
   :special-members: __init__

HierarchicalMixtureEstimator
--------------------------------

.. autoclass:: dmx.stats.hmixture.HierarchicalMixtureEstimator
   :members:
   :special-members: __init__

HierarchicalMixtureSampler
-------------------------------

.. autoclass:: dmx.stats.hmixture.HierarchicalMixtureSampler
   :members:

