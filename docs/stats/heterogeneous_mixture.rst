Heterogeneous Mixture Distribution
===================================
The Heterogeneous mixture distribution can be used to assign heterogeneous mixture components to the :ref:`stats_mixture`. For example, consider observing a postive float *x*. We can define a two component mixture to be :math:`f_1(x \vert \lambda) \sim Exp(\lambda)` and :math:`f_2(x \vert \mu, \sigma) \sim LogNormal(\mu, \sigma)`. The only requirement for the components of the heterogeneous mixture is that the components distributions have the same support as the data type of *x*.   

HeterogeneousMixtureDistribution
---------------------------------

.. autoclass:: dmx.stats.heterogeneous_mixture.HeterogeneousMixtureDistribution
   :members:
   :special-members: __init__

HeterogeneousMixtureEstimator
--------------------------------

.. autoclass:: dmx.stats.heterogeneous_mixture.HeterogeneousMixtureEstimator
   :members:
   :special-members: __init__

HeterogeneousMixtureSampler
-------------------------------

.. autoclass:: dmx.stats.heterogeneous_mixture.HeterogeneousMixtureSampler
   :members:

