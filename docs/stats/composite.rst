Composite Distribution
=========================
The composite distribution is the staple distribtion of *dmx-learn* that allows for distributions over heterogenous tuples of data. Assume we have observed a d-dimensional tuple :math:`x=(x_1, x_2, \dots, x_d)` with component-wise data types :math:`(T_1, T_2, \dots, T_d)`. The composite distribution models the tuple with a likelihood 

.. math::

   f(x_1, \dots, x_d \vert \theta_1, \dots, \theta_k) = \prod_{i=1}^{d} f(x_i \vert \theta_i)

where :math:`f(x_i \vert \theta_i)` are distributions compatible with component data type :math:`T_i`.


CompositeDistribution
---------------------------------

.. autoclass:: dmx.stats.composite.CompositeDistribution
   :members:
   :special-members: __init__

CompositeEstimator
-----------------------------

.. autoclass:: dmx.stats.composite.CompositeEstimator
   :members:
   :special-members: __init__

CompositeSampler
--------------------------

.. autoclass:: dmx.stats.composite.CompositeSampler
   :members:


