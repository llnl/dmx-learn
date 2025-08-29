Conditional Distribution
=========================
The Conditional distribution is used to model conditional dependencies between two random variables. This can be used for separate data types. 

Assume we have observed :math:`(x_i, y_i)` where :math:`x_i` has data type :math:`T_1` and :math:`y_i` has data type :math:`T_2` The Conditional distribution is used to model conditional dependencies between two random variables. This can be used for separate data types. 

Assume we have observed :math:`(x_i, y_i)` where :math:`x_i` has data type :math:`T_1` and :math:`y_i` has data type :math:`T_2`. Choosing a compatible *given* distribution :math:`f(x_i \vert theta_1)` for :math:`x_i` and a distribution :math:`g(y_i \vert \theta_2)`, the conditional density is given by 

.. math::

   f((x_i, y_i)) = g(y_i \vert x_i, \theta_2) h(x_i \vert \theta_1)).

Note that each value of :math:`x_i` emits a distribution over the support of *y* values. 

ConditionalDistribution
---------------------------------

.. autoclass:: dmx.stats.conditional.ConditionalDistribution
   :members:
   :special-members: __init__

ConditionalDistributionEstimator
----------------------------------

.. autoclass:: dmx.stats.conditional.ConditionalDistributionEstimator
   :members:
   :special-members: __init__

ConditionalDistributionSampler
-------------------------------

.. autoclass:: dmx.stats.conditional.ConditionalDistributionSampler
   :members:


