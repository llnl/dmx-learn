Integer Categorical
====================

Data Type: int

The Integer Categorical distribution is a categorical distribution defined on an integer support. The probability mass function is given by 

.. math::

   f(x|\boldsymbol{p}) = \left\{ \begin{array}{ll}
        p_i, & x=k \\
        0, & x \notin [min\_val, min\_val + n)  
   \end{array} \right.

where :math:`\sum_{i} p_i = 1` and *n* is a user-defined length. A categorical distribution defined over sets of objects can be used if the user does not want map a given set of values to the integers (see :doc:`Categorical Distribution <categorical>`.)

For more info see `Integer Categorical Distribution <https://en.wikipedia.org/wiki/Categorical_distribution_>`__.


IntegerCategoricalDistribution
---------------------------------

.. autoclass:: dmx.stats.intrange.IntegerCategoricalDistribution
   :members:
   :special-members: __init__

IntegerCategoricalEstimator
-----------------------------

.. autoclass:: dmx.stats.intrange.IntegerCategoricalEstimator
   :members:
   :special-members: __init__

IntegerCategoricalSampler
--------------------------

.. autoclass:: dmx.stats.intrange.IntegerCategoricalSampler
   :members:


