Categorical
====================

Data Type: str 

The Categorical distributioni, also known as the mutlinomial distribution, is a probability distribution over a set of possible categories. Although this distribution is claimed to support data type str, this distribtion can be used to define a distribution over any set of objects. The probability mass function is given by over a set of values :math:`V=\{v_1, ..., v_k\}` is given by 

.. math::

   f(x|\boldsymbol{p}) = \left\{ \begin{array}{ll}
        p_i, & x=v_i \\
        0, & x \notin V  
   \end{array} \right.

where :math:`\sum_{i=1}^{k} p_i = 1`. Note that any set of values *V* can be enumerated and mapped to the set of integers :math:`0, 1, ..., k-1`. The user can then refer to the faster :doc:`Integer Categorical Distribution <intrange>`.

For more info see `Categorical Distribution <https://en.wikipedia.org/wiki/Categorical_distribution_>`__.


CategoricalDistribution
---------------------------------

.. autoclass:: dmx.stats.categorical.CategoricalDistribution
   :members:
   :special-members: __init__

CategoricalEstimator
-----------------------------

.. autoclass:: dmx.stats.categorical.CategoricalEstimator
   :members:
   :special-members: __init__

CategoricalSampler
--------------------------

.. autoclass:: dmx.stats.categorical.CategoricalSampler
   :members:


