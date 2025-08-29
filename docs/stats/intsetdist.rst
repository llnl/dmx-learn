Integer Bernoulli Set Distribution
====================================

Data Type: Sequence[int]

The integer Bernoulli set distribution is distribution over the power sets of size *n*. Each integer between [0, n) is included in the set with probability :math:`p_i`. Note there is no constraint :math:`\sum_{i} p_i = 1`, as each :math:`p_i` simply models the probability that integer *i* is included in the set. Let :math:`x = (x_0, x_1, ..., x_{n-1})` be a tuple of binary variables indicating set membership. The probability mass function for for a Bernoulli set distribution is given by

.. math::

   f(\boldsymbol{x} \vert \boldsymbol{p}) = \prod_{i=0}^{n-1} p_i^{x_i}(1-p_i)^{1-x_i}

See :doc:`Bernoulli Set Distribution <setdist>` for a more generic implementation over sets of any objects.

IntegerBernoulliSetDistribution
---------------------------------

.. autoclass:: dmx.stats.intsetdist.IntegerBernoulliSetDistribution
   :members:
   :special-members: __init__

IntegerBernoulliSetEstimator
-----------------------------

.. autoclass:: dmx.stats.intsetdist.IntegerBernoulliSetEstimator
   :members:
   :special-members: __init__

IntegerBernoulliSetSampler
--------------------------------

.. autoclass:: dmx.stats.intsetdist.IntegerBernoulliSetSampler
   :members:

