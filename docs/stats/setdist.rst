Bernoulli Set Distribution
====================================

Data Type: Sequence[str]

The Bernoulli set distribution is distribution over the power sets of elements :math:`V = \{v_0, v_1, ..., v_{n-1}\}`. Each element :math:`v_i` is included in the set with probability :math:`p_i`. Note there is no constraint :math:`\sum_{i} p_i = 1`, as each :math:`p_i` simply models the probability that element *v_i* is included in the set. Let **x** be a subset of *V*. The probability mass function for a Bernoulli set distribution is given by

.. math::

   f(\boldsymbol{x} \vert \boldsymbol{p}) = \prod_{i=0}^{n-1} [v_i \in \boldsymbol{x}] p_i + [v_i \notin \boldsymbol{x}] (1-p_i).

For speed, the user can map observed values :math:`v_i \rightarrow i` and use the :doc:`Integer Categorical Distribution <intsetdist>`. 

BernoulliSetDistribution
---------------------------------

.. autoclass:: dmx.stats.setdist.BernoulliSetDistribution
   :members:
   :special-members: __init__

BernoulliSetEstimator
-----------------------------

.. autoclass:: dmx.stats.setdist.BernoulliSetEstimator
   :members:
   :special-members: __init__

BernoulliSetSampler
--------------------------------

.. autoclass:: dmx.stats.setdist.BernoulliSetSampler
   :members:

