Integer Multinomial Distribution
==================================

Data Type: Sequence[Tuple[int, float]]

The integer multinomial distribution is a generalization of the binomial distribution to *k* classes. The multinomial give the probability of observing *n_k* success of class *k* in :math:`n=\sum_{i}n_i` trials. The probability mass function is given by 

.. math::

   f(\boldsymbol{x} \vert n, \boldsymbol{p}) = \frac{n!}{x_1!\dots x_k!} p_1^{x_1}\dots p_k^{x_k}, 

where :math:`\sum_{i=1}^{k} p_i = 1` and :math:`\sum_{i=1}^{k} x_i = n`.

For more info see `Multinomial Distribution <https://en.wikipedia.org/wiki/Multinomial_distribution_>`__.


IntegerMultinomialDistribution
---------------------------------

.. autoclass:: dmx.stats.intmultinomial.IntegerMultinomialDistribution
   :members:
   :special-members: __init__

IntegerMultinomialEstimator
-----------------------------

.. autoclass:: dmx.stats.intmultinomial.IntegerMultinomialEstimator
   :members:
   :special-members: __init__

IntegerMultinomialSampler
--------------------------

.. autoclass:: dmx.stats.intmultinomial.IntegerMultinomialSampler
   :members:


