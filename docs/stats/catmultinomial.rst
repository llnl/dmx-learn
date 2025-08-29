Multinomial Distribution
==================================

Data Type: Sequence[Tuple[int, str]]

The multinomial distribution is a generalization of the binomial distribution to *k* classes. The multinomial give the probability of observing :math:`x_k` success/counts of class object :math:`v_k` in :math:`n=\sum_{i} x_i` trials. The probability mass function is given by 

.. math::

   f(\boldsymbol{x} \vert n, \boldsymbol{p}) = \frac{n!}{x_1!\dots x_k!} p_1^{x_1}\dots p_k^{x_k}, 

where :math:`\sum_{i=1}^{k} p_i = 1` and :math:`\sum_{i=1}^{k} x_i = n`. Here we have allowed the classes to be represented by an object/string :math:`v_i` in the set :math:`V=\{v_1, \dots, v_k\}`. If the user maps the objects to the set of integers, the :doc:`Integer Multinomial Distribtion <intmultinomial>` can be used instead.  

For more info see `Multinomial Distribution <https://en.wikipedia.org/wiki/Multinomial_distribution_>`__.


MultinomialDistribution
---------------------------------

.. autoclass:: dmx.stats.catmultinomial.MultinomialDistribution
   :members:
   :special-members: __init__

MultinomialEstimator
-----------------------------

.. autoclass:: dmx.stats.catmultinomial.MultinomialEstimator
   :members:
   :special-members: __init__

MultinomialSampler
--------------------------

.. autoclass:: dmx.stats.catmultinomial.MultinomialSampler
   :members:


