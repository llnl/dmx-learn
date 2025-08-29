Binomial
========

Data Type: int

The binomial distribution is used for modeling the number of successes for a given number of trials *n*. The distribution has support on the integers between [*min_val*, *min_val* + *n*), where *min_val* is supplied by the user. 

The probability mass function is given by 

.. math::
   f(x | n, p) = {n \choose x} p^x (1-p)^{n-x}, \: min\_val\leq x < min\_val + n.

BinomialDistribution
----------------------

.. autoclass:: dmx.stats.binomial.BinomialDistribution
   :members:
   :special-members: __init__

BinomialEstimator
--------------------

.. autoclass:: dmx.stats.binomial.BinomialEstimator
   :members:
   :special-members: __init__

BinomialSampler
------------------

.. autoclass:: dmx.stats.binomial.BinomialSampler
   :members:

