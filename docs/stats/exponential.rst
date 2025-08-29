Exponential 
=================

Data Type: float 

The exponential distribution is can be used to model arrival times between events. The distribution has support on the positive real line.  The probability density function is given by 

.. math::

   f\left(x | \beta \right) = \frac{1}{\beta}e^{-\frac{1}{\beta}x}, \; x  \geq 0. 

The above is the scale parametarization of the exponential distribution. For more info see `Exponential Distribution <https://en.wikipedia.org/wiki/Exponential_distribution_>`__.


ExponentialDistribution
------------------------

.. autoclass:: dmx.stats.exponential.ExponentialDistribution
   :members:
   :special-members: __init__

ExponentialEstimator
-----------------------

.. autoclass:: dmx.stats.exponential.ExponentialEstimator
   :members:
   :special-members: __init__

ExponentialSampler
-------------------

.. autoclass:: dmx.stats.exponential.ExponentialSampler
   :members:

