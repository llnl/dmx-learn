Gamma 
=================

Data Type: float 

The gamma distribution is a generalization of the exponential distribution. The distribution has support on the positive real line.  The probability density function is given by 

.. math::

   f\left(x | k, \theta \right) = \frac{1}{\Gamma(k)\theta^k}x^{k-1}e^{-\frac{1}{\theta}x}, \; x  \geq 0. 

The above is the scale parametarization of the gamma distribution. For more info see `Gamma Distribution <https://en.wikipedia.org/wiki/Gamma_distribution_>`__.


GammaDistribution
------------------------

.. autoclass:: dmx.stats.gamma.GammaDistribution
   :members:
   :special-members: __init__

GammaEstimator
-----------------------

.. autoclass:: dmx.stats.gamma.GammaEstimator
   :members:
   :special-members: __init__

GammaSampler
-------------------

.. autoclass:: dmx.stats.gamma.GammaSampler
   :members:

