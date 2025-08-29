Log-Gaussian (Log-Normal)
==========================

Data Type: float 

The log-Gaussian distribution is used to model data that is normal on the log scale.  The probability density function is given by 

.. math::

   f\left(x | \mu, \sigma^2 \right) = \frac{1}{\sqrt{2\pi\sigma^2}x}e^{-\frac{(\log{x}-\mu)^2}{2\sigma^2}}, \; x \in \mathbb{R}. 

For more info see `log-Gaussian Distribution <https://en.wikipedia.org/wiki/Log-normal_distribution_>`__.


LogGaussianDistribution
--------------------------

.. autoclass:: dmx.stats.log_gaussian.LogGaussianDistribution
   :members:
   :special-members: __init__

LogGaussianEstimator
-----------------------

.. autoclass:: dmx.stats.log_gaussian.LogGaussianEstimator
   :members:
   :special-members: __init__

LogGaussianSampler
---------------------

.. autoclass:: dmx.stats.log_gaussian.LogGaussianSampler
   :members:

