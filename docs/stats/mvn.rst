Multivariate Gaussian
================================

Data Type: Sequence[float] 

The probability density function of a d-dimensional multivariate Gaussian random variable :math:`(X_1, X_2, ..., X_d)` with mean :math:`\boldsymbol{\mu}` and postive definite covariance matrix :math:`\Sigma` is given by

.. math::

   f(\boldsymbol{x} | \boldsymbol{\mu}, \Sigma) = \left(\frac{1}{2\pi}\right)^{d/2} \vert\Sigma\vert^{d/2} \exp{\left(-frac{1}{2}\left(\boldsymbol{x} - \boldsymbol{\mu}\right)^{t} \Sigma^{-1} \left(\boldsymbol{x} - \boldsymbol{\mu}\right)\right)}.


If you are assuming :math:`Cov(X_i, X_j) = 0 \; i \neq j`, a faster and more efficient option is the :doc:`Diagonal Multivariate Gaussian <dmvn>`. 

For more info see `Multivariate Normal Distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution_>`__.


MultivariateGaussianDistribution
---------------------------------

.. autoclass:: dmx.stats.mvn.MultivariateGaussianDistribution
   :members:
   :special-members: __init__

MultivariateGaussianEstimator
--------------------------------

.. autoclass:: dmx.stats.mvn.MultivariateGaussianEstimator
   :members:
   :special-members: __init__

MultivariateGaussianSampler
------------------------------

.. autoclass:: dmx.stats.mvn.MultivariateGaussianSampler
   :members:

