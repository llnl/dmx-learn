Diagonal Multivariate Gaussian
================================

Data Type: Sequence[float] 

A d-dimensinoal random variable :math:`(X_1, X_2, ..., X_d)` follows a  diagonal multivariate normal distribution if each :math:`X_i \sim N\left(0, \sigma^2_i \right)` and :math:`Cov(X_i, X_j) = 0` for each :math:`i \neq j`. 

The probability density function is given by

.. math::
   f(\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\sigma}) = \left(\frac{1}{2\pi}\right)^{d/2} \prod_{i=1}^{d}\frac{1}{\sigma_i} \exp{-\left(\frac{(x_i-\mu_i)^2}{2\sigma^2_i}\right)}

For more info see `Multivariate Normal Distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution_>`__.


DiagonalGaussianDistribution
---------------------------------

.. autoclass:: dmx.stats.dmvn.DiagonalGaussianDistribution
   :members:
   :special-members: __init__

DiagonalGaussianEstimator
---------------------------

.. autoclass:: dmx.stats.dmvn.DiagonalGaussianEstimator
   :members:
   :special-members: __init__

DiagonalGaussianSampler
---------------------------

.. autoclass:: dmx.stats.dmvn.DiagonalGaussianSampler
   :members:

