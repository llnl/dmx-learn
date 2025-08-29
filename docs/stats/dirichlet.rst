Dirichelt Distribution
================================

Data Type: Sequence[float] 

The Dirichlet distribution is a distribution on the simplex. A d-dimensinoal Dirichlet random variable :math:`(X_1, X_2, ..., X_d)` has a density function is given by

.. math::

   f(\boldsymbol{x} \vert \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})}\prod_{i=1}^{d} x_i^{\alpha_i-1}, \;0 \leq x_i \leq 1  

where :math:`B(\boldsymbol(\alpha)) = \frac{\prod_{i=1}^{d}\Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^{d}\alpha_i\right)}` and :math:`\sum_{i=1}^{d} x_i = 1`.
  
   
For more info see `Dirichlet Distribution <https://en.wikipedia.org/wiki/Dirichlet_normal_distribution_>`__.


DirichletDistribution
---------------------------------

.. autoclass:: dmx.stats.dirichlet.DirichletDistribution
   :members:
   :special-members: __init__

DirichletEstimator
---------------------------

.. autoclass:: dmx.stats.dirichlet.DirichletEstimator
   :members:
   :special-members: __init__

DirichletSampler
---------------------------

.. autoclass:: dmx.stats.dirichlet.DirichletSampler
   :members:

