Optional Distribution
=========================
The Optional distribution assigns a probability (p) to data being missing. With probability (1-p) the data is assumed to come
from a base distribution set by the user.

Assuming the data follows a distribution :math:`g(x_i \vert \theta)`, the
likelihood for the Optional distribution is given by

.. math::

   f(x|\theta, p) = \left\{ \begin{array}{ll}
        p, & x \text{ is missing} \\
        (1-p) g(x \vert \theta), & else
   \end{array} \right.

We allow for the user to define the *missing value*.



OptionalDistribution
---------------------------------

.. autoclass:: dmx.stats.optional.OptionalDistribution
   :members:
   :special-members: __init__

OptionalEstimator
-----------------------------

.. autoclass:: dmx.stats.optional.OptionalEstimator
   :members:
   :special-members: __init__

OptionalSampler
--------------------------

.. autoclass:: dmx.stats.optional.OptionalSampler
   :members:


