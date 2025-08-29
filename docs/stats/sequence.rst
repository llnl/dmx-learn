Sequence Distribution
=========================
The sequence distribution is used to model independent and identitcally distributed (*iid*) sequences of observations or varying lengths. We can also model the distribution for the lengths of the sequences.

Assume :math:`x_i = (x_{i1}, ..., x_{i n_i})` is a sequence of length :math:`n_i` having data type `T`. The sequence distribution models each :math:`x_{i, j}` with a distribution compatible with type *T* data, :math:`g(x_i \vert \theta)`. The lengths of the sequences `n_i` are modeled with a distribution on the integers :math:`h(n_i \vert \phi)`. The likelhood for a set of observed sequences :math:`X=([x_{1,1}, \dots, x_{1, n_1}], \dots, [x_{N, 1}, \dots, x_{N, n_N}])` is 

.. math::

   f(X) = \prod_{i=1}^{N} g(x_{i, 1}, \dots, x_{i, n_i} \vert \theta) h(n_i \vert \phi).

SequenceDistribution
---------------------------------

.. autoclass:: dmx.stats.sequence.SequenceDistribution
   :members:
   :special-members: __init__

SequenceEstimator
-----------------------------

.. autoclass:: dmx.stats.sequence.SequenceEstimator
   :members:
   :special-members: __init__

SequenceSampler
--------------------------

.. autoclass:: dmx.stats.sequence.SequenceSampler
   :members:


