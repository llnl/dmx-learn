Fisher-Geometry Embeddings
==========================

:func:`dmx.utils.humap.humap` embeds records with UMAP applied to the
posterior probabilities of a fitted mixture. Posteriors say which component a
record belongs to but not where it sits inside that component: with a
well-fitted model most posteriors are close to one-hot, so every record in a
component is at distance close to zero from every other one, and the shape of
each cluster in the embedding is arbitrary.

The Fisher-geometry representation keeps that information. It compares
records through small, record-specific perturbations of the fitted model, and
reduces exactly to the posterior representation when the perturbation is
switched off. It is computed per record, so it scales and distributes the
same way posteriors do.

.. note::

   This page documents the design and the distribution API introduced for
   it. Per-family scores, the mixture-level embedding, and the ``humap``
   option are added incrementally; until a distribution implements its score,
   calling it raises :class:`NotImplementedError`. The plan is tracked in
   ``HUMAP_FISHER_ROADMAP.md``.

The metric
----------

Let the fitted mixture be :math:`p(x) = \sum_k \pi_k f_k(x \mid \eta_k)`,
with posteriors :math:`z_{ik} = p(k \mid x_i)`. For component :math:`k` let
:math:`T_k` be its sufficient statistics, :math:`\mu_k = E_k[T_k]` their
mean, and :math:`I_k = \mathrm{Cov}_k[T_k]` its Fisher information.

**Local models.** Each record :math:`x_i` defines a distribution over
(component, observation):

.. math::

   Q_i(k, x) = z_{ik}\, f_k(x \mid \tilde\eta_{ik}), \qquad
   \tilde\eta_{ik} = \eta_k + \varepsilon\, I_k^{-1}\bigl(T_k(x_i) - \mu_k\bigr).

The tilt is a natural-gradient step of component :math:`k` toward
:math:`x_i`. In mean parameters it is
:math:`\tilde\mu_{ik} = (1 - \varepsilon)\mu_k + \varepsilon T_k(x_i)`: a
conjugate update that adds :math:`x_i` with weight :math:`\varepsilon`, that
is, one observation's worth of evidence.

**Distance.** Records are compared by the Hellinger distance between their
local models:

.. math::

   H^2(Q_i, Q_j) = 1 - \sum_k \sqrt{z_{ik} z_{jk}}\;
   \mathrm{BC}\bigl(f_k(\cdot \mid \tilde\eta_{ik}),
                    f_k(\cdot \mid \tilde\eta_{jk})\bigr),

where :math:`\mathrm{BC}` is the Bhattacharyya coefficient. For exponential
families
:math:`\mathrm{BC}(\eta_1, \eta_2) = \exp\bigl(A(\tfrac{\eta_1 + \eta_2}{2})
- \tfrac12 A(\eta_1) - \tfrac12 A(\eta_2)\bigr)`, and for a composite of
independent fields it is the product of the per-field coefficients.

**Second-order form.** With the whitened score
:math:`s_{ik} = I_k^{-1/2}\bigl(T_k(x_i) - \mu_k\bigr)`, expanding the
coefficient for small :math:`\varepsilon` gives

.. math::

   2 H^2(Q_i, Q_j) \approx \bigl\|\sqrt{z_i} - \sqrt{z_j}\bigr\|^2
   + c \sum_k \sqrt{z_{ik} z_{jk}}\, \|s_{ik} - s_{jk}\|^2,
   \qquad c = \varepsilon^2 / 4.

The first term is the Hellinger distance between posteriors, which is what
``humap`` uses today.

**Feature map.** The embedding uses the Euclidean feature vector

.. math::

   \phi_i = \bigl[\sqrt{z_{i1}}, \dots, \sqrt{z_{iK}},\;
   \sqrt{c}\sqrt{z_{i1}}\, s_{i1}, \dots, \sqrt{c}\sqrt{z_{iK}}\, s_{iK}\bigr].

:math:`\|\phi_i - \phi_j\|^2` equals the second-order form whenever
:math:`z_{ik} = z_{jk}`, and otherwise differs by a term proportional to
:math:`\sqrt{z_{ik}} - \sqrt{z_{jk}}`, that is, only for records that are
already far apart. UMAP uses only each record's nearest neighbours, which
almost always share components.

Scale and truncation
--------------------

- Inside one component the neighbour ranking does not depend on :math:`c`;
  :math:`c` only sets how many within-component standard deviations count as
  much as a change of component. The default is :math:`c = 1 / p_{\text{eff}}`,
  where :math:`p_{\text{eff}}` is the posterior-weighted mean score
  dimension, so a typical within-component pair is as far apart as two
  records in different components.
- For exponential families :math:`z_{ik}` decays like
  :math:`\exp(-\|s_{ik}\|^2 / 2)`, faster than :math:`\|s_{ik}\|` grows, so
  keeping each record's few most probable components loses essentially
  nothing and keeps :math:`\phi` sparse.

Why it is principled
--------------------

- At :math:`c = 0` it is exactly the posterior representation.
- For an exponential-family component :math:`T_k` is sufficient, so
  :math:`s_{ik}` carries everything that component can say about
  :math:`x_i`. Only what the model treats as irrelevant is dropped.
- Every field of a heterogeneous record is perturbed by the same amount of
  evidence, not the same number of units. The Fisher metric is the unique
  metric invariant under sufficient-statistic transformations (Chentsov's
  theorem), so the units of a numeric field, the labels of a categorical
  field, and the parametrisation of a distribution do not change distances.
- For all-Gaussian data it reduces to posterior-blended per-component
  Mahalanobis distance. A categorical mismatch between levels :math:`a` and
  :math:`b` costs :math:`1/p_a + 1/p_b`.
- The representation is defined for new records through the fitted model, so
  it supports out-of-sample transforms.

UMAP compatibility
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - UMAP assumption or requirement
     - Status
   * - Locally (approximately) constant Riemannian metric
     - Met by construction: the pull-back of the Fisher--Rao metric, constant
       (:math:`I_k^{-1}`) inside a component and blended smoothly by
       :math:`\sqrt{z}` across components.
   * - Data uniform on the manifold
     - Handled by UMAP's per-point normalisation, as for any input.
   * - Local connectivity (distance to the nearest neighbour is positive)
     - Violated by near-one-hot posteriors and restored by :math:`\phi`.
       Exact duplicate records remain ties; deduplicate them with counts.
   * - Searchable dissimilarity for nearest-neighbour descent
     - :math:`\phi` is Euclidean, so it is a true metric.
   * - Transforming new data
     - Supported.

Caveats: discrete-only data has no true manifold, so UMAP acts as a graph
method there, as it does on any categorical data. The second-order form is
unbounded for very rare categories, which ``prob_floor`` limits. The geometry
is only as good as the fitted model.

Distribution API
----------------

Distributions in :mod:`dmx.stats` and :mod:`dmx.bstats` share three methods,
defined on their ``ProbabilityDistribution`` base classes:

``seq_fisher_score(x, options=None)``
    Whitened score of each observation in an encoded sequence ``x``, as a
    float64 array of shape ``(N, fisher_dim(options))``, or a float64 CSR
    matrix for high-cardinality encodings. Under the
    distribution the score has mean zero and identity covariance on its
    support (a projector for over-complete encodings such as a categorical
    one-hot). Any orthonormal basis is allowed. :mod:`dmx.bstats`
    distributions evaluate it at their plug-in parameters.

    Categorical families use ``e_x / sqrt(q_x) - sqrt(q)`` over their levels,
    where ``q`` are the level probabilities, plus a trailing "other" level for
    unseen values when ``default_value > 0``. Above
    ``dense_category_limit`` levels they return the sparse offset-free form
    ``e_x / sqrt(q_x)``, which gives the same distances between records
    scored by the same distribution but differs by a constant offset, so
    distances across components are approximate.

``fisher_dim(options=None)``
    Number of score columns. Combinators sum their children, so a successful
    call means the whole tree is supported; see
    :func:`dmx.utils.fisher.has_fisher_score`.

``fisher_field_slices(options=None)``
    Field name to column slice, for attributing distances to fields. The
    default is one entry covering the whole score.

All three accept a :class:`dmx.utils.fisher.FisherScoreOptions`, which
combinators pass unchanged to their children. Parameters shared through
estimator ``keys`` do not change the definition: each occurrence is scored
with the parameters it uses.

A family that has no score raises :class:`NotImplementedError` naming its
class.

API reference
-------------

.. automodule:: dmx.utils.fisher
   :members:
