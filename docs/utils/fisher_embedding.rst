Fisher-Geometry Embeddings
==========================

:func:`dmx.utils.humap.humap` embeds records with UMAP applied, by default,
to the posterior probabilities of a fitted mixture. Posteriors say which
component a record belongs to but not where it sits inside that component:
with a well-fitted model most posteriors are close to one-hot, so every record
in a component is at distance close to zero from every other one, and the
shape of each cluster in the embedding is arbitrary.

The Fisher-geometry representation keeps that information. It compares
records through small, record-specific perturbations of the fitted model, and
reduces exactly to the posterior representation when the perturbation is
switched off. It is computed per record, so it scales and distributes the
same way posteriors do. Select it with ``humap(data,
representation="fisher")``; see `Using it with humap`_.

.. note::

   Distributions without a score yet raise :class:`NotImplementedError`;
   pass ``FisherScoreOptions(on_missing="posterior_only")`` to keep only their
   posterior information. The plan is tracked in
   ``HUMAP_FISHER_ROADMAP.md``.

User guide
----------

.. code-block:: python

   from dmx.utils.humap import humap, humap_transform

   coords, model, fit, features, embedding = humap(
       records, representation="fisher", return_embedding=True
   )
   new_coords = humap_transform(fit, model, new_records, embedding)

``examples/utils_examples/humap_fisher_example.py`` compares posterior and
Fisher embeddings side by side on synthetic and repository data.

When to use it
~~~~~~~~~~~~~~

Use ``representation="fisher"`` when the structure *inside* clusters matters:
nearby records within a component should land near each other, and the
layout of each cluster should mean something. Posterior humap keeps only which
component a record belongs to. With a well-separated model most posteriors are
nearly one-hot, so each cluster collapses to a few points. When many records
share exactly the same posteriors, UMAP can even fail to initialize. In the
benchmark (``_benchmarks/fisher-humap.md``), Fisher humap recovered the
within-cluster structure posterior humap lost, and it embedded 10⁶ records
where posterior humap's UMAP fit failed.

Posterior humap remains the default and is enough when you only need to see
which records cluster together. It is also the fallback when the model has
components without a Fisher score; see `Unsupported components`_.

Choosing the options
~~~~~~~~~~~~~~~~~~~~

``fisher_scale`` (default ``None``, meaning :math:`c = 1/p_{\text{eff}}`)
    Weight of the within-component score blocks against the posterior block.
    Neighbour rankings inside a component do not depend on it; it only sets
    how many within-component standard deviations count as much as a change
    of component. Larger values spread each cluster out relative to the gaps
    between clusters, and ``0`` reproduces the posterior geometry. The default
    is a good start: in the benchmark, quality was flat for :math:`c` between
    0.02 and 1.

``top_m`` (default 3)
    Number of most probable components whose score blocks each record keeps.
    Components further down have posteriors so small that their blocks barely
    change distances, so 3 is exact in practice. Raise it only for models
    whose components overlap heavily, so that records often have four or more
    non-negligible posteriors. ``None`` keeps all of them at a higher memory
    cost.

``fisher_options.balance_fields`` (``humap`` default ``True``)
    Divides each field's block by the square root of its dimension. Without
    it, every parameter counts equally, so a field with many parameters, such
    as a categorical with many levels or a long sequence of categoricals,
    can swamp the numeric fields. ``humap`` and ``humap_mpi`` balance fields
    when ``fisher_options`` is None, which did best in the benchmark. Pass
    ``FisherScoreOptions()`` for the pure Fisher weighting, for example when
    every field has a similar number of parameters.

``fisher_options.sequence_scaling`` (default ``"conditional"``)
    For sequences, sets and count vectors. ``"conditional"`` compares records
    by the content of their elements regardless of length, and length
    differences come only from the length distribution's own score.
    ``"marginal"`` also lets longer records sit farther from the center, so
    length separates records more strongly. See `Variable-length records`_.

``fisher_options.prob_floor`` and ``dense_category_limit``
    ``prob_floor`` bounds the cost of rare or unseen categorical levels.
    Categoricals with more than ``dense_category_limit`` levels (default 256)
    use a sparse score, so high-cardinality fields stay cheap.

Variable-length data
~~~~~~~~~~~~~~~~~~~~

Sequences, sets, multinomial counts and Markov chains are supported. Each
record's element scores are summed and scaled by its length (see
``sequence_scaling``), and the length distribution adds its own score block.
Nested mixtures and hidden Markov models are whitened empirically with
deterministic draws from the model, so their scores do not depend on the
batch being embedded. See `Variable-length records`_ and
`Latent-variable components`_.

Unsupported components
~~~~~~~~~~~~~~~~~~~~~~

:func:`dmx.utils.fisher.has_fisher_score` tells whether a whole model tree
has scores. A component without one raises :class:`NotImplementedError` by
default. With ``fisher_options=FisherScoreOptions(on_missing="posterior_only",
balance_fields=True)`` such a component contributes only its posterior and a
single warning names it; the other components keep their score blocks.

Supported distributions
~~~~~~~~~~~~~~~~~~~~~~~

This table is generated from the Fisher contract-test registries, so every
listed family passes the score contract. Families not listed, such as
von Mises-Fisher, integer sets, edit-distance sets, LDA, PLSI, tree HMMs and
conditional distributions, raise until they are implemented.

.. fisher-support-table-start

.. list-table:: Distributions with Fisher scores
   :header-rows: 1
   :widths: 30 40 15 15

   * - Group
     - Distribution
     - ``dmx.stats``
     - ``dmx.bstats``
   * - Numeric leaves
     - ``ExponentialDistribution``
     - yes
     - yes
   * - Numeric leaves
     - ``GammaDistribution``
     - yes
     - yes
   * - Numeric leaves
     - ``GaussianDistribution``
     - yes
     - yes
   * - Numeric leaves
     - ``LogGaussianDistribution``
     - yes
     - -
   * - Count leaves
     - ``BernoulliDistribution``
     - -
     - yes
   * - Count leaves
     - ``BinomialDistribution``
     - yes
     - -
   * - Count leaves
     - ``GeometricDistribution``
     - yes
     - yes
   * - Count leaves
     - ``PoissonDistribution``
     - yes
     - yes
   * - Categorical leaves
     - ``CategoricalDistribution``
     - yes
     - yes
   * - Categorical leaves
     - ``IntegerCategoricalDistribution``
     - yes
     - -
   * - Multivariate leaves
     - ``DiagonalGaussianDistribution``
     - yes
     - -
   * - Multivariate leaves
     - ``DirichletDistribution``
     - yes
     - -
   * - Multivariate leaves
     - ``MultivariateGaussianDistribution``
     - yes
     - -
   * - Combinators
     - ``CompositeDistribution``
     - yes
     - yes
   * - Combinators
     - ``IgnoredDistribution``
     - yes
     - yes
   * - Combinators
     - ``NullDistribution``
     - -
     - yes
   * - Combinators
     - ``OptionalDistribution``
     - yes
     - yes
   * - Variable-length
     - ``BernoulliSetDistribution``
     - yes
     - -
   * - Variable-length
     - ``IntegerMultinomialDistribution``
     - yes
     - -
   * - Variable-length
     - ``MarkovChainDistribution``
     - yes
     - -
   * - Variable-length
     - ``MultinomialDistribution``
     - yes
     - -
   * - Variable-length
     - ``SequenceDistribution``
     - yes
     - yes
   * - Latent (empirically whitened)
     - ``HeterogeneousMixtureDistribution``
     - yes
     - -
   * - Latent (empirically whitened)
     - ``HiddenMarkovModelDistribution``
     - yes
     - -
   * - Latent (empirically whitened)
     - ``IntegerHiddenMarkovModelDistribution``
     - yes
     - -
   * - Latent (empirically whitened)
     - ``MixtureDistribution``
     - yes
     - -

.. fisher-support-table-end

Large data
~~~~~~~~~~

Building the features is cheap: about a second for 10⁶ records on one core.
UMAP dominates the cost. For data that does not fit one process, use
:func:`dmx.mpi4py.utils.humap.humap_mpi` (which calls
:func:`dmx.mpi4py.utils.fisher.fisher_embedding_mpi`) or
:func:`dmx.utils.spark.fisher_embedding_rdd`; see `Distributed computation`_.
:func:`dmx.utils.fisher.fisher_embedding` computes the features without UMAP.

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
``humap`` uses by default.

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

Variable-length records
-----------------------

Sequences, multinomial count vectors, and sets have a different number of
elements per record. A sequence of ``n`` independent elements has score

.. math::

   \bigl[\textstyle\sum_{t=1}^{n} s_{\text{elem}}(x_t) / \mathrm{scale}(n),\;
   s_{\text{len}}(n)\bigr],

where :math:`s_{\text{len}}` is the length distribution's own score (omitted
when the length distribution is null). The summed element score has
covariance :math:`n I`, and ``FisherScoreOptions.sequence_scaling`` decides how
it is whitened:

``"conditional"`` (default)
    :math:`\mathrm{scale}(n) = \sqrt{n}`. The element block is whitened given
    the observed length, so records are compared by the content of their
    elements regardless of length; length differences show up only in the
    length block. An empty record has a zero element block.

``"marginal"``
    :math:`\mathrm{scale}(n) = \sqrt{E[n]}` under the length distribution. The
    element block is whitened on average, so longer records sit farther from
    the origin and length also separates records through the element block.
    It needs the mean of the length distribution, which is known for
    Poisson, geometric, binomial, integer-categorical, and categorical
    lengths.

Multinomials use the same rule with the total count in place of ``n``; for a
categorical element with conditional scaling the count block is
:math:`(n_c - N p_c) / \sqrt{N p_c}`.

Latent-variable components
--------------------------

A mixture used as a component, for example as a field of a composite, has no
closed-form Fisher information. With posteriors :math:`z_j`, weights
:math:`w_j`, and component scores :math:`s_j`, its raw score is

.. math::

   \bigl[(z_j - w_j) / \sqrt{w_j},\; z_j\, s_j(x) / \sqrt{w_j}\bigr]_{j=1}^{J},

the observed-data score scaled by the complete-data Fisher information. Its
mean is exactly zero, but the complete-data information over-states the
observed information, so its covariance is below the identity where
components overlap. It is therefore whitened with
:class:`dmx.utils.fisher.EmpiricalFisherWhitener` (symmetric, "ZCA"),
fitted on ``FisherScoreOptions.whitening_samples`` draws from the mixture
itself with ``whitening_seed``. The whitening depends only on the model, so
scoring records in chunks or scoring new records gives the same rows. The
weight block sums to zero, so the result has a projector covariance with one
null direction.

With ``on_missing="posterior_only"``, components without a score contribute
only through the weight block, and one warning per call lists their
classes; with the default ``"raise"`` they raise
:class:`NotImplementedError`.

Hidden Markov models (Gaussian-style and integer emissions) use per-sequence
forward-backward posteriors: the raw score stacks the initial block
:math:`\gamma_0 - w`, the transition block
:math:`\xi_{ij} - (\sum_j \xi_{ij}) A_{ij}` from the expected transition counts,
the emission blocks :math:`\sum_t \gamma_t(k)\, s_k(x_t)`, and the length
score, and is whitened the same way. With conditional scaling the transition
block is divided by :math:`\sqrt{T - 1}` and the emission blocks by
:math:`\sqrt{T}`; marginal scaling leaves them unscaled, since whitening makes
a constant factor irrelevant. Whitening draws sequences from the model, so an
HMM needs a length distribution (or terminal values) to be scored. HMMs with
``taus`` (topic mixtures) have no score yet.

Mixture embedding
-----------------

``MixtureDistribution.seq_fisher_embedding`` (on both :mod:`dmx.stats` and
:mod:`dmx.bstats` mixtures) assembles the feature map :math:`\phi` for an
encoded sequence and returns a :class:`dmx.utils.fisher.FisherEmbedding`:

.. code-block:: python

   from dmx.utils.fisher import fisher_embedding

   embedding = fisher_embedding(model, records, chunk_size=10_000)
   embedding.matrix             # CSR float32, one row per record
   embedding.scale              # the c used
   embedding.component_slices   # {k: columns of component k's score block}
   embedding.field_slices       # {k: {field: columns}}

   new_rows = fisher_embedding(model, new_records, like=embedding)

- Columns ``0..K-1`` hold :math:`\sqrt{z_i}`; with ``scale=0`` nothing else
  is stored, which is the current posterior representation.
- Each component with nonzero weight and a score has a block of
  ``fisher_dim`` columns holding :math:`\sqrt{c\, z_{ik}}\, s_{ik}`, kept only
  for the ``top_m`` most probable components of each record (default 3;
  None keeps all).
- The default scale is :math:`c = 1 / p_{\text{eff}}` with
  :math:`p_{\text{eff}} = \operatorname{mean}_i \sum_k z_{ik} \dim_k` over the
  records embedded.
- :func:`dmx.utils.fisher.fisher_embedding` encodes the records and embeds
  them in chunks; rows are identical to a single unchunked call. With the
  default scale it makes one extra pass over the chunks to compute
  :math:`p_{\text{eff}}`.
- ``whiten="full"`` (or ``"diag"``, ``"pca"``) additionally whitens each
  component's score with an :class:`dmx.utils.fisher.EmpiricalFisherWhitener`
  fitted on the records, weighted by their posteriors. This corrects
  scores whose model covariance is off, for example plug-in
  :mod:`dmx.bstats` parameters. The moments are summed over chunks and the
  whiteners are stored on the embedding.
- ``like=embedding`` reuses an embedding's scale, ``top_m``, options and
  whiteners, so new records are placed in the same feature space, as needed
  for ``UMAP.transform``.

Using it with humap
-------------------

``humap`` and ``humap_mpi`` take keyword-only options for the Fisher
representation; the default stays ``representation="posterior"``:

.. code-block:: python

   from dmx.utils.humap import humap, humap_transform

   coords, model, fit, features, embedding = humap(
       records,
       representation="fisher",   # default "posterior"
       fisher_scale=None,         # c; None means 1 / p_eff, 0 means posteriors
       top_m=3,                   # score blocks kept per record
       fisher_options=None,       # None means balance_fields=True
       umap_kwargs={"random_state": 0},
       return_embedding=True,     # append the FisherEmbedding
   )
   new_coords = humap_transform(fit, model, new_records, embedding)

- With ``fisher_options=None``, ``humap`` and ``humap_mpi`` use
  :data:`dmx.utils.humap.HUMAP_FISHER_OPTIONS`, which sets
  ``balance_fields=True``: each field's block is divided by the square root
  of its dimension, so a many-level categorical does not swamp numeric
  fields. It recovered within-cluster structure best in the benchmark
  (``_benchmarks/fisher-humap.md``). Pass ``FisherScoreOptions()`` for the
  pure Fisher weighting. Other callers of the scores and of
  :func:`~dmx.utils.fisher.fisher_embedding` keep ``balance_fields=False``.
- The default UMAP metric becomes ``"euclidean"``; a ``metric`` set in
  ``umap_kwargs`` is kept. The CSR feature matrix is passed to UMAP directly
  and returned as the fourth value, where posterior mode returns the
  posteriors.
- ``progress_callback`` receives an ``"embedding"`` event with the scale,
  ``p_eff``, ``top_m``, ``balance_fields`` and feature dimension, which are
  also logged at DEBUG.
- :func:`dmx.utils.humap.humap_transform` computes the representation the fit
  used (pass ``embedding=None`` for a posterior fit) and calls
  ``fit.transform``.
- ``humap_mpi`` scatters the records from rank 0, computes the features with
  :func:`dmx.mpi4py.utils.fisher.fisher_embedding_mpi`, and fits UMAP on
  rank 0.

Distributed computation
-----------------------

Rows are computed per record, so the embedding distributes like posteriors.
Only two quantities need records from every process, and both are reduced
as small sums:

- the default scale needs :math:`\sum_i p_i` and the record count. The sum
  is accumulated exactly, as an integer multiple of :math:`2^{-1126}`, so it
  is the same however the records are split, and :math:`p_{\text{eff}}` is
  its correctly rounded mean;
- ``whiten`` needs each component's posterior-weighted score moments, one
  float64 vector per process.

:func:`dmx.mpi4py.utils.fisher.fisher_embedding_mpi` runs on every rank with
that rank's records, reduces each quantity with one ``allreduce``, and
gathers the rows to ``root`` (or leaves them on each rank with
``gather=False``):

.. code-block:: python

   from mpi4py import MPI
   from dmx.mpi4py.utils.fisher import fisher_embedding_mpi

   embedding = fisher_embedding_mpi(model, local_records, MPI.COMM_WORLD)

:func:`dmx.utils.spark.fisher_embedding_rdd` reduces the same quantities with
Spark jobs and returns a lazy RDD of ``(index, row)`` pairs together with a
row-less :class:`~dmx.utils.fisher.FisherEmbedding` that holds the scale,
slices and whiteners:

.. code-block:: python

   from dmx.utils.spark import fisher_embedding_rdd

   rows, layout = fisher_embedding_rdd(model, records_rdd)

Without ``whiten``, both reproduce the rows of
:func:`~dmx.utils.fisher.fisher_embedding` on all records bit for bit,
provided each component evaluates a record the same way whatever batch it is
in. Fisher scores and whitener transforms avoid BLAS for this reason. The log
densities of high-dimensional multivariate Gaussians use a blocked LAPACK
solve and can differ in the last bit between batch sizes, which carries over
to the posteriors. With ``whiten`` the moments are summed in floating point,
so rows agree only up to summation order.

API reference
-------------

.. automodule:: dmx.utils.fisher
   :members:
