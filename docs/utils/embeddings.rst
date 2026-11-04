Embedding Utilities
===================

Heterogeneous t-SNE
-------------------

.. automodule:: dmx.utils.htsne
   :members:

Heterogeneous UMAP
------------------

``humap`` embeds mixture posteriors by default. With
``representation="fisher"`` it embeds the Fisher-geometry features described
in :doc:`fisher_embedding`, which also keep each record's position inside its
component.

Fisher representation
~~~~~~~~~~~~~~~~~~~~~

Posteriors say which component a record belongs to, so with a well-fitted
model each cluster collapses to a few points. The Fisher representation adds
each record's whitened score under its most probable components, so records
are also spread out by where they sit inside their component:

.. code-block:: python

   from dmx.utils.humap import humap, humap_transform

   coords, model, fit, features, embedding = humap(
       records,
       representation="fisher",   # default "posterior"
       umap_kwargs={"random_state": 0},
       return_embedding=True,
   )
   new_coords = humap_transform(fit, model, new_records, embedding)

The features are a sparse matrix compared with the Euclidean metric. Each
field's block is balanced by its dimension by default, and ``fisher_scale``,
``top_m`` and ``fisher_options`` tune the representation. See the
:doc:`fisher_embedding` user guide for when to use it, how to choose the
options, and which distributions are supported, and
``examples/utils_examples/humap_fisher_example.py`` for a side-by-side
comparison.

Related functions:

- :func:`dmx.utils.humap.humap_transform` places new records in a fitted
  embedding.
- :func:`dmx.utils.fisher.fisher_embedding` computes the features without
  UMAP.
- :func:`dmx.mpi4py.utils.humap.humap_mpi`,
  :func:`dmx.mpi4py.utils.fisher.fisher_embedding_mpi` and
  :func:`dmx.utils.spark.fisher_embedding_rdd` compute them in parallel.

.. automodule:: dmx.utils.humap
   :members:
