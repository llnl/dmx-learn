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

.. automodule:: dmx.utils.humap
   :members:
