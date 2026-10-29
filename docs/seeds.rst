.. _seeds:

Seeds and reproducibility
=========================

Every stochastic entry point in dmx-learn accepts a ``seed`` (an integer) or an
``rng`` (a ``numpy.random.RandomState``), and none of them falls back to
NumPy's global random state. The rules are the same everywhere:

- **Same seed, same result.** Calling an entry point twice with the same
  ``seed`` and the same inputs gives identical results. ``seed=s`` is the same
  random stream as ``rng=numpy.random.RandomState(s)``.
- **A caller-owned** ``rng`` **is used as is**, so existing code that passes an
  ``rng`` gets exactly the results it got before.
- **With neither, a fresh seed is drawn and reported.** The seed comes from
  operating-system entropy (:func:`dmx.utils.seeding.new_seed`), builds a
  private ``RandomState``, and is reported so the run can be repeated by passing
  it back. Pass at most one of ``seed`` and ``rng``; passing both raises
  ``ValueError``.

How the seed used is reported:

- Functions with a ``progress_callback`` send a
  :class:`~dmx.utils.progress.ProgressEvent` with ``event="start"`` and
  ``metrics={"seed": seed}`` before any other event, and log
  ``"<function> seed=<seed>"`` at ``DEBUG`` on the module's logger. Nothing is
  added to the default printed output. The seed is ``None`` when the caller
  passed its own ``rng``; nested calls receive their caller's random state, so
  the outermost call's start event carries the seed.
- Samplers store it as ``sampler.seed``, so
  ``dist.sampler(seed=sampler.seed)`` reproduces an unseeded sampler.
- Functions without a callback log it at ``DEBUG``.

.. code-block:: python

   events = []
   model = optimize(data, estimator, verbose=False, progress_callback=events.append)
   seed = events[0].metrics["seed"]
   same = optimize(data, estimator, verbose=False, seed=seed)  # identical fit

Entry point audit
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Entry point
     - Seed parameters
     - Seed used is exposed by
   * - ``dmx.utils.estimation.optimize``, ``iterate``
     - ``rng=None``, ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.estimation.best_of``
     - ``rng`` (positional, may be ``None``), ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.estimation.hill_climb``
     - none (deterministic)
     - not applicable
   * - ``dmx.utils.estimation.k_fold_split_index``, ``partition_data``,
       ``partition_data_index``
     - ``rng`` (required)
     - caller-owned
   * - ``dmx.stats.seq_initialize``, ``initialize``
     - ``rng`` (required)
     - caller-owned
   * - ``dmx.bstats.bestimation.optimize``, ``iterate``
     - ``rng=None``, ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.bstats.bestimation.best_of``
     - ``rng`` (positional, may be ``None``), ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.bstats.bestimation.hill_climb``
     - none (deterministic)
     - not applicable
   * - ``dmx.utils.automatic.get_dpm_mixture``
     - ``rng=None``, ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.automatic.prepare_mixture_model``
     - ``rng`` (positional, may be ``None``), ``seed=None``
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.htsne.htsne``
     - ``seed=None`` (mixture fit and initial coordinates)
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.htsne.dpmsne``
     - ``seed=None`` (initial coordinates)
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.humap.humap``
     - ``seed=None`` (mixture fit); UMAP uses ``umap_kwargs["random_state"]``
     - ``"start"`` event and DEBUG log
   * - ``dmx.utils.scores.kfold_log_likelihood``
     - ``seed=0`` (folds and per-fold fits)
     - the ``seed`` field of the result
   * - ``dmx.utils.vector.row_choice``
     - ``rng`` (may be ``None``)
     - ``None`` uses a private entropy-seeded state, not reported
   * - ``dmx.mpi4py.utils.estimation.optimize_mpi``,
       ``dmx.mpi4py.utils.bestimation.optimize_mpi``
     - ``rng=None``, ``seed=None``; a drawn seed is broadcast from rank 0
     - ``"start"`` event and DEBUG log on rank 0
   * - ``dmx.mpi4py.utils.estimation.best_of_mpi``
     - ``rng`` (positional, may be ``None``), ``seed=None``; broadcast from
       rank 0
     - ``"start"`` event and DEBUG log on rank 0
   * - ``dmx.mpi4py.utils.automatic.get_dpm_mixture_mpi``
     - ``rng=None``, ``seed=None`` (passed to the optimizer)
     - the optimizer's ``"start"`` event on rank 0
   * - ``dmx.mpi4py.utils.humap.humap_mpi``
     - ``seed=None``; broadcast from rank 0; UMAP uses
       ``umap_kwargs["random_state"]``
     - ``"start"`` event and DEBUG log on rank 0
   * - ``dmx.torch_utils.estimation.optimize``, ``optimize_mp``
     - ``seed=None`` (``optimize_mp`` uses rank 0's value)
     - DEBUG log (rank 0 for ``optimize_mp``)
   * - Samplers: ``dist.sampler(seed)`` in ``dmx.stats``, ``dmx.bstats``, and
       ``dmx.torch_stats``
     - ``seed=None``; array-like seeds accepted by NumPy still work
     - ``sampler.seed``
   * - ``dmx.stats.rdd_sampler.take_sample``, ``sample_rdd``,
       ``sample_seq_as_rdd``
     - ``seed=None``
     - DEBUG log

Helpers
-------

.. automodule:: dmx.utils.seeding
   :members:
