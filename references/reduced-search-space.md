# Reduced Search Space

This notebook distillation captures the main lessons from
[../notebooks/dmx_example_reduced_search_space.ipynb](../notebooks/dmx_example_reduced_search_space.ipynb):
when the search vocabulary is large, reduce the downstream search space only
after fitting one reusable shared model, keep the reductions aligned with the
actual retrieval objective, and prefer approximations that preserve posterior
structure instead of inventing a separate cheap model.

## Observation Structure

One observation is a heterogeneous record:

```python
(
    key: str,
    counter: int,
    timer: float,
    key_sequence: list[str],
)
```

The notebook rewrites `key_sequence` as value-counted data:

```python
list[tuple[str, int]]
```

after applying `count_by_value`.

That changes the modeling claim from:

- ordered sequence with possible dependence

to:

- unordered repeated-token bag with counts

This rewrite is the first and most important search-space shortcut in the
notebook. It is only safe when the sequence order is not a core part of the
signal.

## Modeling Goal

The task is not to estimate a full conditional model for every one of the
roughly 1000 keys. The goal is narrower:

- rank likely keys for a test observation
- make that ranking cheap enough to evaluate at search time
- preserve enough latent structure that search depth remains useful

That goal is what justifies search-space reduction.

## Chosen Construction

The notebook fits one shared mixture-of-composites while ignoring `key` during
estimation:

```python
word_est = CategoricalEstimator(
    suff_stat={k: 1.0 / len(vocab) for k in vocab},
    pseudo_count=1.0e-16,
)
len_est = MixtureEstimator([PoissonEstimator()] * 2, keys=("w", "comps"))
est2 = MultinomialEstimator(estimator=word_est, len_estimator=len_est)
est3 = CompositeEstimator([PoissonEstimator(), ExponentialEstimator(), est2])
est = MixtureEstimator([est3] * 15)
```

Interpretation:

- `counter` gets a count model
- `timer` gets a positive-real model
- `key_sequence` is treated as value-counted multinomial content
- one shared latent mixture is fit across all data
- per-key behavior is recovered after the fit by averaging posteriors within
  each key

This matches the repo-wide pattern: fit one reusable shared model first, then
derive key-specific views after fitting.

## Search-Space Reduction Heuristics

The notebook applies several reductions that are appropriate in this setting.

### 1. Restrict The Search Targets To The High-Frequency Head

The notebook keeps only the top 100 keys by observation count as explicit
search targets and lumps the rest into one `"non-target"` group.

Use this when:

- the vocabulary is large
- the retrieval objective mainly cares about the common or operationally
  important keys
- the long tail has too little data per key to support stable separate ranking

Why it works here:

- the top keys have enough observations to estimate stable posterior averages
- the tail would otherwise create a large expensive search set with weak
  per-key estimates

This is a downstream search reduction, not a training-data deletion strategy.
The shared mixture is still fit on all observations.

### 2. Compress Repeated Tokens Into Value Counts

The notebook uses:

```python
count_by_value(x)
```

to turn repeated token sequences into `list[tuple[key, count]]` and then fits a
`MultinomialEstimator`.

Use this when:

- the sequence is better interpreted as repeated content than as a path
- repeated values are common
- exact order is unlikely to change the search ranking much

This can drastically reduce effective sequence complexity while keeping token
frequency information.

### 3. Share A Single Nuisance Length Model

The sequence-length model inside the multinomial uses:

```python
MixtureEstimator([PoissonEstimator()] * 2, keys=("w", "comps"))
```

The notebook uses this because sequence lengths look bimodal, but length is
treated as nuisance structure that should not vary independently across every
outer mixture component.

Use this when:

- length variation is real and needs a flexible model
- but a separate length model per latent component would add cost without a
  clear retrieval benefit

### 4. Hold Out Validation Only From The Explicit Target Keys

The notebook removes two samples per target key for validation and leaves the
non-target tail in training.

Use this when:

- the tail is already data-poor
- validation is for selecting a good shared fit for the explicit search set
- removing scarce tail examples would damage the shared model more than help
  evaluation

### 5. Reuse Posteriors Instead Of Fitting Per-Key Conditional Models

After fitting the shared mixture, the notebook computes mean posterior vectors
for each target key and for one aggregated non-target bucket, then reweights
the shared components for ranking.

This is the core reduction heuristic:

- fit one global mixture
- cache per-key posterior summaries
- score test observations against those summaries

That is much cheaper than spelling out or refitting a full conditional model
for every key.

### 6. Use Cosine Similarity As A Fast Surrogate When Full Reweighting Is Costly

The notebook compares two search-time routes:

- full conditional scoring with `MixtureDistribution(..., w=key_posteriors[i])`
- cosine similarity between normalized test posteriors and normalized key
  posterior averages

The reported result is that cosine similarity performs reasonably well as a
surrogate for search depth when evaluating the full conditional probabilities
is too expensive.

Use this when:

- you already trust the fitted shared mixture
- the posterior vectors are the main signal
- search-time throughput matters more than exact probabilistic scoring

## Safe Shortcuts

These reductions are relatively safe when the notebook's assumptions hold.

### Safe: Reduce The Explicit Search Set, Not The Shared Fit

Keep the shared model trained on all available data, but only build explicit
retrieval targets for the head of the vocabulary plus one aggregated tail
bucket.

This preserves global latent structure while shrinking the search index.

### Safe: Collapse Repeated Tokens When Order Is Not The Point

If the sequence is effectively "what appeared and how often," not "what
happened in what order," value-count compression is usually a good trade.

### Safe: Use Posterior Embeddings For Search

Posterior means by key are a principled low-dimensional summary derived from
the fitted model. They are a safer approximation than training an unrelated
cheap baseline just for retrieval.

### Safe: Use A Fast Similarity Proxy After Checking It Against Search Depth

Cosine similarity is reasonable here because it is tested against the mixture
ranking rather than assumed to be equivalent a priori.

## Risky Shortcuts

These reductions become dangerous when their assumptions are false.

### Risky: Throw Away The Tail During Fitting

If you drop the non-target observations from the fit entirely, the shared
mixture loses information about the broader vocabulary and the aggregated
non-target bucket becomes poorly defined.

### Risky: Collapse Sequence Order When Order Carries Signal

`count_by_value` is only appropriate if sequential dependence is not important.
If the sequence is path-like or transition-like, use a true sequence or HMM
route instead.

### Risky: Treat The Entire Tail As One Real Class

The `"non-target"` bucket is a search convenience, not a claim that all rare
keys are one coherent population. This shortcut is acceptable for coarse search
depth evaluation but can hide important tail heterogeneity.

### Risky: Use The Head-Only Search Set When Rare Keys Matter Operationally

If the real task requires distinguishing many rare keys, top-N truncation is
the wrong optimization. It improves speed by changing the objective.

### Risky: Assume Cosine Similarity Is Always Good Enough

Cosine similarity is only a surrogate. If exact ranking quality matters, or if
posterior magnitudes carry important information beyond direction, use the full
reweighted mixture scores.

## When Search-Space Reduction Is Appropriate

Use the notebook's strategy when:

- the candidate key space is large enough that exhaustive per-key modeling or
  scoring is costly
- the task is retrieval or ranking rather than fully calibrated classification
- the heavy head of the vocabulary matters most
- a shared latent model captures reusable structure across all keys
- you can tolerate one aggregated tail bucket or another coarse search stage

Do not use it when:

- the long tail is the main business objective
- order within the sequence is central
- you need exact per-key calibration for all keys
- the compressed or truncated search target would change the real task

## Routing Rule

For large-vocabulary retrieval problems in `dmx-learn`:

1. Fit one shared composite or composite-mixture model first.
2. Reduce the search space after the fit, not by default during estimator
   construction.
3. Keep only the explicit targets that are well-supported and operationally
   relevant.
4. Recover target-specific views from posterior averages or reweighting.
5. Use cosine or another embedding-space proxy only after checking that it
   preserves search-depth behavior well enough for the task.
