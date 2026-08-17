# PLSI

This notebook distillation captures the main lesson from
[../notebooks/dmx_advanced_plsi.ipynb](../notebooks/dmx_advanced_plsi.ipynb):
plain `dmx.stats` combinators are enough to express PLSI as a composite
mixture, but when the data really are fixed-document-id plus integer
bag-of-words counts, `IntegerPLSIEstimator` is the better default because it
fits the same structure faster and exposes the document-topic quantities
directly.

## Observation Structure

One observation is:

```python
(
    document_id: int,
    word_counts: list[tuple[word_id: int, count: int]],
)
```

Structural facts that drive the model choice:

- `document_id` comes from a fixed finite set.
- `word_counts` is a sparse integer bag-of-words over a fixed vocabulary.
- the latent structure is topic-style: each document has topic weights, and
  each topic has a word distribution
- the downstream task is document recommendation through document-topic
  weights, not an arbitrary heterogeneous prediction problem

## Generic Composite-Mixture Construction

The notebook first shows how to recover the PLSI factorization with standard
`dmx.stats` pieces:

```python
doc_est = IntegerCategoricalEstimator(
    min_val=0,
    max_val=num_documents,
    pseudo_count=1.0,
    suff_stat=(0, np.ones(num_documents) / num_documents),
)

word_est = IntegerMultinomialEstimator(
    min_val=0,
    max_val=num_words,
    pseudo_count=1.0,
    suff_stat=(0, np.ones(num_words) / num_words),
)

est = MixtureEstimator(
    [CompositeEstimator([doc_est, word_est])] * num_topics
)
```

Interpretation:

- each outer mixture component is one topic
- inside each topic, one leaf models `P(D=d | Z=z)`
- the other leaf models `P(W | Z=z)` for sparse integer word counts

After fitting, recover the document-topic weights by popping off the document
sub-distributions and applying Bayes:

```python
doc_comps = [comp.dists[0] for comp in fit.components]
doc_mix = MixtureDistribution(doc_comps, fit.w)
doc_weights = doc_mix.seq_posterior(encoded_doc_ids)
```

So the generic route is fully package-native and statistically correct, but it
pushes some PLSI-specific posterior plumbing onto the caller.

## Specialized PLSI Route

When the data match the fixed integer PLSI shape directly, use the specialized
estimator:

```python
est = IntegerPLSIEstimator(
    num_vals=num_words,
    num_states=num_topics,
    num_docs=num_documents,
    pseudo_count=(1, 1, 1),
)
```

The fit pattern is the same as elsewhere in the repo:

```python
_, mm_plsi = best_of(
    data=wiki_data,
    est=est,
    trials=5,
    init_p=0.10,
    vdata=wiki_data,
    delta=1.0e-6,
    max_its=100,
    init_estimator=est,
    rng=np.random.RandomState(10),
)

plsi_model = optimize(
    data=wiki_data,
    estimator=est,
    prev_estimate=mm_plsi,
    delta=1.0e-6,
    max_its=1000,
)
```

The downstream quantities are then available directly:

- `plsi_model.state_mat` gives `P(Z=z | D=d)` for every document
- `plsi_model.doc_vec` stores document probabilities
- `plsi_model.prob_mat` stores topic-word probabilities

## When Generic Is Enough

Use the composite-mixture construction when:

- you want to prototype the topic model from standard building blocks first
- you need to stay inside a larger generic composite model
- the data only approximately look like PLSI and may soon gain extra fields or
  different local structure
- fitting speed is acceptable and a little manual posterior extraction is not a
  problem

This route is important because it proves the specialized estimator is not a
different statistical idea. It is a more efficient implementation of the same
basic topic-style construction.

## Why And When The Specialized Route Wins

Prefer `IntegerPLSIEstimator` when the model really is "fixed doc id plus
integer bag-of-words counts with latent topics."

Why it wins in the notebook:

- it is simpler to specify than spelling the model out with generic
  combinators
- it fits faster than the conditional and composite alternatives
- it avoids the extra posterior-recovery step needed by the composite route
- it gives slightly better recommendation behavior in the notebook's neighbor
  search plots
- it handles the numeric and memory-management details more robustly for this
  exact data format

The practical routing rule is:

1. use the generic composite mixture when you need flexibility
2. switch to `IntegerPLSIEstimator` when the observation really is the PLSI
   shape and repeated fitting or inference cost matters

## Recommendation Workflow

For document recommendation, compare documents by cosine similarity of their
document-topic vectors.

- generic route: derive those vectors by posterior reweighting from the
  document leaf
- specialized route: use `state_mat` directly

That direct access is part of the specialized estimator's value: the notebook's
main downstream object is exactly the quantity that the specialized model
stores natively.
