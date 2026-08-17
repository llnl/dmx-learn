# Benchmark: PLSI

Source notebook:
[../notebooks/dmx_advanced_plsi.ipynb](../notebooks/dmx_advanced_plsi.ipynb)

## Prompt

You have a local corpus where each observation is:

```python
(
    document_id: int,
    word_counts: list[tuple[word_id: int, count: int]],
)
```

`document_id` ranges over a fixed set of known documents, and `word_id` ranges
over a fixed vocabulary. The main task after fitting is document
recommendation: compare documents using their inferred topic-weight vectors and
retrieve nearby documents with similar topics.

Recommend one primary `dmx.stats` modeling route and one meaningful generic
baseline. Explain:

1. why this observation structure can be written as a composite mixture
2. how the generic construction would be built from standard estimators
3. when the specialized estimator is the better default
4. what initialization or smoothing choices help keep the fit stable
5. how document-topic weights are recovered for recommendation in each route

## Expected Behavior

A strong answer should:

- identify the observation as fixed document id plus sparse integer
  bag-of-words counts with latent topic structure
- choose `IntegerPLSIEstimator` as the primary route when the task is standard
  PLSI-style recommendation on this exact data shape
- use a generic composite-mixture construction as the baseline:
  `MixtureEstimator([CompositeEstimator([doc_est, word_est])] * num_topics)`
- describe `doc_est` as an integer categorical model over document ids and
  `word_est` as an integer multinomial model over vocabulary counts
- mention smoothed or pooled initialization with pseudo-counts plus multiple
  `best_of` restarts before a longer `optimize` pass
- explain that the generic route needs posterior recovery of `P(Z | D)` from
  the document leaf, while the specialized route exposes those weights directly
  through `state_mat`
- explain that the specialized route wins because it matches the fixed PLSI
  structure, is faster, needs less manual inference plumbing, and is typically
  more numerically robust

## Regression Checks

Treat these as failures:

- choosing the generic composite construction as the default without explaining
  why the fixed PLSI shape makes the specialized estimator preferable
- recommending a conditional-per-document model as the primary route for this
  task
- omitting the generic composite-mixture baseline entirely
- describing recommendation as if it required refitting a separate model rather
  than using fitted document-topic weights
- failing to explain how `P(Z | D)` is obtained differently in the generic and
  specialized routes
