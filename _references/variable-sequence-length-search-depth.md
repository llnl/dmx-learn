# Variable Sequence Length Search Depth

This notebook distillation captures the main lesson from
[../notebooks/dmx_variable_sequence_length_search_depth.ipynb](../notebooks/dmx_variable_sequence_length_search_depth.ipynb):
for paired retrieval problems with one variable-length sequence field and one
low-dimensional side feature, the mathematically valid posterior predictive
route is not automatically the best search route. Sequence length changes how
sharp the mixture posterior becomes, so search depth should be checked in both
conditioning directions before choosing the route.

## Observation Structure

One observation is a paired heterogeneous record:

```python
(
    document_tokens: list[str],
    score: float,
)
```

The notebook models this as a composite mixture:

```python
seq_dist = SequenceDistribution(
    CategoricalDistribution(...),
    PoissonDistribution(...),
)
joint_comp = CompositeDistribution((seq_dist, GaussianDistribution(...)))
model = MixtureDistribution([joint_comp] * k, w=[...])
```

Important structural facts:

- `document_tokens` is variable-length, and both token composition and sequence
  length vary by latent component.
- `score` is a single scalar side feature with much weaker component
  separation.
- the downstream task is paired search, not just density estimation

That last point matters: the notebook is about ranking the true partner near
the front of the queue, measured by search depth, not about whether the joint
model is internally coherent.

## Why Variable Sequence Length Changes The Route

The sequence side carries asymmetric evidence.

For a mixture component `Z`, the sequence posterior uses both:

- the token probabilities across every element of the sequence
- the length model `P(length | Z)`

So longer sequences usually produce a much sharper `P(Z | document_tokens)`
than a single overlapping Gaussian produces `P(Z | score)`.

That changes the retrieval route:

- ranking documents with `P(document_tokens | score)` performs poorly when the
  score only weakly updates the mixture weights
- ranking scores with `P(score | document_tokens)` performs much better when
  the sequence side is long enough to localize the latent component

The notebook's main routing lesson is therefore:

1. do not assume retrieval is symmetric across paired views
2. treat variable sequence length as part of the information budget
3. prefer the conditioning direction that uses the sharper posterior evidence

## Sequence-Length-Dependent Routing Rules

Use the notebook as guidance for these cases.

### Prefer Sequence-To-Scalar Search When Length And Content Are Informative

Prefer `P(score | document_tokens)` when:

- topic identity is mostly visible from token composition
- sequence lengths differ enough to sharpen component posteriors
- the side feature is low-dimensional or noisy

This is the notebook's winning direction because the document side identifies
the mixture topic much more reliably than the score side.

### Be Careful With Scalar-To-Sequence Search When The Sequence Side Is Richer

Treat `P(document_tokens | score)` as risky when:

- the side feature has overlapping component distributions
- the target set contains variable-length sequences
- the length model contributes meaningful component evidence

In that setting, a weak posterior over `Z` from the scalar feature is not
enough to rank the true sequence partner well.

### Re-Evaluate The Asymmetry When Sequences Are Short Or Length Is Nuisance

The notebook's asymmetry is strongest because the sequence field is informative
and variable-length. Re-check the route when:

- sequences are very short
- sequence lengths are nearly constant across components
- length should be treated as nuisance structure instead of topic evidence

In those cases the sequence posterior may no longer dominate, so either
direction could be acceptable after benchmarking.

## Search-Depth Tradeoffs

The notebook highlights three tradeoffs that matter for routing.

### 1. Probabilistic Correctness Vs Retrieval Quality

`P(document_tokens | score)` is a valid conditional distribution, but it is not
the right search objective if it yields poor search depth. For recommendation
or pairing tasks, held-out ranking behavior matters more than whether the
conditioning formula looks natural.

### 2. Richer Query Evidence Vs More Expensive Search

Using the sequence as the conditioning side gives better retrieval here, but it
also means encoding and scoring the variable-length field at query time.
That cost is often worth paying when the alternative collapses search quality.

### 3. Exact Reweighting Vs Cheap Heuristics

The notebook still uses an efficient exact route:

- encode the sequence data once
- compute component log densities for the candidate side
- compute mixture posteriors for the query side
- combine them with matrix operations instead of per-pair refits

That is a better first optimization than replacing the search score with an
unvalidated cheap proxy.

## Safe Shortcuts

These shortcuts are consistent with the notebook.

### Safe: Reuse One Shared Mixture Instead Of Refitting Conditional Models

Fit one joint composite mixture, then pop off the sequence and scalar
sub-distributions to compute the conditional search score you need.

### Safe: Benchmark Both Conditioning Directions On Search Depth

When one view is variable-length and the other is low-dimensional, compare
`P(sequence | scalar)` against `P(scalar | sequence)` on held-out pairings.
The better direction is an empirical routing decision, not a naming decision.

### Safe: Cache Encodings And Component Likelihoods

Use `seq_encode`, `seq_component_log_density`, and posterior reuse so the
search loop stays in the shared latent space rather than expanding into many
small model evaluations.

## Risky Shortcuts

These are the main failure modes from the notebook.

### Risky: Assume Posterior Predictive Ranking Is Symmetric

`P(document_tokens | score)` and `P(score | document_tokens)` can behave very
differently once one side is variable-length. Do not pick a direction just
because it matches the user-facing query wording.

### Risky: Condition On A Weak Feature To Rank Rich Variable-Length Targets

If the conditioning feature barely separates the mixture components, it will
mostly recover generic mixture weights. That causes poor search depth against a
large set of informative sequence candidates.

### Risky: Ignore Length-Driven Evidence Accumulation

Longer sequences contribute more component evidence. If that is not part of the
intended retrieval signal, uncritical use of the sequence posterior can
overstate how decisive the route really is.

### Risky: Judge The Route By Likelihood Alone

A route can be fully probabilistic and still be poor for pairing or
recommendation. Always inspect search-depth curves or equivalent ranking
metrics before standardizing the shortcut.

## Preferred Routing Rule

For paired retrieval problems with one variable-length sequence field and one
weaker side feature:

1. fit one shared joint mixture over both views
2. treat sequence length as part of the evidence budget, not as a harmless
   detail
3. compare search depth in both conditioning directions
4. prefer the route conditioned on the view that gives the sharper posterior,
   which in this notebook is the sequence-to-scalar direction
