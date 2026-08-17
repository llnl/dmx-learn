# Process Sequences

This notebook distillation captures the main lessons from
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb):
for grouped sequence data, start from the sequence observation structure, share
the latent parts that should be reusable across users, key sequence length only
when it is mostly nuisance structure, and choose between a sequence-of-mixtures
route and an HMM route based on whether within-sequence dependence matters.

## Observation Structure

One training observation is:

```python
(
    user_id: str,
    process_sequence: list[tuple[float, str]],
)
```

where each `process_sequence` is ordered and variable-length.

The inner tuple is heterogeneous:

- `float`: process time
- `str`: process name

The notebook models each sequence element with a composite estimator:

```python
est_tuple = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
```

and then wraps that element model in either:

- `SequenceEstimator(...)` when the sequence is treated as conditionally
  independent given a shared latent topic mixture
- `HiddenMarkovEstimator(...)` when the order dependence itself is part of the
  modeling claim

So the structure-first decision is:

1. one observation is a grouped sequence, not a flat row
2. each sequence element is a heterogeneous tuple
3. `user_id` is a grouping key for sharing or post-fit aggregation, not just
   another categorical leaf

## Sequence Routes Compared

The notebook walks through three increasingly useful routes.

### 1. Separate Conditional Sequence Models Per User

```python
est_proc_seqs = SequenceEstimator(estimator=est_tuple, len_estimator=len_est)
est = ConditionalDistributionEstimator(
    estimator_map={user_id: est_proc_seqs for user_id in user_ids}
)
```

This is the simplest baseline: one sequence model per user.

Why it fails as a primary route:

- unseen process names inside one user's training data produce `-inf`
  log-likelihood for otherwise plausible target sequences
- there is no shared topic structure across users
- the model is rigid when users should borrow strength from one another

The notebook stabilizes this route with:

- categorical pseudo-count smoothing for process names
- a flattened categorical `suff_stat`
- a keyed shared length estimator

That fixes the zero-support issue, but it still keeps the sequence-content
model mostly separate by user.

### 2. Conditional Sequence Of Shared Mixtures

```python
est_mix = MixtureEstimator([est_tuple] * 5, keys=(None, "mix_comps"))
len_est = IntegerCategoricalEstimator(min_val=5, max_val=8, keys="length_dist")
est_proc_seqs = SequenceEstimator(estimator=est_mix, len_estimator=len_est)
est = ConditionalDistributionEstimator(
    estimator_map={user_id: est_proc_seqs for user_id in user_ids}
)
```

This is the main shared-topic sequence route in the notebook.

Interpretation:

- each user gets its own mixture weights
- the mixture components are shared across users
- the sequence-length law is also shared
- the sequence still ignores explicit within-sequence state dependence

This is the right route when:

- users or groups appear to draw from the same process motifs or topics
- you want "same topics, different proportions"
- order inside the sequence is less important than content frequencies
- you want explicit user-conditional scoring after the fit

### 3. Mixture Of HMMs With Post-Fit Group Aggregation

```python
len_est = CategoricalEstimator(keys="len_dist")
seq_est = HiddenMarkovEstimator(
    estimators=[est_tuple] * 10,
    len_estimator=len_est,
    keys=(None, None, "topics"),
)
est = MixtureEstimator([seq_est] * 5)
```

This route changes two things:

- it models sequential dependence through HMM state transitions
- it drops the explicit per-user conditional wrapper during fitting

Instead of fitting `P(X | user)` directly, the notebook:

1. fits one shared mixture of HMMs to all sequences
2. computes posterior embeddings for each observed sequence
3. averages those embeddings within each user
4. uses those user embeddings to rank likely aliases for new target sequences

This is the better route when:

- order and local transition behavior are likely informative
- you want one reusable sequence model instead of a large `estimator_map`
- user-specific behavior can be recovered after the fit from posterior
  summaries

## Keyed Sharing Strategy

The notebook shows two different sharing claims.

### Share Topics Across Users Or Groups When The Latent Building Blocks Match

Use keyed topic or component sharing when users are expected to reuse the same
underlying process motifs, but in different proportions.

Sequence-of-mixtures route:

```python
MixtureEstimator([est_tuple] * 5, keys=(None, "mix_comps"))
```

Meaning:

- do not share mixture weights
- do share the component subtree

HMM route:

```python
HiddenMarkovEstimator(..., keys=(None, None, "topics"))
```

Meaning:

- do not share initial-state weights
- do not share transition matrices
- do share emission topics

Share topics across users or groups when:

- the same process names or time/name motifs recur across groups
- per-group sample sizes are modest and pooling improves stability
- the main difference between groups is which motifs are emphasized, not what
  the motifs are

Do not share topics when:

- groups have genuinely different support or incompatible latent states
- the emissions themselves, not just the weights, should vary by group

### Key Sequence Length When Length Is Common Nuisance Structure

The notebook keys the length estimator in both the `SequenceEstimator` and HMM
routes because sequence lengths fall in a small common range and are not the
main discriminative signal.

Use a keyed `len_estimator` when:

- sequence length is similar across users or groups
- length mostly regularizes the model instead of defining the groups
- you want a stabler shared estimate for a short or sparse length histogram

Keep sequence length unshared when:

- sequence length is one of the main ways groups differ
- one group systematically has longer or shorter sequences and that difference
  matters downstream

## Post-Fit Usage

The notebook shows two post-fit patterns.

### Direct User Scoring From A Conditional Sequence Model

For the conditional routes, encode the target sequences and score them under
each user's fitted sequence model:

```python
enc_target = seq_encode(target, model=mm.dmap["user_0"])[0][1]
ll[i] = mm.dmap[f"user_{i}"].seq_log_density(enc_target)
```

This is the simplest retrieval workflow when the model was fit as
`P(sequence | user)`.

### Posterior Embeddings For User Retrieval After A Shared HMM Fit

For the HMM route, the fit is user-agnostic. User-specific behavior is
recovered by averaging posterior mixture embeddings within each user:

```python
user_embeddings[user_idx] += np.mean(mm.seq_posterior(v), axis=0)
```

Then new target sequences are embedded under the shared model and compared to
the user embeddings for ranking or nearest-neighbor style search.

This is the important post-fit lesson: if a reusable shared sequence model is
available, grouped retrieval can be built from posterior summaries instead of a
separate conditional refit.

## Preferred Routing Rule

Use this notebook as the routing guide for grouped sequence data:

1. Start from `list[tuple[time, process_name]]` as the observation structure.
2. Use a per-user conditional sequence model only as a baseline.
3. Prefer shared topics across users when the same latent process motifs should
   appear in multiple groups.
4. Key sequence length only when length is mostly shared nuisance structure.
5. Choose a sequence-of-mixtures route when shared content matters more than
   explicit transition dependence.
6. Choose an HMM route when order and transitions matter enough to justify a
   more structured fit and post-fit user aggregation.

## Tradeoff Summary

- Separate conditional sequence models are simple but brittle and data-hungry.
- Shared-topic sequence mixtures are a strong default for grouped sequence data
  when order dependence is weak or secondary.
- HMM-style routes are preferable when temporal dependence is part of the
  signal, but they usually shift user-specific modeling to a post-fit embedding
  or reweighting step rather than an explicit conditional wrapper.
