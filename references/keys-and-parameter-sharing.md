# Keys And Parameter Sharing

`keys` are how `dmx-learn` tells multiple estimators or distributions to share
parameters. They are not a cosmetic label. A repeated key means "aggregate
sufficient statistics into the same parameter update." No key means "fit this
part separately."

This matters most in composite, mixture, conditional, and sequence models where
you often want a shared latent vocabulary, topic set, or length model without
forcing every other parameter to be identical.

## What `keys` Do

The package uses `keys` at the parameter slot level.

- Simple estimators often take one key string, for example `PoissonEstimator(keys="rate")`.
- Structured estimators often take a tuple of keys, one per shareable parameter
  block. For example, `DiagonalGaussianEstimator(keys=("mu", "covar"))` can
  share the mean, the covariance, or both.
- `None` means "do not share this slot."

The practical rule is:

1. Put the same key on subtrees that should learn from pooled evidence.
2. Use different keys, or `None`, where groups should keep separate parameters.
3. Share the smallest subtree that matches your modeling claim.

The tests in [../tests/stats/dmvn_test.py](../tests/stats/dmvn_test.py),
[../tests/stats/heterogeneous_mixture_test.py](../tests/stats/heterogeneous_mixture_test.py),
and [../tests/stats/hidden_markov_test.py](../tests/stats/hidden_markov_test.py)
show this slot-by-slot pattern directly.

## Subtree Sharing Patterns

### Share A Leaf Parameter

Use this when only one local parameter should be common across many models.

Example:

```python
DiagonalGaussianEstimator(keys=("mu", None))
DiagonalGaussianEstimator(keys=(None, "covar"))
```

This is the smallest possible sharing claim: only the named slot is pooled.
See [../tests/stats/dmvn_test.py](../tests/stats/dmvn_test.py).

### Share Mixture Components But Not Weights

This is one of the most useful patterns in the repo.

```python
est_mix = MixtureEstimator([est_tuple] * 5, keys=(None, "mix_comps"))
```

The component distributions are shared, but each parent model still gets its
own mixture weights. Conceptually: "same topics, different proportions."

That is exactly the pattern used in
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
and
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb).
The lower-level API shape also appears in
[../tests/stats/heterogeneous_mixture_test.py](../tests/stats/heterogeneous_mixture_test.py)
with `keys=(None, "comps")`.

### Share Emissions Or Topics Inside A Larger Sequence Model

For HMM-style models, you often want shared emissions but different initial
state weights or transition matrices.

```python
seq_est = HiddenMarkovEstimator(
    estimators=[est_tuple] * 10,
    len_estimator=len_est,
    keys=(None, None, "topics"),
)
```

This says:

- do not share the initial-state weights
- do not share the transition matrix
- do share the emission/topic subtree

That pattern appears in
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb)
and the slot variants are exercised in
[../tests/stats/hidden_markov_test.py](../tests/stats/hidden_markov_test.py).

### Share Sequence-Length Estimators

Length is often its own estimand and should be treated explicitly.

```python
len_est = IntegerCategoricalEstimator(min_val=5, max_val=8, keys="length_dist")
est_proc_seqs = SequenceEstimator(estimator=est_mix, len_estimator=len_est)
```

and

```python
len_est = CategoricalEstimator(keys="len_dist")
seq_est = HiddenMarkovEstimator(
    estimators=[est_tuple] * 10,
    len_estimator=len_est,
    keys=(None, None, "topics"),
)
```

These repo examples show a keyed length model shared across many conditional or
mixture-wrapped sequence models:
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb).

## Common Winning Patterns

### Shared Components With Different Weights

This is the main "win" pattern for `keys`.

Use it when groups, users, labels, or conditions appear to draw from the same
latent building blocks but with different prevalences. In practice, that often
means:

- shared topics across users
- shared component distributions across classes
- shared emissions across several HMMs

The per-group weights stay unshared because the whole point is that each group
combines the common building blocks differently.

Repo examples:

- [../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
- [../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb)
- [../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb)

### Keyed Sequence-Length Estimators

Use keyed length estimators when you believe groups share the same sequence
length law even if they differ in content or mixture weights.

This is especially useful when:

- the sample size per group is small
- the length model is mostly nuisance structure
- you want stable shared regularization on sequence size

The process-sequence notebook shows both a shared sequence-length estimator and
a shared HMM length estimator:
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb).

### Share Only The Stable Part Of A Larger Model

Often the best model is not "share everything" or "share nothing." It is
"share the reusable latent vocabulary, keep the routing parameters local."

Examples:

- HMM: share emissions, keep `w` and `trans` separate
- Mixture: share components, keep weights separate
- Gaussian-like estimators: share covariance but not means, or the reverse

The tests in
[../tests/stats/hidden_markov_test.py](../tests/stats/hidden_markov_test.py)
and [../tests/stats/dmvn_test.py](../tests/stats/dmvn_test.py) are useful
because they show these partial-sharing combinations explicitly.

## When Not To Share

Do not add a key just because two subtrees have the same Python type.

Avoid sharing when:

- the groups have meaningfully different support, scales, or regimes
- sequence length is itself discriminative and should remain group-specific
- the hidden states are not semantically aligned across groups
- the model needs group-specific transitions, priors, or calibration to fit
  correctly
- pooling would erase the signal you actually care about

The key question is not "can these subtrees share?" It is "would pooled
sufficient statistics represent the same underlying parameter?"

## Common Failure Modes

### Over-Sharing

The most common mistake is keying a larger subtree than the data support. This
can wash out real group differences and force a misleading compromise model.

Typical symptom: all groups look similar after fitting, but the fit quality or
downstream discrimination gets worse.

### Sharing Weights When You Only Wanted Shared Components

If your modeling story is "same components, different mixtures," then sharing
the weight slot is wrong. That collapses the per-group differences you were
trying to estimate.

For mixture-style models, default to unshared weights unless you have a strong
reason otherwise. The repo examples consistently favor shared components with
group-specific weights.

### False State Or Topic Alignment

Keys force parameter tying. They do not solve semantic alignment by themselves.
If two groups really need different latent states, keying their topics can make
the fit unstable or uninterpretable.

This is a real risk in HMMs and mixture-of-mixtures: sharing can be powerful,
but only when the latent components are genuinely reusable.

### Sharing Length When Length Carries Signal

A keyed `len_estimator` is useful only if a common length law is a reasonable
assumption. If sequence length is one of the main ways groups differ, sharing it
throws away that information.

### Reusing A Key Name For Different Statistical Claims

A key name should correspond to one semantic parameter-sharing claim. Reusing
the same string for unrelated subtrees can accidentally pool statistics that
should remain separate.

Use distinct names for distinct shared objects such as `"topics"`,
`"mix_comps"`, `"length_dist"`, `"trans"`, or `"covar"`.

### Expecting A Wrapper To Imply Sharing

A `ConditionalDistribution`, `SequenceEstimator`, or outer `MixtureEstimator`
does not automatically cause inner parameters to share. The inner subtree must
carry the shared key on the relevant slot.

The examples in
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
and [../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb)
are useful here because the keyed child estimator is explicit.

## Repo Examples To Start From

- [../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb):
  user-conditional mixtures with shared components and different user weights
- [../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb):
  shared mixture topics, shared sequence-length estimators, and shared HMM
  emissions/topics
- [../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb):
  "mixture of sequence of mixtures" with keyed inner components
- [../notebooks/dmx_basics_hmm.ipynb](../notebooks/dmx_basics_hmm.ipynb):
  keyed HMM topics and discussion of keyed sharing in sequence models
- [../tests/stats/heterogeneous_mixture_test.py](../tests/stats/heterogeneous_mixture_test.py):
  compact examples of sharing weights, components, or both
- [../tests/stats/hidden_markov_test.py](../tests/stats/hidden_markov_test.py):
  compact examples of sharing `w`, `trans`, and emission components separately
- [../tests/stats/dmvn_test.py](../tests/stats/dmvn_test.py):
  slot-level sharing for mean and covariance

If you are deciding where to place a key, start from the smallest subtree that
expresses your intended shared latent structure, then leave the rest unshared
until the data justify a stronger tying assumption.
