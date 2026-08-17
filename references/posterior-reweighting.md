# Posterior Reweighting

`dmx-learn` models are often most useful after the fit, not just during the
fit. A shared mixture, composite mixture, or joint mixture can act as a
reusable latent basis that supports several downstream tasks:

- inspect learned components
- compute posterior responsibilities for new observations
- derive conditional or task-specific views by replacing the mixture weights
- avoid refitting separate models when the downstream task changes

This is the core "fit the model first, use it later" workflow described in
[../AGENTIC_DMX_PLAN.md](../AGENTIC_DMX_PLAN.md).

## Inspect The Fitted Components

For ordinary mixture-style models, the fitted object already exposes the parts
you need:

```python
model.components
model.w
model.num_components
```

This is the basic inspection surface for
`MixtureDistribution` and `HeterogeneousMixtureDistribution` in
[../src/dmx/stats/mixture.py](../src/dmx/stats/mixture.py) and
[../src/dmx/stats/heterogeneous_mixture.py](../src/dmx/stats/heterogeneous_mixture.py).

Typical post-fit inspection looks like:

```python
for k, comp in enumerate(model.components):
    print(k, model.w[k], comp)
```

If each mixture component is itself a `CompositeDistribution`, inspect the
sub-distributions through `comp.dists`:

```python
for k, comp in enumerate(model.components):
    user_model = comp.dists[0]
    feature_model = comp.dists[1]
    target_model = comp.dists[2]
```

That "pop off one view, keep the others" pattern is used repeatedly in
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb).

For joint mixtures, inspect the two component families and their transition
matrices directly:

```python
joint.components1
joint.components2
joint.w1
joint.w2
joint.taus12
joint.taus21
```

This is the reusable structure that lets you move posterior mass from one view
to the other after fitting, as shown in
[../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb).

## Compute Posteriors

For a single observation, use `posterior`:

```python
posterior = model.posterior(x)
```

For a batch, encode once and use `seq_posterior`:

```python
enc = model.seq_encode(data)
posteriors = model.seq_posterior(enc)
```

The returned vector is the posterior over latent components for each
observation. For mixture-style models this is exactly

```python
P(Z = k | x) proportional to P(x | Z = k) * P(Z = k)
```

If you need to inspect the unnormalized evidence before posterior
normalization, use component-wise log-densities:

```python
log_fk = model.component_log_density(x)
log_fk_batch = model.seq_component_log_density(enc)
```

Those methods are useful when you want to debug why one component dominates, or
when you need a custom reweighting rule built from per-component scores rather
than the normalized posterior alone.

The mixture-model notebook uses this pattern for missing-data imputation and
posterior embeddings:
[../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb).

## Derive Conditional Views By Reweighting

The usual pattern is:

1. keep the fitted component distributions fixed
2. compute a posterior over latent components from the observed part
3. build a new mixture over the downstream view using those posterior weights

### Composite Mixture To Conditional Target View

If each component is a composite tuple such as `(user, features, attack_flag)`,
you can condition on the observed fields and reuse the fitted target
sub-distributions:

```python
observed_components = [
    CompositeDistribution(comp.dists[:-1]) for comp in model.components
]
observed_mix = MixtureDistribution(observed_components, model.w)

target_components = [comp.dists[-1] for comp in model.components]

posterior = observed_mix.posterior(x_observed)
target_mix = MixtureDistribution(target_components, posterior)
```

At that point `target_mix` is the conditional predictive view. You can score,
sample, or summarize it without refitting the original model.

This is the exact downstream pattern used for attack prediction in
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb).

### Aggregate Posteriors To Build Entity-Specific Weights

If you fit one shared model over all observations, then later want a
user-specific, group-specific, or session-specific view, average the
observation-level posteriors within that entity:

```python
enc = model.seq_encode(user_data)
user_weights = model.seq_posterior(enc).mean(axis=0)
user_view = MixtureDistribution(model.components, user_weights)
```

This gives a cheap conditional approximation of "what this user tends to look
like under the shared latent basis." It is the notebook pattern used in both:

- [../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
- [../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb)

In the process-sequence notebook, those averaged posteriors are used as
user embeddings for similarity search and downstream visualization.

### Joint Mixture To Cross-View Reweighting

A joint mixture already learns how posterior mass in one view should transfer
to the other. The notebook pattern is:

```python
mix1 = MixtureDistribution(joint.components1, joint.w1)
posterior1 = mix1.posterior(x1)

weights2 = posterior1 @ joint.taus12
weights2 /= weights2.sum()

mix2_given_x1 = MixtureDistribution(joint.components2, weights2)
```

Then `mix2_given_x1` is the reweighted second-view mixture induced by the first
view. You can combine it with a second observation, compare it to `P(H | Y)`,
or use it as a fast cross-view proposal.

See the "joint mixture model" section of
[../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb).

## When Reweighting Is Better Than Separate Models

Prefer posterior reweighting when:

- the downstream task is not fully fixed at fit time
- many groups or labels should reuse the same latent components
- per-group sample sizes are too small for stable separate fits
- you want fast post-fit conditional views, ranking, or search
- the goal is low-rank conditional approximation rather than a fully separate
  `p(x | y)` model

This is especially attractive for keyed shared-component mixtures where the
modeling claim is "same latent parts, different mixture weights." The notebooks
on conditional-vs-composite mixtures and process sequences both use this idea:
share topics or components, then recover group-specific behavior by posterior
aggregation instead of fitting a fresh model per group.

## When Separate Models Are Better

Posterior reweighting is not a free replacement for task-specific models.

Fit separate models when:

- the groups do not plausibly share latent components
- support, scale, or calibration differs sharply by task
- the downstream target changes the component shapes, not just their weights
- the shared model is clearly underfitting the task-specific conditional

The decision rule is simple: if the downstream task mainly changes how often
shared components are used, reweighting is usually the right first move. If the
task changes what the components themselves should be, fit a different model.
