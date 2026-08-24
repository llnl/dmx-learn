# Conditional Vs Composite Mixture

This notebook distillation captures the main lesson from
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb):
when many groups share the same latent structure, fit one reusable composite
mixture first and recover group-specific behavior from posteriors instead of
starting with a separate conditional model per group.

## Observation Structure

One observation is a heterogeneous record:

```python
(
    user_id,
    packet_size,
    session_duration,
    failed_logins,
    protocol_type?,
    encryption?,
    browser_type?,
    attack_detected?,
)
```

Important structural facts:

- `user_id` is a high-cardinality grouping field with some users having only a
  few observations.
- `packet_size`, `session_duration`, and `failed_logins` are numeric leaves.
- `protocol_type`, `encryption`, and `browser_type` are optional categoricals.
- `attack_detected` is optional and treated as "missing because unlabeled,"
  not ordinary missing-at-random noise.

That is composite structure first, then a possible mixture over the composite.

## Compared Model Families

The notebook compares three closely related routes.

### 1. Conditional Mixture Per User

Use `ConditionalDistributionEstimator` with one per-user mixture, but share the
mixture components across users with `keys`:

```python
est_mix = MixtureEstimator([est_tuple] * 10, keys=(None, "comps"))
est = ConditionalDistributionEstimator(
    estimator_map={user_id: est_mix for user_id in user_ids}
)
```

Interpretation: same latent components, different user-specific mixture
weights.

### 2. Composite Mixture With `user_id` Inside The Record

Model the entire row as one composite tuple and put a mixture above it:

```python
est_tuple = CompositeEstimator(
    [est_user, est_packet, est_duration, est_login, est_proto, est_enc,
     est_browser, est_attack]
)
est = MixtureEstimator([est_tuple] * 10, keys=(None, "comps"))
```

Interpretation: one latent mixture explains users and features jointly.

### 3. Unsupervised Composite Mixture Plus Grouped Posterior Averaging

Drop `user_id` from the fitted model, fit the mixture to the non-user fields,
then estimate user-specific weights by averaging `P(Z | x)` within each user.

Interpretation: the fitted mixture is a reusable latent basis; group-specific
views are derived after the fit.

## Preferred Route

Prefer the composite-mixture route, usually the unsupervised version when later
reuse matters.

Why it wins in this notebook:

- it preserves the same "shared components, different weights" idea as the
  conditional model
- it fits much faster because there is one main mixture instead of a
  conditional wrapper over many per-user mixtures
- its user-search performance is essentially the same as the conditional route
- it is more reusable because the fitted model is not tied to one downstream
  conditional query

The notebook explicitly shows the conditional and composite routes giving
nearly identical search-depth curves, with the composite fit being the more
practical choice.

## Why `keys` Matter

The critical modeling claim is not "fit separate user models." It is:

- share the latent components across users
- keep the mixture weights user-specific

That is exactly what `keys=(None, "comps")` expresses for the mixture:
unshared weights, shared component subtree.

## Downstream Use After Fit

The fitted composite mixture is then reused in two ways.

### User Retrieval

Recover user-specific weights from the shared mixture and score new
non-`user_id` observations against each user view.

Two equivalent notebook patterns are:

- include `user_id` in the composite mixture and derive `P(Z | user_id)`
- exclude `user_id` from fitting and average observation-level posteriors by
  user

### Attack Prediction

Pop off the `attack_detected` field, compute a posterior over mixture
components from the other fields, then reweight the attack sub-distributions:

```python
observed_mix = MixtureDistribution(
    [CompositeDistribution(comp.dists[:-1]) for comp in model.components],
    model.w,
)
attack_comps = [comp.dists[-1] for comp in model.components]

posterior = observed_mix.posterior(x_without_attack)
attack_mix = MixtureDistribution(attack_comps, posterior)
```

This is the main downstream lesson: fit one shared latent model, then derive
conditional views by posterior reweighting instead of refitting a new model for
each task.

## Routing Rule

When the data look like "many entities, few observations per entity, shared
latent behavior, and more than one likely downstream task":

1. Treat one row as a composite observation.
2. Fit one composite mixture as the primary model.
3. Use the conditional-per-user mixture only as a baseline or when a truly
   explicit conditional API is required.
4. Recover entity-specific behavior from posterior-derived weights after fit.
