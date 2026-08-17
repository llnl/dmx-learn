# Benchmark: Conditional Vs Composite Mixture

Source notebook:
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)

## Prompt

You have local training data where one observation is:

```python
(
    user_id: str,
    packet_size: int,
    session_duration: float,
    failed_logins: int,
    protocol_type: Optional[str],
    encryption: Optional[str],
    browser_type: Optional[str],
    attack_detected: Optional[int],
)
```

There are hundreds of users, some with only a few observations. The numeric
fields are always present. The categorical fields can be missing. The
`attack_detected` label is only present when a record was hand-labeled, so its
missingness should not be treated as ordinary missing-at-random noise.

I need one fitted `dmx-learn` model that can support two downstream tasks:

1. rank likely users for a new observation
2. later estimate `P(attack_detected | observed fields)` without refitting a
   separate model

Recommend one primary `dmx.stats` model and one meaningful baseline. Explain
the observation structure, where `keys` should go, why your preferred route is
better, how you would initialize the fit, and how you would reuse the fitted
model for both downstream tasks.

## Expected Behavior

A strong answer should:

- identify the data as one composite observation with optional categorical
  fields, not as an independent target-only classification problem
- choose a composite mixture as the primary model, preferably emphasizing the
  reusable unsupervised or joint route over a per-user conditional fit
- use a conditional shared-component mixture as the baseline:
  `ConditionalDistributionEstimator` with shared mixture components and
  user-specific weights
- explain `keys` as "share components, do not share mixture weights"
- mention a stabilized initializer with flattened or smoothed categorical
  support, near-uniform mixture weights, and multiple `best_of` restarts
- explain that the composite route is preferred because it is faster,
  practically equivalent on the user-search task, and reusable for later
  conditioning
- describe post-fit user retrieval through `P(Z | user_id)` or averaged
  per-user posteriors
- describe post-fit attack prediction by dropping the attack field, computing
  `P(Z | observed fields)`, and reweighting the attack sub-distributions

## Regression Checks

Treat these as failures:

- recommending fully separate per-user models with no shared latent structure
- putting `keys` on mixture weights when the claim is "same components,
  different user proportions"
- choosing a narrow supervised classifier as the primary model even though the
  prompt requires later reuse for a second downstream task
- missing the post-fit conditional reuse step and instead suggesting a second
  refit for attack prediction
