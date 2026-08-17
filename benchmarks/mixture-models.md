# Benchmark: Mixture Models

Source notebook:
[../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb)

## Prompt

You need to recommend package-native `dmx.stats` mixture constructions for
three local modeling problems.

Problem A:

```python
(
    real_value: float,
    maybe_missing_count: Optional[int],
    token_sequence: list[str],
    value_set: set[str],
)
```

Each observation is one heterogeneous record. The likely downstream tasks are
imputation of the missing count field and later posterior-based embeddings.

Problem B:

```python
list[float]
```

Each observation is a short sequence drawn from one of several outer mixture
types, but every outer type should reuse the same inner Gaussian topics with
different weights. You can either use a specialized estimator or spell the
model out with nested mixtures and `keys`.

Problem C:

```python
(
    x_view,
    y_view,
)
```

Both views are observed together at training time. After fitting, the main
task is to use an observation from `x_view` to cheaply construct a predictive
mixture over `y_view` without refitting a separate conditional model.

Recommend the right primary route for each problem. Explain:

1. the observation structure that drives the choice
2. why the package-native construction is preferred
3. where `keys` should go, if any
4. when a specialized estimator should win over a generic keyed construction
5. how the fitted model should be reused after training

## Expected Behavior

A strong answer should:

- choose a composite mixture for Problem A, with field-level estimators wrapped
  in `CompositeEstimator` and then mixed with `MixtureEstimator`
- explain that the composite route is preferred because it keeps one reusable
  latent model for imputation and posterior embeddings instead of introducing a
  narrow conditional model too early
- choose a hierarchical or equivalent keyed inner-mixture route for Problem B
- explain `MixtureEstimator(..., keys=(None, "comps"))` as "share inner
  components, do not share mixture weights"
- prefer the specialized hierarchical estimator for Problem B when it matches
  the intended structure, because it is faster and more direct than the fully
  generic keyed fallback
- choose `JointMixtureEstimator` for Problem C
- explain that the joint route is preferred because it learns aligned latent
  structure plus a transition matrix that supports fast posterior transfer from
  `x_view` to `y_view`
- describe post-fit reuse through posterior reweighting for Problem A and
  cross-view reweighting with `taus12` or `taus21` for Problem C

## Regression Checks

Treat these as failures:

- recommending a plain flat mixture for Problem A without modeling the
  heterogeneous fields explicitly
- putting `keys` on the mixture-weight slot when the claim is "same inner
  topics, different outer proportions"
- choosing the slow generic keyed construction as the default for Problem B
  without mentioning the matching specialized hierarchical estimator
- recommending separate independent fits for `x_view` and `y_view` in Problem
  C with no learned cross-view transfer
- missing the post-fit reuse story and instead suggesting a new refit for
  imputation or for view-to-view prediction
