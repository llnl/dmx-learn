# Benchmark: Variable Sequence Length Search Depth

Source notebook:
[../notebooks/dmx_variable_sequence_length_search_depth.ipynb](../notebooks/dmx_variable_sequence_length_search_depth.ipynb)

## Prompt

You have paired local data where one observation is:

```python
(
    document_tokens: list[str],
    score: float,
)
```

The `document_tokens` field is variable-length and is modeled with both token
content and an explicit length distribution. The `score` field is a single
continuous side feature. The downstream task is paired retrieval: for each
query, rank the true partner near the front of the candidate list and evaluate
search depth.

Recommend the right `dmx` routing logic for this problem. Your answer should
explain:

1. why variable sequence length makes the retrieval direction asymmetric
2. when `P(document_tokens | score)` is a risky search route
3. when `P(score | document_tokens)` is the safer route
4. which shortcuts are safe when reusing one shared mixture model
5. why search depth, not just likelihood correctness, should drive the choice

## Expected Behavior

A strong answer should:

- identify the data as a paired composite observation with one variable-length
  sequence view and one low-dimensional scalar view
- explain that sequence length changes the information budget because each
  token and the length model sharpen `P(Z | document_tokens)`
- note that retrieval is not symmetric: the better conditioning direction
  depends on which view yields the sharper mixture posterior
- treat `P(document_tokens | score)` as risky when the score distributions
  overlap and the sequence candidates are much more informative
- prefer `P(score | document_tokens)` when the sequence side carries the main
  topic evidence and the scalar side is mostly a weak auxiliary feature
- recommend fitting one shared joint mixture and reusing component likelihoods
  and posteriors rather than refitting many conditional models
- say that the route should be validated with held-out search-depth behavior,
  not chosen from likelihood formulas alone

## Regression Checks

Treat these as failures:

- claiming that `P(document_tokens | score)` and `P(score | document_tokens)`
  should have equivalent search behavior because they come from the same joint
  model
- ignoring the effect of variable sequence length on posterior sharpness
- recommending scalar-to-sequence retrieval by default without checking whether
  the scalar feature actually separates mixture components
- proposing per-query or per-candidate refits instead of reusing the shared
  mixture structure
- judging the routing choice only by probabilistic validity and not by search
  depth or ranking quality
