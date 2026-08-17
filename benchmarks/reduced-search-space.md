# Benchmark: Reduced Search Space

Source notebook:
[../notebooks/dmx_example_reduced_search_space.ipynb](../notebooks/dmx_example_reduced_search_space.ipynb)

## Prompt

The user points you at a local retrieval dataset:
`data/benchmarks/search/key_counter_timer_sequences.parquet`

Start by inspecting that file and confirm that one observation is:

```python
(
    key: str,
    counter: int,
    timer: float,
    key_sequence: list[str],
)
```

There are about 1000 unique keys. The downstream task is search: for each test
observation, rank likely keys and measure search depth. The `key_sequence`
field often contains repeated tokens. The top 100 keys have much more support
than the long tail. Search-time cost matters.

Recommend one primary `dmx.stats` route and explain which search-space
reductions are appropriate. Your answer should cover:

1. whether and when it is safe to rewrite `key_sequence` as value counts
2. whether the search target should be reduced to a top-N head plus one
   non-target bucket
3. why a shared mixture fit is preferable to separate per-key fits
4. how to recover key-specific search views after fitting
5. when cosine similarity over posterior embeddings is an acceptable surrogate
   for full mixture reweighting

## Expected Success Characteristics

A strong answer should:

- identify the data as a heterogeneous composite observation with a large-key
  retrieval objective
- recommend fitting one shared composite mixture while excluding `key` from the
  fitted observation and recovering per-key summaries post-fit
- explain that value-count compression is safe only when sequence order is not
  central to the task
- explain that top-N target reduction is appropriate for search when the head
  has enough support and the tail can be treated as one coarse non-target
  bucket
- note that reducing the explicit search set is safer than dropping the tail
  from training
- describe per-key posterior averaging and reweighted component scoring as the
  main search-time mechanism
- describe cosine similarity on normalized posterior vectors as a speed
  surrogate that should be checked against actual search-depth behavior

## Regression Checks

Treat these as failures:

- recommending fully separate per-key models as the default route for a
  1000-key search problem
- collapsing sequence order without stating the assumption that order is not
  important
- dropping the long-tail data from the shared fit with no discussion of the
  information loss
- treating the aggregated non-target bucket as a universally safe replacement
  for tail keys in every task
- recommending cosine similarity as equivalent to full probabilistic scoring
  without any validation step
