# Benchmark: Process Sequences

Source notebook:
[../notebooks/dmx_example_process_sequences.ipynb](../notebooks/dmx_example_process_sequences.ipynb)

## Prompt

The user points you at a local grouped-sequence file:
`data/benchmarks/process/user_process_sequences.jsonl`

Start by inspecting that file and confirm that one observation is:

```python
(
    user_id: str,
    process_sequence: list[tuple[float, str]],
)
```

Each `process_sequence` is ordered, variable-length, and usually has length
between 5 and 8. Each element is `(process_time, process_name)`. There are
many users, each with a moderate number of sequences. The downstream task is
to rank likely user aliases for new target sequences from that local dataset.

I want one primary `dmx.stats` modeling route and one meaningful baseline.
Explain:

1. what the observation structure is
2. when topics or emissions should be shared across users with `keys`
3. when sequence length should be keyed
4. when to prefer a sequence-of-mixtures route over an HMM-style route
5. how to use the fitted model after training to rank likely users

## Expected Success Characteristics

A strong answer should:

- identify one observation as a grouped variable-length sequence of
  heterogeneous tuples, not as an unordered bag or flat tabular row
- use a per-user conditional sequence model as the simpler baseline, while
  noting that fully separate user models are brittle without smoothing
- prefer shared topics across users when the same latent process motifs should
  be reused with different user-specific weights
- explain `MixtureEstimator(..., keys=(None, "mix_comps"))` as "share
  components, do not share weights"
- explain `HiddenMarkovEstimator(..., keys=(None, None, "topics"))` as
  "share emissions, do not share initial weights or transitions"
- say that sequence length should be keyed only when length is mostly common
  nuisance structure rather than a discriminative group feature
- recommend the sequence-of-mixtures route when content dominates and explicit
  transition dependence is not the main modeling need
- recommend the HMM route when order and transition behavior matter enough to
  justify a more structured sequence model
- describe direct per-user sequence scoring for conditional fits and
  posterior-embedding aggregation for the shared HMM route

## Regression Checks

Treat these as failures:

- recommending fully separate user models as the primary route with no shared
  latent structure
- sharing mixture weights when the intended claim is "same topics, different
  users emphasize them differently"
- keying sequence length by default without checking whether length carries the
  group signal
- recommending an HMM solely because the data are sequences, without discussing
  whether transition dependence matters
- missing the post-fit user-embedding workflow for the shared HMM route
