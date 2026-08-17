# Agentic DMX Benchmark Prompt Suite

This directory holds a small repo-local benchmark suite for incremental
validation of the agentic DMX skill system.

The suite is intentionally capped at five prompts. Each benchmark is derived
from a real notebook pattern in this repo and includes:

- a notebook-like prompt with a concrete local data path and downstream goal
- expected success characteristics for a strong answer
- regression checks for common failure modes

## Canonical Prompt Set

1. [conditional-vs-composite-mixture.md](conditional-vs-composite-mixture.md)
   Source notebook: `notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb`
   Covers: `keys`, shared-component vs composite-mixture routing, model
   comparison, posterior reweighting
2. [mixture-models.md](mixture-models.md)
   Source notebook: `notebooks/dmx_basics_mixture_models.ipynb`
   Covers: composite mixtures, keyed sharing, specialized-vs-generic routes,
   posterior reuse
3. [process-sequences.md](process-sequences.md)
   Source notebook: `notebooks/dmx_example_process_sequences.ipynb`
   Covers: grouped sequences, keyed topic sharing, HMM vs sequence-of-mixtures
4. [geotweet.md](geotweet.md)
   Source notebook: `notebooks/dmx_advanced_geotweet.ipynb`
   Covers: joint mixtures, cross-view transfer, map-oriented downstream queries
5. [reduced-search-space.md](reduced-search-space.md)
   Source notebook: `notebooks/dmx_example_reduced_search_space.ipynb`
   Covers: retrieval-oriented shared fits, search-depth evaluation, practical
   search-space reduction

Use these prompts as the canonical small benchmark set for testing.
