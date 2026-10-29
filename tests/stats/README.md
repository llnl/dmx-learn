# `tests/stats`

This directory holds the regression and conformance tests for the `dmx.stats`
distribution layer.

## Structure

- `stats_tests.py` defines the shared test harness used by many distribution test
  modules. It centralizes the standard checks for string round-tripping,
  estimator wiring, encoder wiring, sampler repeatability, log-density
  consistency, accumulator factories, and sequence updates.
- `*_test.py` files provide distribution-specific fixtures and any extra checks
  that are unique to a distribution family, such as key validation, posterior
  behavior, or component log-density handling.
- `fisher_score_test.py` is the registry for the whitened Fisher score
  contract (see "Fisher score contract" below).

## What These Tests Cover

- Distribution `str(...)` output can be evaluated back into an equivalent
  distribution.
- `estimator()`, `dist_to_encoder()`, and accumulator factories return the
  expected companion objects.
- Samplers are repeatable for a fixed seed.
- Scalar and sequence log-density paths agree.
- Sequence update paths improve or preserve the encoded data likelihood.
- Distribution-specific type and key validation raises the expected errors.

## Fisher score contract

Every distribution that implements `seq_fisher_score` (see
`docs/utils/fisher_embedding.rst`) is checked by one shared contract in
`tests/fisher_contract.py`, used by `tests/stats/fisher_score_test.py` and
`tests/bstats/fisher_score_test.py`:

- scores have shape `(n, fisher_dim(options))`, dtype float64, and finite
  values;
- two calls on the same input return identical scores;
- `fisher_field_slices` covers each score column exactly once;
- under the distribution, the score mean is zero and the covariance is the
  identity, or a projector when the case declares a `rank_deficit` (one per
  one-hot block). Families whose scores are whitened empirically skip this
  check (`check_moments=False`), and offset-free sparse forms are centered
  first (`centered=False`).

Each case draws `n = 1000` samples with a seed derived from its id. Moment
tolerances are 5 standard errors estimated from the same sample, which
catches mean offsets of about 0.2 and covariance errors of about 0.3; finer
errors are covered by each family's hand-computed value tests. The whole
contract suite runs in well under a second once `dmx.stats` is imported.

To add a family, implement its score and add or update its `FisherCase`
entries; do not change the shared checks. A case whose family has no score
yet is skipped with the GitHub issue that implements it, so
`pytest -rs tests/stats/fisher_score_test.py` lists the remaining work.
`tests/utils/test_fisher_contract.py` checks the contract itself against
correct and deliberately broken toy scores.

## Notes

- Many modules intentionally follow the same setup pattern so new distribution
  tests can match the existing shape quickly.
- A few tests include custom helpers when a distribution needs weighted data,
  component-wise densities, posteriors, or sequence-specific behaviors that do
  not fit the base harness exactly.
