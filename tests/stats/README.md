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

## What These Tests Cover

- Distribution `str(...)` output can be evaluated back into an equivalent
  distribution.
- `estimator()`, `dist_to_encoder()`, and accumulator factories return the
  expected companion objects.
- Samplers are repeatable for a fixed seed.
- Scalar and sequence log-density paths agree.
- Sequence update paths improve or preserve the encoded data likelihood.
- Distribution-specific type and key validation raises the expected errors.

## Notes

- Many modules intentionally follow the same setup pattern so new distribution
  tests can match the existing shape quickly.
- A few tests include custom helpers when a distribution needs weighted data,
  component-wise densities, posteriors, or sequence-specific behaviors that do
  not fit the base harness exactly.
