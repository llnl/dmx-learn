#!/bin/bash
# Helper script to run tests with Poetry

# Ensure we're not in a venv
unset VIRTUAL_ENV

# Navigate to project root
cd "$(dirname "$0")"

# Check arguments
case "$1" in
    "stats")
        echo "Running stats tests..."
        poetry run pytest tests/stats "$@"
        ;;
    "torch")
        echo "Running torch_stats tests..."
        export NUMBA_DISABLE_JIT=1
        poetry run pytest tests/torch_stats "${@:2}"
        ;;
    "torch-mps")
        echo "Running torch_stats tests on MPS device..."
        export NUMBA_DISABLE_JIT=1
        TEST_TORCH_DEVICE=mps poetry run pytest tests/torch_stats "${@:2}"
        ;;
    "all")
        echo "Running all tests..."
        poetry run pytest tests/stats -q
        export NUMBA_DISABLE_JIT=1
        poetry run pytest tests/torch_stats -q
        ;;
    "quick")
        echo "Running quick sanity check..."
        poetry run pytest tests/stats/binomial_test.py -q
        export NUMBA_DISABLE_JIT=1
        poetry run pytest tests/torch_stats/binomial_test.py -q
        ;;
    *)
        echo "Usage: $0 {stats|torch|torch-mps|all|quick}"
        echo ""
        echo "Examples:"
        echo "  $0 stats              # Run all stats tests"
        echo "  $0 torch              # Run all torch tests (CPU)"
        echo "  $0 torch-mps          # Run all torch tests (MPS)"
        echo "  $0 all                # Run all tests"
        echo "  $0 quick              # Quick sanity check"
        echo "  $0 stats -v           # Run stats tests verbosely"
        echo "  $0 torch -k mixture   # Run torch tests matching 'mixture'"
        exit 1
        ;;
esac
