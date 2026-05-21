"""Helpers for runnable example script tests."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_example(
    script_name: str,
    examples_path: str = "examples",
    disable_numba_jit: bool = True,
):
    """Run an example script and assert that it exits successfully."""
    env = os.environ.copy()

    if disable_numba_jit:
        env["NUMBA_DISABLE_JIT"] = "1"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / examples_path / script_name)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert (
        result.returncode == 0
    ), f"Script {script_name} failed with error:\n{result.stderr}"

    return result
