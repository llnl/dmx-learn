"""Run pytests on all py files in examples folder."""

# pylint: disable=duplicate-code

import os
import subprocess

examples_path = "examples"


def test_auto_example():
    result = subprocess.run(
        ["python", os.path.join(examples_path, "auto_example.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_detailed_estimation_example():
    result = subprocess.run(
        ["python", os.path.join(examples_path, "detailed_estimation_example.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_htsne_example():
    result = subprocess.run(
        ["python", os.path.join(examples_path, "htsne_example.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
