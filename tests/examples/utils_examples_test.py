"""Run pytests on utility example scripts."""

from . import run_example

EXAMPLES_PATH = "examples/utils_examples"


def test_auto_example() -> None:
    run_example("auto_example.py", examples_path=EXAMPLES_PATH)


def test_detailed_estimation_example() -> None:
    run_example("detailed_estimation_example.py", examples_path=EXAMPLES_PATH)


def test_htsne_example() -> None:
    run_example("htsne_example.py", examples_path=EXAMPLES_PATH)
