"""Run pytests on all py files in examples folder."""

from . import run_example

EXAMPLES_PATH = "examples"


def test_auto_example():
    run_example("auto_example.py", examples_path=EXAMPLES_PATH)


def test_detailed_estimation_example():
    run_example("detailed_estimation_example.py", examples_path=EXAMPLES_PATH)


def test_htsne_example():
    run_example("htsne_example.py", examples_path=EXAMPLES_PATH)
