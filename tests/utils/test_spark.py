"""Tests for utils/spark.py and the optional PySpark dependency."""

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pytest

import dmx
from dmx import bstats, stats
from dmx.utils.fisher import fisher_embedding
from dmx.utils.spark import (
    SPARK_INSTALL_HINT,
    fisher_embedding_rdd,
    is_rdd,
    require_pyspark,
)

# Setting sys.modules["pyspark"] to None makes any later import of pyspark fail,
# so these checks behave the same whether or not the spark extra is installed.
BLOCK_PYSPARK = "import sys; sys.modules['pyspark'] = None; "


def _run_without_pyspark(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter where pyspark cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_PYSPARK + code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_imports_without_pyspark() -> None:
    """Import the core packages without loading pyspark."""
    result = _run_without_pyspark(
        "import dmx, dmx.stats, dmx.bstats, dmx.utils.estimation; "
        "assert sys.modules['pyspark'] is None"
    )

    assert result.returncode == 0, result.stderr


def test_rdd_sampler_without_pyspark_names_spark_extra() -> None:
    """Fail to import the Spark sampler with an install hint."""
    result = _run_without_pyspark("import dmx.stats.rdd_sampler")

    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert SPARK_INSTALL_HINT in result.stderr


def test_require_pyspark_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the feature and the spark extra when pyspark is missing."""
    monkeypatch.setitem(sys.modules, "pyspark", None)

    with pytest.raises(ImportError, match=r"my_feature requires PySpark") as exc:
        require_pyspark("my_feature")

    assert SPARK_INSTALL_HINT in str(exc.value)


def test_is_rdd_false_without_pyspark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat every object as local data when pyspark is not loaded."""
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)

    assert not is_rdd([1, 2, 3])
    assert "pyspark" not in sys.modules


def test_is_rdd_detects_rdd_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognize instances of pyspark.RDD once pyspark is loaded."""
    fake_pyspark = types.ModuleType("pyspark")

    class RDD:  # pylint: disable=too-few-public-methods
        """Stand-in for pyspark.RDD."""

    setattr(fake_pyspark, "RDD", RDD)
    monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)

    assert is_rdd(RDD())
    assert not is_rdd([1, 2, 3])


# Fisher embeddings of RDDs, checked against the local fisher_embedding rows.


@pytest.fixture(name="spark_context", scope="module")
def fixture_spark_context() -> Iterator[Any]:
    """Start a local SparkContext whose workers import this dmx checkout."""
    pyspark = pytest.importorskip("pyspark")
    saved = {name: os.environ.get(name) for name in ("PYSPARK_PYTHON", "PYTHONPATH")}
    source_root = str(Path(dmx.__file__).resolve().parents[1])
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [source_root] + ([saved["PYTHONPATH"]] if saved["PYTHONPATH"] else [])
    )
    try:
        context = pyspark.SparkContext("local[2]", "dmx-fisher-embedding-test")
    except Exception as err:  # pylint: disable=broad-exception-caught
        pytest.skip(f"Cannot start a local SparkContext: {err}")
    context.setLogLevel("ERROR")
    try:
        yield context
    finally:
        context.stop()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fisher_model() -> stats.MixtureDistribution:
    """Heterogeneous two-component mixture for the embedding tests."""
    return stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(-2.0, 1.0),
                    stats.CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
                )
            ),
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(2.0, 2.0),
                    stats.CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
                )
            ),
        ],
        [0.4, 0.6],
    )


def _bstats_fisher_model() -> bstats.MixtureDistribution:
    """Bayesian Gaussian mixture for the embedding tests."""
    return bstats.MixtureDistribution(
        (
            bstats.GaussianDistribution(-3.0, 1.0),
            bstats.GaussianDistribution(0.0, 0.5),
            bstats.GaussianDistribution(3.0, 2.0),
        ),
        [0.3, 0.3, 0.4],
    )


def _collect_rows(rows: Any, width: int) -> np.ndarray:
    """Return the rows of an ``(index, row)`` RDD as a dense array in order."""
    pairs = sorted(rows.collect(), key=lambda pair: pair[0])
    assert [index for index, _ in pairs] == list(range(len(pairs)))
    assert all(row.shape == (1, width) for _, row in pairs)
    return np.vstack([row.toarray() for _, row in pairs])


@pytest.mark.parametrize("make", [_fisher_model, _bstats_fisher_model])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_fisher_embedding_rdd_matches_local_rows(
    spark_context: Any, make: Any, dtype: Any, chunk_size: Optional[int]
) -> None:
    model = make()
    data = list(model.sampler(5).sample(41))
    expected = fisher_embedding(model, data, dtype=dtype)

    rows, layout = fisher_embedding_rdd(
        model, spark_context.parallelize(data, 3), dtype=dtype, chunk_size=chunk_size
    )

    assert layout.scale == expected.scale
    assert layout.matrix.shape == (0, expected.matrix.shape[1])
    assert layout.component_slices == expected.component_slices
    assert layout.field_slices == expected.field_slices
    actual = _collect_rows(rows, expected.matrix.shape[1])
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected.matrix.toarray())


def test_fisher_embedding_rdd_whitening_and_like(spark_context: Any) -> None:
    model = _fisher_model()
    data = list(model.sampler(6).sample(40))
    expected = fisher_embedding(model, data, whiten="full", dtype=np.float64)

    rows, layout = fisher_embedding_rdd(
        model, spark_context.parallelize(data, 4), whiten="full", dtype=np.float64
    )

    np.testing.assert_allclose(
        _collect_rows(rows, expected.matrix.shape[1]),
        expected.matrix.toarray(),
        rtol=1e-9,
        atol=1e-12,
    )
    new_data = data[:7]
    new_rows, _ = fisher_embedding_rdd(
        model, spark_context.parallelize(new_data, 2), like=layout
    )
    np.testing.assert_array_equal(
        _collect_rows(new_rows, expected.matrix.shape[1]),
        fisher_embedding(model, new_data, like=layout).matrix.toarray(),
    )
