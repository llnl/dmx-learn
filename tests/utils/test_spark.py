"""Tests for utils/spark.py and the optional PySpark dependency."""

import subprocess
import sys
import types

import pytest

from dmx.utils.spark import SPARK_INSTALL_HINT, is_rdd, require_pyspark

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
