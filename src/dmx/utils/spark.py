"""Optional PySpark support for the Spark-backed code paths.

PySpark is not a core dependency of dmx-learn; it is installed with the
``spark`` (or ``distributed``/``all``) extra. Core modules use :func:`is_rdd`
to accept RDDs without importing PySpark, and Spark-only modules call
:func:`require_pyspark` at import time so a missing installation fails with an
actionable message instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark import RDD
    from typing_extensions import TypeIs

SPARK_INSTALL_HINT = "pip install dmx-learn[spark]"


def require_pyspark(feature: str = "This feature") -> ModuleType:
    """Import PySpark or raise an error explaining how to install it.

    Args:
        feature: Name of the module or feature that needs PySpark, used in the
            error message.

    Returns:
        The imported ``pyspark`` module.

    Raises:
        ImportError: If PySpark is not installed.
    """
    try:
        return importlib.import_module("pyspark")
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires PySpark, which is not installed. "
            f"Install it with: {SPARK_INSTALL_HINT} "
            "(or `poetry install -E spark` from a source checkout)."
        ) from exc


def is_rdd(obj: Any) -> TypeIs[RDD[Any]]:
    """Return whether ``obj`` is a PySpark RDD, without importing PySpark.

    An RDD can only exist once PySpark has been imported, so when PySpark is
    absent from ``sys.modules`` the answer is ``False`` and PySpark is never
    loaded.

    Args:
        obj: Object to test.

    Returns:
        ``True`` if ``obj`` is an instance of ``pyspark.RDD``.
    """
    pyspark = sys.modules.get("pyspark")
    rdd_type = getattr(pyspark, "RDD", None)
    return isinstance(rdd_type, type) and isinstance(obj, rdd_type)
