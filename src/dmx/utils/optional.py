"""Import optional dependencies with actionable install hints.

dmx-learn keeps its core installation small. Features that need extra packages
are grouped into pip extras declared in ``pyproject.toml``:

- ``torch``: PyTorch-backed distributions (:mod:`dmx.torch_stats`)
- ``spark``: PySpark RDD estimation and sampling
- ``viz``: matplotlib plotting helpers
- ``agent``: the MCP server for the expert agent (includes ``viz``)

Code that needs one of these packages should import it through
:func:`require_optional` (or a thin wrapper such as :func:`require_matplotlib`)
so a missing installation fails with an :class:`ImportError` that names the
extra to install, instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Optional


def install_hint(extra: str) -> str:
    """Return the pip command that installs a dmx-learn extra.

    Args:
        extra: Name of the extra, for example ``"viz"``.

    Returns:
        A command such as ``pip install dmx-learn[viz]``.
    """
    return f"pip install dmx-learn[{extra}]"


def missing_dependency_message(package: str, extra: str, feature: str) -> str:
    """Build the error message for a missing optional dependency.

    Args:
        package: Display name of the missing package.
        extra: Name of the dmx-learn extra that installs it.
        feature: Name of the module or feature that needs the package.

    Returns:
        A message naming the feature, the package, and how to install it.
    """
    return (
        f"{feature} requires {package}, which is not installed. "
        f"Install it with: {install_hint(extra)} "
        f"(or `poetry install -E {extra}` from a source checkout)."
    )


def require_optional(
    module: str,
    extra: str,
    feature: str = "This feature",
    package: Optional[str] = None,
) -> ModuleType:
    """Import an optional module or raise an error naming the extra to install.

    Only a missing ``module`` (or one of its parent packages) is reported as a
    missing extra. If the module is installed but one of its own dependencies
    fails to import, the original error propagates unchanged.

    Args:
        module: Dotted module name to import, for example
            ``"matplotlib.pyplot"``.
        extra: Name of the dmx-learn extra that provides the module.
        feature: Name of the module or feature that needs it, used in the
            error message.
        package: Display name of the package in the error message. Defaults to
            the top-level package of ``module``.

    Returns:
        The imported module.

    Raises:
        ImportError: If ``module`` is not installed.
    """
    top_level = module.split(".", 1)[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        missing = exc.name
        if missing is not None and missing.split(".", 1)[0] != top_level:
            raise
        raise ImportError(
            missing_dependency_message(package or top_level, extra, feature)
        ) from exc


def is_available(module: str) -> bool:
    """Return whether an optional module can be imported, without importing it.

    Args:
        module: Dotted module name, for example ``"mcp"``.

    Returns:
        ``True`` if the module is installed.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require_matplotlib(feature: str = "Plotting") -> ModuleType:
    """Import ``matplotlib.pyplot`` from the ``viz`` extra.

    Args:
        feature: Name of the feature that needs matplotlib.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    return require_optional("matplotlib.pyplot", "viz", feature, package="matplotlib")


def require_mcp(feature: str = "The dmx agent") -> ModuleType:
    """Import the ``mcp`` SDK from the ``agent`` extra.

    Args:
        feature: Name of the feature that needs the MCP SDK.

    Returns:
        The ``mcp`` module.

    Raises:
        ImportError: If the MCP SDK is not installed.
    """
    return require_optional("mcp", "agent", feature, package="the MCP Python SDK")
