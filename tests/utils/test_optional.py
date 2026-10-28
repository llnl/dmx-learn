"""Tests for utils/optional.py and the optional dependency extras."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

from dmx.utils.optional import (
    install_hint,
    is_available,
    missing_dependency_message,
    require_matplotlib,
    require_mcp,
    require_optional,
)


def _run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_install_hint_names_extra() -> None:
    """Format the pip command for an extra."""
    assert install_hint("viz") == "pip install dmx-learn[viz]"


def test_missing_dependency_message_names_feature_package_and_extra() -> None:
    """Include the feature, package, and both install commands."""
    message = missing_dependency_message("matplotlib", "viz", "Plotting")

    assert message.startswith("Plotting requires matplotlib")
    assert "pip install dmx-learn[viz]" in message
    assert "poetry install -E viz" in message


def test_require_optional_returns_installed_module() -> None:
    """Return the module when it is importable."""
    module = require_optional("json", "viz")

    assert module.__name__ == "json"


def test_require_optional_missing_module_names_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise an install hint when the module is missing."""
    monkeypatch.setitem(sys.modules, "dmx_fake_optional", None)

    with pytest.raises(ImportError, match=r"my_feature requires dmx_fake_optional"):
        require_optional("dmx_fake_optional", "viz", "my_feature")


def test_require_optional_missing_parent_names_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the top-level package when a submodule's parent is missing."""
    monkeypatch.setitem(sys.modules, "dmx_fake_optional", None)

    with pytest.raises(ImportError, match=r"requires dmx_fake_optional,") as exc:
        require_optional("dmx_fake_optional.sub", "viz")

    assert install_hint("viz") in str(exc.value)


def test_require_optional_keeps_errors_from_broken_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate a missing transitive dependency instead of blaming the extra."""
    (tmp_path / "dmx_fake_broken.py").write_text("import dmx_fake_missing_dep\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc:
        require_optional("dmx_fake_broken", "viz")

    assert exc.value.name == "dmx_fake_missing_dep"
    assert install_hint("viz") not in str(exc.value)


def test_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report importability without importing the module."""
    monkeypatch.setitem(sys.modules, "dmx_fake_optional", None)

    assert is_available("json")
    assert not is_available("dmx_fake_optional")
    assert not is_available("dmx_fake_not_installed")


@pytest.mark.parametrize(
    ("blocked", "require", "extra"),
    [("matplotlib", require_matplotlib, "viz"), ("mcp", require_mcp, "agent")],
)
def test_extra_wrappers_name_their_extra(
    monkeypatch: pytest.MonkeyPatch, blocked: str, require: object, extra: str
) -> None:
    """Point matplotlib at the viz extra and mcp at the agent extra."""
    monkeypatch.setitem(sys.modules, blocked, None)

    assert callable(require)
    with pytest.raises(ImportError) as exc:
        require()

    assert install_hint(extra) in str(exc.value)


def test_core_imports_do_not_load_viz_or_agent_dependencies() -> None:
    """Import the core packages without loading matplotlib or mcp."""
    result = _run(
        "import sys; "
        "import dmx, dmx.stats, dmx.bstats, dmx.utils.estimation, "
        "dmx.utils.automatic, dmx.utils.optional; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib'; "
        "assert 'mcp' not in sys.modules, 'mcp'"
    )

    assert result.returncode == 0, result.stderr


def test_core_requirements_exclude_viz_and_agent_dependencies() -> None:
    """Declare matplotlib and mcp only behind extras."""
    requirements = importlib.metadata.requires("dmx-learn") or []
    optional = [
        req for req in requirements if req.split()[0].startswith(("matplotlib", "mcp"))
    ]

    assert optional
    assert all("extra ==" in req for req in optional)
    for extra in ("viz", "agent", "all"):
        assert any(f'extra == "{extra}"' in req for req in optional)


def test_torch_utils_error_names_torch_extra() -> None:
    """Point a missing PyTorch at the torch extra."""
    result = _run(
        "import sys; sys.modules['torch'] = None; "
        "from dmx.torch_utils import detect_device; detect_device()"
    )

    assert result.returncode != 0
    assert install_hint("torch") in result.stderr
