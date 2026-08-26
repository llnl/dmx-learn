# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# The project uses a ``src`` layout.  Include it explicitly so autodoc can
# import ``dmx`` even when documentation dependencies are installed outside
# Poetry's project environment (as on Read the Docs).
sys.path.insert(0, os.path.abspath("../src"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dmx-learn"
copyright = "2025, Adam Walder"
author = "Adam Walder"
version = "1.1.1"
release = "1.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # auto-generate documentation
    "sphinx.ext.napoleon",  # support for Google and NumPy docstrings
    "sphinx_autodoc_typehints",  # auto-document type hints
    "sphinx.ext.mathjax",
]

# Preserve declared defaults without evaluating their recursive model reprs.
autodoc_preserve_defaults = True

# Several backends intentionally use the same short protocol and model names.
# Generated annotations are qualified; legacy prose keeps its backend-local
# spelling, whose otherwise valid targets are necessarily ambiguous globally.
autodoc_typehints = "description"
typehints_fully_qualified = True
napoleon_use_param = False
napoleon_use_rtype = False
suppress_warnings = ["ref.python"]

# Optional GPU and distributed backends are not part of the RTD environment.
# Mock them so autodoc can render the corresponding API references without
# downloading PyTorch or requiring an MPI runtime.
autodoc_mock_imports = ["torch", "mpi4py", "umap"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
