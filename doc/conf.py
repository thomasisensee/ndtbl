# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
doc_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(doc_dir, ".."))
python_package_src = os.path.join(repo_root, "python", "ndtbl", "src")
sys.path.insert(0, python_package_src)


# -- Project information -----------------------------------------------------

project = "ndtbl"
copyright = "2026, Thomas Isensee"
author = "Thomas Isensee"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Breathe Configuration: Breathe is the bridge between the information extracted
# from the C++ sources by Doxygen and Sphinx.
breathe_projects = {
    "ndtbl": os.path.join(repo_root, "build", "doc", "xml"),
}
breathe_default_project = "ndtbl"
