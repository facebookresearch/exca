# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path

project = "Exca"
copyright = "Meta Platforms, Inc"
author = "FAIR"
release = "0.1"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    # "sphinx.ext.autosectionlabel",
    # "sphinx.ext.githubpages",
    # "sphinx.ext.coverage",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.autosummary",
    # "recommonmark",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "pythonplusplus", ".pytest_cache"]

# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []  # ["_static"]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    base = Path().absolute().parent
    module = info["module"].replace(".", "/")
    if (base / module).with_suffix(".py").exists():
        filepath = module
    else:
        filepath = (
            module
            + "/"
            + info["fullname"].split(".", maxsplit=1)[0].replace("Infra", "").lower()
        )
        if not (base / filepath).with_suffix(".py").exists():
            return None
    return "https://github.com/fairinternal/exca/blob/main/%s.py" % filepath
