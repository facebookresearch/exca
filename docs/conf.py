# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Make sure exca is importable for autodoc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "Exca"
copyright = "Meta Platforms, Inc"
author = "FAIR"
release = "0.5.15"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "substitution",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "pythonplusplus",
    ".pytest_cache",
    "internal",
]

# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/exca.png"

html_theme_options = {
    "logo": {
        "image_light": "_static/exca.png",
        "image_dark": "_static/exca.png",
        "text": "Exca",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/facebookresearch/exca",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/exca/",
            "icon": "fa-brands fa-python",
        },
    ],
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "announcement": None,
}

html_context = {
    "github_user": "facebookresearch",
    "github_repo": "exca",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "index": [],
    "**": ["sidebar-nav-bs"],
}

html_show_sourcelink = False

# -- Linkcode ----------------------------------------------------------------


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
    return "https://github.com/facebookresearch/exca/blob/main/%s.py" % filepath
