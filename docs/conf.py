# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exca.steps  # noqa: E402 — after sys.path bootstrap above

# Resolve forward refs so autodoc reads real fields, not pydantic mocks.
for cls in (
    exca.steps.Chain,
    exca.steps.helpers.Func,
    exca.steps.backends.Cached,
    exca.steps.backends.LocalProcess,
    exca.steps.backends.SubmititDebug,
    exca.steps.backends.Slurm,
    exca.steps.backends.Auto,
    exca.steps.backends.ProcessPool,
    exca.steps.backends.ThreadPool,
):
    cls.model_rebuild()

project = "Exca"
copyright = "Meta Platforms, Inc"
author = "FAIR"
release = exca.__version__

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

# Hide pydantic BaseModel boilerplate from every autoclass.
autodoc_default_options = {
    "exclude-members": ", ".join(
        [
            "model_post_init",
            "model_fields",
            "model_computed_fields",
            "model_config",
            "model_construct",
            "model_copy",
            "model_dump",
            "model_dump_json",
            "model_extra",
            "model_fields_set",
            "model_json_schema",
            "model_parametrized_name",
            "model_rebuild",
            "model_validate",
            "model_validate_json",
            "model_validate_strings",
        ]
    ),
}

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
    "article_header_start": ["breadcrumbs"],
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
