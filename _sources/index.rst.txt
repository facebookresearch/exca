:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div class="hero-banner">
     <h1>Exca</h1>
     <p>Execution and caching for Python — validate configs, cache results, and scale to Slurm clusters with zero boilerplate.</p>
     <a class="sd-btn sd-btn-primary" href="infra/tutorials.html">Get Started</a>&nbsp;&nbsp;
     <a class="sd-btn sd-btn-outline-light" href="https://github.com/facebookresearch/exca">GitHub</a>
   </div>

.. raw:: html

   <div class="install-snippet">
     <code>pip install exca</code>
   </div>

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Introduction
      :link: infra/introduction
      :link-type: doc
      :class-card: sd-border-0

      Learn *why* Exca exists — config validation, hierarchical
      experiments, and transparent caching.

   .. grid-item-card:: Tutorials
      :link: infra/tutorials
      :link-type: doc
      :class-card: sd-border-0

      Step-by-step guides covering ``TaskInfra``, ``MapInfra``,
      caching, remote computation, and more.

   .. grid-item-card:: How-to guides
      :link: infra/howto
      :link-type: doc
      :class-card: sd-border-0

      Recipes for common tasks — debugging, job arrays,
      uid exclusion, versioning, workdir, and discriminated unions.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Explanation
      :link: infra/explanation
      :link-type: doc
      :class-card: sd-border-0

      Deep dives into the design philosophy, uid computation,
      ``ConfDict``, ``CacheDict``, and more.

   .. grid-item-card:: Serialization
      :link: infra/serialization
      :link-type: doc
      :class-card: sd-border-0

      How ``CacheDict`` stores data, built-in handlers, and how
      to write your own custom handler.

   .. grid-item-card:: API Reference
      :link: infra/reference
      :link-type: doc
      :class-card: sd-border-0

      Complete API for ``TaskInfra``, ``MapInfra``, ``SubmitInfra``,
      ``CacheDict``, helpers, and associated classes.

.. raw:: html

   <div class="section-header">
     <h2>Why Exca?</h2>
     <p>A few features that make experiment management simpler.</p>
   </div>

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-border-0

      .. raw:: html

         <span class="feature-badge feature-badge-config">Config</span>

      **Validated configs** — Pydantic-powered hierarchical configurations
      catch errors *before* you hit the cluster.

   .. grid-item-card::
      :class-card: sd-border-0

      .. raw:: html

         <span class="feature-badge feature-badge-cache">Cache</span>

      **Transparent caching** — Filesystem-backed, uid-based caching
      that Just Works™. Rerun only what changed.

   .. grid-item-card::
      :class-card: sd-border-0

      .. raw:: html

         <span class="feature-badge feature-badge-cluster">Cluster</span>

      **Cluster-ready** — Submit to Slurm or run locally — switch between
      the two by changing one field.

   .. grid-item-card::
      :class-card: sd-border-0

      .. raw:: html

         <span class="feature-badge feature-badge-config">Modular</span>

      **Modular pipelines** — Discriminated unions, composable sub-configs,
      and job arrays for grid-search with ease.


.. raw:: html

   <div class="section-header">
     <h2>Comparison</h2>
   </div>

.. raw:: html

   <div class="comparison-table">

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 12 12 12

   * - Feature
     - lru_cache
     - hydra
     - submitit
     - stool
     - **exca**
   * - RAM cache
     - ✔
     - ✘
     - ✘
     - ✘
     - ✔
   * - File cache
     - ✘
     - ✘
     - ✘
     - ✘
     - ✔
   * - Remote compute
     - ✘
     - ✔
     - ✔
     - ✔
     - ✔
   * - Pure Python
     - ✔
     - ✘
     - ✔
     - ✘
     - ✔
   * - Hierarchical config
     - ✘
     - ✔
     - ✘
     - ✘
     - ✔

.. raw:: html

   </div>

----

Citing
------

.. code-block:: bibtex

    @misc{exca,
        author = {J. Rapin and J.-R. King},
        title = {{Exca - Execution and caching}},
        year = {2024},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/facebookresearch/exca}},
    }

Legal
-----

:code:`exca` is MIT licensed, as found in the LICENSE file.
Also check-out Meta Open Source `Terms of Use <https://opensource.fb.com/legal/terms>`_ and `Privacy Policy <https://opensource.fb.com/legal/privacy>`_.


.. toctree::
   :maxdepth: 2
   :hidden:

   infra/getting-started
   infra/user-guide
   infra/reference
