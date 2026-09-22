API Reference
=============

.. note::

   The Step API is part of Exca core but its public surface is
   still evolving — expect breaking changes between minor
   versions. See :doc:`index` for context.

Core classes
------------

.. autoclass:: exca.steps.Step
    :members: run, run_many, fit_many, lookup, clone, item_uid, CACHE_TYPE

.. autoclass:: exca.steps.Chain
    :show-inheritance:

``Chain`` has the same public API as :class:`Step`; the constructor
adds a ``steps`` field (a ``Sequence[Step]`` or ``OrderedDict[str,
Step]``). ``chain[i]`` and ``len(chain)`` index into the sub-steps;
slicing returns a new ``Chain``.

Fitting a cohort
----------------

A :class:`Fit` step fits one artifact over a set of items, then
transforms each item with it. Call :meth:`Step.fit_many` with the
items to fit on; call :meth:`Step.run_many` to transform with an
artifact already fitted:

.. code-block:: python

    norm = Normalize(infra={"backend": "Cached", "folder": cache})
    norm.fit_many(train)  # fits on train, then transforms it
    norm.run_many(test)   # transforms with the same artifact

The artifact is named by the fingerprint of the items it was fitted
on, so the cohort scopes its cache and every downstream cache
without anything to declare. Declare a ``cohort`` name on the step to
recover its artifact in a process that never presents those items —
a fresh interpreter loading the same config, which cannot compute the
fingerprint:

.. code-block:: python

    named = Normalize(infra=..., cohort="train2024")
    named.fit_many(train)                               # fitting process
    Normalize(infra=..., cohort="train2024").run(item)  # any later process

The name is part of the config, which is what makes the recovery
above need nothing else. ``fit_many(..., cohort="train2024")`` asks
for the same name for that call alone: it scopes the fit as it runs
but is not recorded, so re-presenting those items to a config dumped
from that run fits again under an automatic name, beside the named
artifact. Use it to point an unnamed config at an existing cohort
within one process; with no items it recovers that cohort and fits
nothing:

.. code-block:: python

    norm.fit_many(cohort="train2024")  # recovers, fits nothing
    norm.run_many(test)

A name stays bound to the items it was fitted on: handing it a
different cohort raises rather than silently refitting.

A failed call declares nothing: every ``Fit`` it touched is left as
the call found it, so a field can be corrected and the call retried.
The cache is not rolled back with it — in a chain whose second fold
raises, the first fold's artifact stays on disk, and the retry reads
it back instead of refitting.

``fit_many`` lives on every :class:`Step`, not just on ``Fit``, because a
``Chain``, a ``Scatter`` or any resolving step can contain one. On a
pipeline with no ``Fit`` it is exactly ``run_many`` — same outputs, same
identities, same cache entries — so two pipelines differing only in a
fitting step stay comparable at one call site, and a ``cohort`` name
given to one is checked for shape and then ignored.

A ``Parallel`` sweep is the exception: its variants are dispatched
through the shared backend rather than through their own pipeline, so
one holding a ``Fit`` would run without its cohort. Such a sweep is
refused before it dispatches anything, however it is reached: when it
is built if the variant declares the ``Fit``, when it runs if the
variant only resolves to one. Fit each variant on
its own, or put the ``Fit`` before the sweep —
``Chain(steps=[Fit(...), Parallel(...)])``.

.. autoclass:: exca.steps.Fit
    :show-inheritance:
    :members: fitted, lookup, ARTIFACT_CACHE_TYPE

Batched execution
-----------------

.. autoclass:: exca.steps.items.StepItems

.. autoexception:: exca.steps.items.BatchProtocolError

Cache lookup
------------

.. autoclass:: exca.steps.backends.LookupHandle
    :members:

``LookupHandle.paths`` exposes a ``StepPaths`` dataclass with two
relevant attributes: ``step_folder`` (logs, metadata) and
``cache_folder`` (cache entries).

Backends
--------

All backends share the fields declared on the base
:class:`Backend`:

.. autoclass:: exca.steps.backends.Backend
    :members: folder, mode, keep_in_ram

Submitit-based backends — ``LocalProcess``, ``SubmititDebug``,
``Slurm``, ``Auto`` — additionally accept:

- ``job_name: str | None``
- ``timeout_min: int | None``
- ``nodes, tasks_per_node, cpus_per_task, gpus_per_node``
  (``int | None``), ``mem_gb: float | None``
- ``max_jobs: int = 128`` — maximum number of array sub-jobs.
- ``min_items_per_job: int = 1`` — combine items into one job
  below this count.

Pool backends — ``ProcessPool``, ``ThreadPool`` — additionally
accept:

- ``max_jobs: int | None = 128`` — maximum pool size.

.. autoclass:: exca.steps.backends.Cached
    :show-inheritance:

.. autoclass:: exca.steps.backends.LocalProcess
    :show-inheritance:

.. autoclass:: exca.steps.backends.SubmititDebug
    :show-inheritance:

.. autoclass:: exca.steps.backends.Slurm
    :show-inheritance:
    :members:

.. autoclass:: exca.steps.backends.Auto
    :show-inheritance:

.. autoclass:: exca.steps.backends.ProcessPool
    :show-inheritance:

.. autoclass:: exca.steps.backends.ThreadPool
    :show-inheritance:

Helpers
-------

.. autoclass:: exca.steps.helpers.Func
    :members:
