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

.. _fitting-a-cohort:

Fitting a cohort
----------------

A :class:`Fit` step fits one artifact over a set of items, then
transforms each item with it. Call :meth:`Step.fit_many` with the
items to fit on; call :meth:`Step.run_many` to transform with an
artifact already fitted:

.. code-block:: python

    from exca import steps

    class Normalize(steps.Fit):
        def _fit(self, values):  # the cohort, streamed
            values = list(values)
            return sum(values) / len(values)

        def _run(self, value):  # one item
            return value - self.fitted

    cache = "/tmp/exca-fit-cache"
    train, test = [1.0, 2.0, 3.0], [10.0]

    norm = Normalize(infra={"backend": "Cached", "folder": cache})
    list(norm.fit_many(train))  # fits on train, then transforms it
    list(norm.run_many(test))   # transforms with the same artifact

The artifact is named by the fingerprint of the items it was fitted
on, so the cohort scopes its cache and every downstream cache
without anything to declare. Declare a ``cohort`` name on the step to
recover its artifact in a process that never presents those items —
a fresh interpreter loading the same config, which cannot compute the
fingerprint:

.. code-block:: python

    infra = {"backend": "Cached", "folder": cache}
    named = Normalize(infra=infra, cohort="train2024")
    list(named.fit_many(train))                              # fitting process
    Normalize(infra=infra, cohort="train2024").run(test[0])  # any later process

The name is part of the config, which is what makes the recovery
above need nothing else. ``fit_many(..., cohort="train2024")`` asks
for the same name for that call alone: it scopes the fit as it runs
but is not recorded, so re-presenting those items to a config dumped
from that run fits again under an automatic name, beside the named
artifact. Use it to point an unnamed instance at an existing cohort
when the caller supplies the name explicitly. That takes a *fresh*
instance: the one that fitted is already bound to its own cohort, and
with no items the fresh one recovers the named cohort and fits nothing:

.. code-block:: python

    recovered = Normalize(infra=infra)
    list(recovered.fit_many(cohort="train2024"))  # recovers, fits nothing
    list(recovered.run_many(test))

A name stays bound to the items it was fitted on: handing it a
different cohort raises rather than silently refitting.

If ``fit_many`` raises before returning, every ``Fit`` it touched is
restored to the declaration it held, so a field can be corrected and the
call retried. Cache writes already made remain — in a chain whose second
fit raises, the first fit's artifact stays on disk, and the retry reads it
back instead of refitting. After ``fit_many`` returns, a failure while
consuming the results leaves the fit declared and the completed outputs
cached, as in ``run_many``.

``fit_many`` lives on every :class:`Step`, not just on ``Fit``, because a
``Chain``, a ``Scatter`` or any resolving step can contain one. On a
pipeline with no ``Fit`` it is exactly ``run_many`` — same outputs, same
identities, same cache entries — so two pipelines differing only in a
fitting step stay comparable at one call site, and a ``cohort`` name
given to one is checked for shape and then ignored.

Three costs to plan for:

- **The upstream is read twice**, once for the fit and once per item,
  so give an expensive upstream its own ``infra``.
- **A cohort name needs a cache folder** to recover from: set ``infra``
  or ``fit_infra`` on the fit itself. An enclosing container may supply
  its folder, but does not create the backend.
- **A fit cannot live inside a sweep or a sharding container.** A
  ``Parallel`` variant holding one is refused before the sweep
  dispatches anything, as is a fit under a backend that would hand it
  a shard of the cohort. Fit each variant on its own, or put the
  ``Fit`` before the sweep —
  ``Chain(steps=[Fit(...), Parallel(...)])``.

.. autoclass:: exca.steps.Fit
    :show-inheritance:
    :members: fitted, lookup, ARTIFACT_CACHE_TYPE

Batched execution
-----------------

.. autoclass:: exca.steps.items.StepItems

.. autoexception:: exca.steps.items.BatchProtocolError

.. autoclass:: exca.steps.backends.RunContext()
    :members: folder

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

- ``workdir: exca.workdir.WorkDir | None`` — copied code snapshot for
  ``LocalProcess``, ``Slurm`` and ``Auto``; see :doc:`howto`.

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
