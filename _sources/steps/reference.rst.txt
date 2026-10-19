API Reference
=============

.. note::

   The Step API is part of Exca core but its public surface is
   still evolving — expect breaking changes between minor
   versions. See :doc:`index` for context.

Core classes
------------

.. autoclass:: exca.steps.Step
    :members: run, run_many, lookup, clone, item_uid, CACHE_TYPE

.. autoclass:: exca.steps.Chain
    :show-inheritance:

``Chain`` has the same public API as :class:`Step`; the constructor
adds a ``steps`` field (a ``Sequence[Step]`` or ``OrderedDict[str,
Step]``). ``chain[i]`` and ``len(chain)`` index into the sub-steps;
slicing returns a new ``Chain``.

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
