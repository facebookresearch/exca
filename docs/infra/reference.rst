API Reference
=============

.. autoclass:: exca.TaskInfra
    :members:
    :inherited-members:
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config, model_construct, model_copy, model_dump, model_dump_json, model_extra, model_fields_set, model_json_schema, model_parametrized_name, model_rebuild, model_validate, model_validate_json, model_validate_strings, copy, apply_on


.. autoclass:: exca.MapInfra
    :members:
    :inherited-members: 
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config, model_construct, model_copy, model_dump, model_dump_json, model_extra, model_fields_set, model_json_schema, model_parametrized_name, model_rebuild, model_validate, model_validate_json, model_validate_strings, copy, apply_on

.. autoclass:: exca.SubmitInfra
    :members:
    :inherited-members:
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config, model_construct, model_copy, model_dump, model_dump_json, model_extra, model_fields_set, model_json_schema, model_parametrized_name, model_rebuild, model_validate, model_validate_json, model_validate_strings, copy, apply_on

Associated classes and functions
--------------------------------

.. autoclass:: exca.slurm.SubmititMixin
    :members:
    :inherited-members: 
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config, model_construct, model_copy, model_dump, model_dump_json, model_extra, model_fields_set, model_json_schema, model_parametrized_name, model_rebuild, model_validate, model_validate_json, model_validate_strings, copy, apply_on

.. autoclass:: exca.workdir.WorkDir
    :members:
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config

.. autoclass:: exca.ConfDict
    :members:
    :exclude-members: model_post_init, model_fields, model_computed_fields, model_config

.. autoclass:: exca.cachedict.CacheDict
    :members:

.. automodule:: exca.helpers
    :members: with_infra, find_slurm_job
