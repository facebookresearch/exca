# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test script to reproduce the recursive discriminated union issue."""

from typing import Annotated, Literal

from exca import TaskInfra
from pydantic import BaseModel, Field, ConfigDict


class Leaf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_type: Literal["leaf"] = "leaf"


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(cluster=None, folder="/tmp/")
    edge_type: Literal["edge"] = "edge"
    child: "Elem"

    @infra.apply
    def run(self) -> None:
        return


Elem = Annotated[
    Leaf | Edge,
    Field(discriminator="edge_type"),
]

if __name__ == "__main__":
    print("Creating Edge with Leaf child...")
    cfg = Edge(child=Leaf())
    
    print("Testing model_dump_json (should work)...")
    try:
        result = cfg.model_dump_json(exclude_defaults=True)
        print(f"✓ model_dump_json succeeded: {result}")
    except Exception as e:
        print(f"✗ model_dump_json failed: {e}")
        raise
    
    print("\nTesting infra.config (should work after fix)...")
    try:
        result = cfg.infra.config(exclude_defaults=True)
        print(f"✓ infra.config succeeded: {result}")
    except KeyError as e:
        print(f"✗ KeyError still occurs: {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

