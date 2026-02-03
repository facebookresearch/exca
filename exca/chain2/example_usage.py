# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of the chain2 API.

Key patterns:
- Override `_forward()` for step logic
- Use `forward()` to execute (handles caching/backend)
- Use `with_input()` to attach input for cache key
- Use dict syntax for infra: {"backend": "Cached", "folder": "..."}
- All infra inherits from Cached (so all have caching)
"""

from __future__ import annotations

# Import only Step and Chain - no need to import backend classes
# from exca.chain2 import Step, Chain, StepInfra

# =============================================================================
# EXAMPLE 1: Basic Step Definition
# =============================================================================
#
# Steps override `_forward()` for their logic.
# Use `forward()` to execute.


# class Multiply(Step):
#     """A simple step that multiplies input by a coefficient."""
#
#     coeff: float = 2.0
#
#     def _forward(self, value: float) -> float:
#         return value * self.coeff
#
#
# class Add(Step):
#     """A simple step that adds a value to input."""
#
#     amount: float = 0.0
#
#     def _forward(self, value: float) -> float:
#         return value + self.amount
#
#
# def example_basic_step():
#     step = Multiply(coeff=3.0)
#     result = step.forward(5.0)  # Returns 15.0
#     print(f"Basic step: 5.0 * 3.0 = {result}")


# =============================================================================
# EXAMPLE 2: Step with Caching (dict syntax)
# =============================================================================
#
# Use dict syntax for infra - no imports needed, pydantic handles it.


# class ExpensiveComputation(Step):
#     seed: int = 42
#     size: int = 1000
#
#     def _forward(self, scale: float) -> np.ndarray:
#         rng = np.random.RandomState(self.seed)
#         return rng.randn(self.size) * scale
#
#
# def example_cached_step(tmp_path: Path):
#     # Dict syntax - no need to import Cached
#     step = ExpensiveComputation(
#         seed=123,
#         size=10000,
#         infra={"backend": "Cached", "folder": str(tmp_path / "cache")},
#     )
#
#     result1 = step.forward(2.5)  # Computes and caches
#     result2 = step.forward(2.5)  # From cache
#
#     assert np.allclose(result1, result2)
#     print(f"Cached step: array of shape {result1.shape}")


# =============================================================================
# EXAMPLE 3: Execution Modes
# =============================================================================
#
# All infra classes have mode: cached/force/read-only/retry


# def example_modes(tmp_path: Path):
#     # Default: cached
#     step = Multiply(
#         coeff=2.0,
#         infra={"backend": "Cached", "folder": str(tmp_path / "cache"), "mode": "cached"},
#     )
#     result = step.forward(5.0)  # Computes and caches
#     result = step.forward(5.0)  # Uses cache
#
#     # Force recomputation
#     step = Multiply(
#         coeff=2.0,
#         infra={"backend": "Cached", "folder": str(tmp_path / "cache"), "mode": "force"},
#     )
#     result = step.forward(5.0)  # Always recomputes
#
#     # Read-only
#     step = Multiply(
#         coeff=3.0,
#         infra={"backend": "Cached", "folder": str(tmp_path / "cache"), "mode": "read-only"},
#     )
#     try:
#         result = step.forward(5.0)  # Raises if not cached
#     except RuntimeError as e:
#         print(f"Expected error: {e}")


# =============================================================================
# EXAMPLE 4: Slurm Backend (dict syntax)
# =============================================================================
#
# Slurm inherits from Cached, so it has caching built-in.
# Use dict syntax - all slurm options in the same dict.


# class GPUComputation(Step):
#     hidden_size: int = 512
#
#     def _forward(self, data: np.ndarray) -> np.ndarray:
#         return data @ np.random.randn(data.shape[1], self.hidden_size)
#
#
# def example_slurm_step():
#     # For production: use Slurm
#     # step = GPUComputation(
#     #     hidden_size=256,
#     #     infra={"backend": "Slurm", "folder": "/tmp/gpu", "gpus_per_node": 1},
#     # )
#
#     # For local testing: use Cached
#     step = GPUComputation(
#         hidden_size=256,
#         infra={"backend": "Cached", "folder": "/tmp/test"},
#     )
#     data = np.random.randn(100, 64)
#     result = step.forward(data)
#     print(f"GPU step: {data.shape} -> {result.shape}")


# =============================================================================
# EXAMPLE 5: Chain of Steps
# =============================================================================


# def example_chain(tmp_path: Path):
#     chain = Chain(
#         steps=[
#             Multiply(coeff=2.0),
#             Add(value=10.0),
#             Multiply(coeff=3.0),
#         ],
#         infra={"backend": "Cached", "folder": str(tmp_path / "chain")},
#     )
#
#     # (5 * 2 + 10) * 3 = 60
#     result = chain.forward(5.0)
#     print(f"Chain: (5 * 2 + 10) * 3 = {result}")


# =============================================================================
# EXAMPLE 6: Chain with Folder Propagation
# =============================================================================
#
# propagate_folder=True creates Cached infra for steps without infra


# def example_folder_propagation(tmp_path: Path):
#     chain = Chain(
#         steps=[
#             Multiply(coeff=2.0),  # No infra -> gets Cached(folder=chain_folder)
#             Add(value=10.0),      # No infra -> gets Cached(folder=chain_folder)
#             Multiply(
#                 coeff=3.0,
#                 infra={"backend": "Slurm", "folder": str(tmp_path / "slurm"), "gpus_per_node": 1},
#             ),  # Keeps its own infra
#         ],
#         infra={"backend": "Cached", "folder": str(tmp_path / "chain")},
#         propagate_folder=True,
#     )
#
#     result = chain.forward(5.0)


# =============================================================================
# EXAMPLE 7: Mixed Backends
# =============================================================================


# class DataLoader(Step):
#     path: str
#
#     def _forward(self) -> np.ndarray:  # Generator - no input
#         return np.random.randn(1000, 64)
#
#
# class Preprocessor(Step):
#     normalize: bool = True
#
#     def _forward(self, data: np.ndarray) -> np.ndarray:
#         if self.normalize:
#             return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
#         return data
#
#
# class Trainer(Step):
#     epochs: int = 10
#
#     def _forward(self, data: np.ndarray) -> dict:
#         return {"weights": np.random.randn(64, 10), "epochs": self.epochs}
#
#
# def example_mixed_backends(tmp_path: Path):
#     pipeline = Chain(
#         steps=[
#             DataLoader(path="/data/train.npy"),
#             Preprocessor(
#                 normalize=True,
#                 infra={"backend": "LocalProcess", "folder": "/cache/preprocess", "cpus_per_task": 4},
#             ),
#             Trainer(
#                 epochs=50,
#                 infra={"backend": "Slurm", "folder": "/models", "gpus_per_node": 4},
#             ),
#         ],
#         infra={"backend": "Cached", "folder": str(tmp_path / "experiment")},
#     )
#
#     result = pipeline.forward()  # No input - DataLoader generates data
#     print(f"Pipeline result: {result}")


# =============================================================================
# EXAMPLE 8: YAML Configuration
# =============================================================================
#
# Infra uses discriminator_key="backend"


EXAMPLE_YAML = """
type: Chain
infra:
  backend: Cached
  folder: /tmp/experiment
  mode: cached
propagate_folder: true
steps:
  - type: Multiply
    coeff: 2.0

  - type: Add
    value: 10.0

  - type: Multiply
    coeff: 3.0
    infra:
      backend: LocalProcess
      folder: /tmp/local
      cpus_per_task: 2
"""


SLURM_YAML = """
type: Train
epochs: 100
infra:
  backend: Slurm
  folder: /models/experiment
  mode: cached
  gpus_per_node: 8
  mem_gb: 128
  timeout_min: 120
  slurm_partition: gpu
  slurm_account: my_account
"""


# def example_from_yaml():
#     from exca import ConfDict
#
#     config = ConfDict.from_yaml_string(EXAMPLE_YAML)
#     pipeline = Step.model_validate(config.to_dict())
#
#     result = pipeline.with_input(5.0).forward()
#     print(f"From YAML: {result}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("Chain2 Example Usage")
    print("=" * 60)
    print()
    print("Key patterns:")
    print("- Override _forward() for step logic")
    print("- Use forward() to execute")
    print('- Use dict syntax: infra={"backend": "Cached", "folder": "..."}')
    print("- All infra inherits from Cached (built-in caching)")
    print()

    # Uncomment once fully tested:
    # with tempfile.TemporaryDirectory() as tmp:
    #     tmp_path = Path(tmp)
    #     example_basic_step()
    #     example_cached_step(tmp_path)
    #     example_modes(tmp_path)
    #     example_slurm_step()
    #     example_chain(tmp_path)
    #     example_folder_propagation(tmp_path)
    #     example_mixed_backends(tmp_path)
    #     example_from_yaml()

    print("Examples would run here once chain2 is fully tested.")


if __name__ == "__main__":
    main()
