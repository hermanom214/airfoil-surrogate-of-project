# Mesh Workflow (blockMesh-only)

## Overview

The project currently uses a **blockMesh-only C-grid topology** around NACA airfoils.
No `snappyHexMesh` or STL surface meshing is used in the active branch.

## Generator and builder

- topology generator: [src/blockmesh_generator.py](../src/blockmesh_generator.py)
- case assembly: [src/case_builder.py](../src/case_builder.py)
- build entrypoint: [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py)

## Config-driven mesh parameters

All key mesh parameters are in [configs/dataset_config.yaml](../configs/dataset_config.yaml) under `blockmesh`:
- domain extents (`x_min`, `x_max`, `x_far`, `y_min`, `y_max`, `z_half`)
- resolution (`n_airfoil_half`, `n_streamwise_near`, `n_wall_normal`, `n_wake_x`, `n_far_wake_x`)
- grading (`grading_to_wall`, `grading_le_tangent`, `grading_wake_x`, `grading_far_wake_x`)
- geometry/topology controls (`le_*`, `te_transition_fraction`, `wake_cut_length`)

## Downstream extension

The wake domain is extended using far-wake blocks (`x_max -> x_far`) to keep wake development inside the domain while preserving near-airfoil cell sizing.

## Build artifacts

Each built case contains:
- generated `system/blockMeshDict`
- updated `0/U`
- updated `system/forceCoeffs`
- `params.json` metadata

## Sampling table dependency

Sampling rows (NACA + AoA + inlet velocity) come from `airfoil_sampling_table.csv`.
If the file is missing, it is generated automatically from `dataset_config.yaml` by [src/generate_sampling_table.py](../src/generate_sampling_table.py).
