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

## How new blockMeshDict is generated

For each sampling row, the generator creates a fresh `system/blockMeshDict` from code (not by manual editing):

1. Load one row from sampling table (`naca`, `aoa`, `inlet_velocity`)
2. Build NACA upper/lower curves with LE clustering
3. Rotate profile by case AoA (around quarter chord)
4. Build C-grid topology points/blocks/edges from config values
5. Write final dictionary into the case folder

Where this happens:
- [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) (loop over all rows)
- [src/case_builder.py](../src/case_builder.py) (`build_single_blockmesh_case`)
- [src/blockmesh_generator.py](../src/blockmesh_generator.py) (`build_cgrid_blockmesh_dict` + `write_blockmesh_dict`)

## What changes per case vs what stays fixed

Per-case changing inputs (from `sampling`):
- `naca_code` from `(camber, camber_position, thickness)`
- `aoa_deg`
- `inlet_velocity` (used in `0/U` and `forceCoeffs`, not in mesh topology)

Note:
- `system/forceCoeffs` is updated per case not only for CFD postprocessing,
  but also as a direct source for the scalar Cl/Cd ML branch (`clcd_mlp`).

Current active sampling ranges in [configs/dataset_config.yaml](../configs/dataset_config.yaml):
- `camber_values`: `0, 2, 4`
- `camber_position_values`: `2, 4` (for `camber=0` the code is forced to symmetric `00xx`)
- `thickness_values`: `8, 12, 16`
- `aoa_values`: `-4.0, 0.0, 4.0 deg`
- `inlet_velocity_values`: `15.0, 20.0, 25.0 m/s`

Mesh controls (`blockmesh`) are currently global/fixed for all generated cases in one build run.

## Mesh parameters: what they mean (current values)

Domain size:
- `x_min=-5.0`, `x_max=12.0`, `x_far=20.0`: inlet, near outlet, and far-wake outlet plane
- `y_min=-5.0`, `y_max=5.0`: vertical far-field limits
- `z_half=0.05`: half-thickness of 2D extrusion (`frontAndBack` is `empty`)

Resolution:
- `n_airfoil_half=520`: points used per airfoil side curve during spline construction
- `n_streamwise_near=240`: near-airfoil streamwise block resolution
- `n_wall_normal=170`: near-wall normal resolution
- `n_wake_x=320`: wake resolution in near wake (`x_te -> x_max`)
- `n_far_wake_x=80`: additional far wake resolution (`x_max -> x_far`)
- `n_z=1`: one cell in spanwise direction (2D setup)

Grading:
- `grading_to_wall=2400.0`: strong clustering toward airfoil wall
- `grading_le_tangent=0.65`: tangential clustering near leading-edge side blocks
- `grading_wake_x=1.35`: expansion in wake direction in near wake
- `grading_far_wake_x=4.0`: stronger expansion in far wake extension

Topology/geometry controls:
- `le_cluster_exp=2.8`: stronger point clustering toward LE in generated NACA curve
- `enable_le_cap=false`: geometric LE smoothing cap currently disabled
- `le_cap_fraction=0.035`, `le_cap_power=1.2`: LE cap shape parameters (used only if enabled)
- `le_topology_fraction=0.028`: size of topological LE cap region
- `n_le_cap_normal=42`: cell count through LE cap block
- `te_transition_fraction=0.995`: start of TE sub-curve split along chord
- `wake_cut_length=0.0001`: short TE wake-cut point for C-grid closure

## Practical note for manual-style runs

If you were preparing cases manually, the usual sequence would be:
- choose geometry (`NACA`, `AoA`) from sampling row
- keep one validated mesh parameter set fixed for all cases
- regenerate `blockMeshDict`
- run `blockMesh` + `checkMesh`
- only then run solver chain

This project follows exactly that pattern, just scripted and config-driven.

## Downstream extension

The wake domain is extended using far-wake blocks (`x_max -> x_far`) to keep wake development inside the domain while preserving near-airfoil cell sizing.

## Build artifacts

Each built case contains:
- generated `system/blockMeshDict`
- updated `0/U`
- updated `system/forceCoeffs`
- `params.json` metadata

This makes each case immediately compatible with both downstream branches:

- spatial branch: extraction to NPZ (`p`, `U`, mask)
- scalar branch: robust parsing of `forceCoeffs.dat` for Cl/Cd targets

## Sampling table dependency

Sampling rows (NACA + AoA + inlet velocity) come from `airfoil_sampling_table.csv`.
If the file is missing, it is generated automatically from `dataset_config.yaml` by [src/generate_sampling_table.py](../src/generate_sampling_table.py).
