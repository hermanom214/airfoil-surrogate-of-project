# Project Overview

## Goal

Build a reproducible pipeline for:
- generating parametric airfoil CFD cases,
- running OpenFOAM simulations on a blockMesh-only topology,
- extracting structured flow fields for ML,
- training surrogate models for pressure and velocity fields.

## Current Workflow (active)

1. Generate/refresh sampling table (CSV)
2. Build OpenFOAM cases with generated `blockMeshDict`
3. Run CFD chain (`blockMesh`, `checkMesh`, `decomposePar`, `simpleFoam -parallel`, `reconstructPar`)
4. Convert latest results to VTK and interpolate to regular grid
5. Save dataset as `.npz` + dataset index CSV
6. Train ML model from config-driven hyperparameters

## What is no longer used

- STL-driven meshing branch (`snappyHexMesh`, STL split workflow)
- legacy geometry export pipeline (`.dat`/`.stl`) for case building

## Configuration-first approach

The pipeline is now controlled by YAML configs:
- [configs/paths.yaml](../configs/paths.yaml) — filesystem/runtime paths
- [configs/dataset_config.yaml](../configs/dataset_config.yaml) — sampling + blockMesh build parameters
- [configs/solver_config.yaml](../configs/solver_config.yaml) — OpenFOAM run commands and process settings
- [configs/ml_models_config.yaml](../configs/ml_models_config.yaml) — model/training/physics/output settings

## Main entry scripts

- [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py)
- [scripts/run_cases.py](../scripts/run_cases.py)
- [scripts/extract_flow_fields.py](../scripts/extract_flow_fields.py)
- [scripts/train_ml_model.py](../scripts/train_ml_model.py)

## Script/module dependencies (simple map)

| File | Where it is used |
|---|---|
| [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) | main build entrypoint |
| [scripts/run_cases.py](../scripts/run_cases.py) | main CFD run entrypoint |
| [scripts/extract_flow_fields.py](../scripts/extract_flow_fields.py) | main flow-field extraction entrypoint |
| [scripts/train_ml_model.py](../scripts/train_ml_model.py) | main ML training entrypoint |
| [src/config.py](../src/config.py) | used by all main scripts + [src/case_runner.py](../src/case_runner.py) + [src/flow_extractor.py](../src/flow_extractor.py) |
| [src/generate_sampling_table.py](../src/generate_sampling_table.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) |
| [src/sampling.py](../src/sampling.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py); `BuildCaseRow` is used by [src/case_builder.py](../src/case_builder.py) |
| [src/case_builder.py](../src/case_builder.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) |
| [src/blockmesh_generator.py](../src/blockmesh_generator.py) | used by [src/case_builder.py](../src/case_builder.py) |
| [src/file_editors.py](../src/file_editors.py) | used by [src/case_builder.py](../src/case_builder.py) |
| [src/case_runner.py](../src/case_runner.py) | used by [scripts/run_cases.py](../scripts/run_cases.py) and [src/flow_extractor.py](../src/flow_extractor.py) |
| [src/flow_extractor.py](../src/flow_extractor.py) | used by [scripts/extract_flow_fields.py](../scripts/extract_flow_fields.py) |
| [src/ml_dataset.py](../src/ml_dataset.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_models.py](../src/ml_models.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_training.py](../src/ml_training.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_validation.py](../src/ml_validation.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
