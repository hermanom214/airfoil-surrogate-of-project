# Project Overview

## Goal

Build a reproducible pipeline for:
- generating parametric airfoil CFD cases,
- running OpenFOAM simulations on a blockMesh-only topology,
- extracting structured flow fields for ML,
- training surrogate models for pressure and velocity fields,
- training scalar surrogate model for Cl/Cd coefficients,
- validating both model families with dedicated evaluation workflows.

## Current Workflow (active)

1. Generate/refresh sampling table (CSV)
2. Build OpenFOAM cases with generated `blockMeshDict`
3. Run CFD chain (`blockMesh`, `checkMesh`, `decomposePar`, `simpleFoam -parallel`, `reconstructPar`)
4. Convert latest results to VTK and interpolate to regular grid
5. Save dataset as `.npz` + dataset index CSV
6. Train model selected by `model.name` in config:
	- spatial branch: `simple_unet` / `rans_pinn`
	- scalar branch: `clcd_mlp`
7. Run evaluation:
	- `scripts/evaluate_ml_model.py` for spatial branch
	- `scripts/evaluate_clcd_model.py` for scalar Cl/Cd branch

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
- [scripts/evaluate_ml_model.py](../scripts/evaluate_ml_model.py)
- [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py)

## src folder: quick module summary

- [src/config.py](../src/config.py) — loads YAML configs into typed dataclasses for paths, CFD run settings, dataset build settings, and ML training/model settings.
- [src/generate_sampling_table.py](../src/generate_sampling_table.py) — generates the sampling CSV with NACA profile combinations expanded by AoA and inlet velocity.
- [src/sampling.py](../src/sampling.py) — reads sampling CSV and converts rows to typed `BuildCaseRow` records used by the case builder.
- [src/case_builder.py](../src/case_builder.py) — creates case directories from template, writes `blockMeshDict`, updates `0/U` and `system/forceCoeffs`, and stores `params.json`.
- [src/blockmesh_generator.py](../src/blockmesh_generator.py) — procedural generator of C-grid `blockMeshDict` for NACA 4-digit profiles, including LE clustering and topology options.
- [src/file_editors.py](../src/file_editors.py) — helper text editors for OpenFOAM files (`0/U`, `system/forceCoeffs`) to inject case-specific inlet velocity values.
- [src/case_runner.py](../src/case_runner.py) — runs OpenFOAM commands per case, writes logs and run status records, and performs post-run cleanup of `processor*` folders.
- [src/flow_extractor.py](../src/flow_extractor.py) — runs `foamToVTK`, reads VTK fields, interpolates to a regular grid, creates airfoil mask, and exports compressed `.npz` datasets.
- [src/ml_dataset.py](../src/ml_dataset.py) — PyTorch dataset for loading `*_flow.npz`, parsing conditions from filenames, and normalising inputs/targets.
- [src/ml_clcd_dataset.py](../src/ml_clcd_dataset.py) — robust scalar dataset for Cl/Cd from `forceCoeffs.dat`, finite-value filtering, skip reasons, and safe normalization checks.
- [src/ml_models.py](../src/ml_models.py) — contains model architectures (`SimpleUNet`, physics-informed CNN, `ClCdMLP`) and a model factory.
- [src/ml_training.py](../src/ml_training.py) — training/evaluation utilities with masked MSE loss for fluid-domain-only optimization.
- [src/ml_validation.py](../src/ml_validation.py) — plotting utility for train/validation loss curves.
- [src/clcd_inference.py](../src/clcd_inference.py) — reusable inference API for scalar Cl/Cd model (checkpoint loading, normalization, prediction, denormalization).
- [src/evaluate_clcd_metrics.py](../src/evaluate_clcd_metrics.py) — per-case and global metrics for Cl/Cd evaluation.
- [src/evaluate_clcd_plotting.py](../src/evaluate_clcd_plotting.py) — scalar diagnostic plots (scatter, residual, sorted absolute errors).
- [src/evaluate_io.py](../src/evaluate_io.py) — shared JSON sanitization and metadata helpers used by evaluation scripts.
- [src/evaluate_metrics.py](../src/evaluate_metrics.py) — shared numeric summarization utility reused by both evaluation branches.
- [src/BACKUPblockmesh_generator.py](../src/BACKUPblockmesh_generator.py) — legacy backup version of the blockMesh generator kept for reference.
- [src/__init__.py](../src/__init__.py) — package marker (currently empty).

## Script/module dependencies (simple map)

| File | Where it is used |
|---|---|
| [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) | main build entrypoint |
| [scripts/run_cases.py](../scripts/run_cases.py) | main CFD run entrypoint |
| [scripts/extract_flow_fields.py](../scripts/extract_flow_fields.py) | main flow-field extraction entrypoint |
| [scripts/train_ml_model.py](../scripts/train_ml_model.py) | main ML training entrypoint |
| [scripts/evaluate_ml_model.py](../scripts/evaluate_ml_model.py) | spatial model evaluation entrypoint |
| [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py) | scalar Cl/Cd model evaluation entrypoint |
| [src/config.py](../src/config.py) | used by all main scripts + [src/case_runner.py](../src/case_runner.py) + [src/flow_extractor.py](../src/flow_extractor.py) |
| [src/generate_sampling_table.py](../src/generate_sampling_table.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) |
| [src/sampling.py](../src/sampling.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py); `BuildCaseRow` is used by [src/case_builder.py](../src/case_builder.py) |
| [src/case_builder.py](../src/case_builder.py) | called by [scripts/build_blockmesh_cases.py](../scripts/build_blockmesh_cases.py) |
| [src/blockmesh_generator.py](../src/blockmesh_generator.py) | used by [src/case_builder.py](../src/case_builder.py) |
| [src/file_editors.py](../src/file_editors.py) | used by [src/case_builder.py](../src/case_builder.py) |
| [src/case_runner.py](../src/case_runner.py) | used by [scripts/run_cases.py](../scripts/run_cases.py) and [src/flow_extractor.py](../src/flow_extractor.py) |
| [src/flow_extractor.py](../src/flow_extractor.py) | used by [scripts/extract_flow_fields.py](../scripts/extract_flow_fields.py) |
| [src/ml_dataset.py](../src/ml_dataset.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_clcd_dataset.py](../src/ml_clcd_dataset.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) and [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py) |
| [src/ml_models.py](../src/ml_models.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_training.py](../src/ml_training.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/ml_validation.py](../src/ml_validation.py) | used by [scripts/train_ml_model.py](../scripts/train_ml_model.py) |
| [src/clcd_inference.py](../src/clcd_inference.py) | used by [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py), reusable for future GUI/API prediction |
| [src/evaluate_clcd_metrics.py](../src/evaluate_clcd_metrics.py) | used by [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py) |
| [src/evaluate_clcd_plotting.py](../src/evaluate_clcd_plotting.py) | used by [scripts/evaluate_clcd_model.py](../scripts/evaluate_clcd_model.py) |
