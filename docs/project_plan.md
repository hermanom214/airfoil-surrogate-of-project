# Project Plan (Current Branch)

## Goal

Deliver a stable and reproducible end-to-end workflow:
- blockMesh-based CFD case generation,
- OpenFOAM run automation,
- flow-field export to NPZ,
- surrogate-model training from config-driven settings.

## Current Status

### Done

- migrated to blockMesh-only pipeline (legacy STL/snappy branch is inactive)
- centralized key runtime/build/training settings into YAML configs
- refactored core modules (`case_builder`, `case_runner`, `flow_extractor`, `sampling`, `config`)
- implemented config-driven ML training with selectable model family (`simple_unet`, `rans_pinn`)
- reduced CFD disk usage:
  - `processor*` folders are auto-removed after each finished case
  - compressed binary write in `controlDict` + `purgeWrite 1`
- refreshed technical docs for overview, pipeline, CFD setup, mesh setup, and ML models

### In Progress

- mesh-quality and wake-length tuning for robust convergence across all sampled cases
- full-sweep stability checks (failure rate, convergence consistency)
- iterative calibration of physics-loss contribution for `rans_pinn`

## Next Steps (Priority)

1. Add config validation (schema + sanity bounds) before running build/train scripts.
2. Add a smoke-test mode for a very small subset (`build -> run -> extract -> train`).
3. Add automatic run report after CFD stage:
	- success/fail counts,
	- runtime distribution,
	- simple convergence summary from logs.
4. Add NPZ dataset QA checks:
	- NaN/Inf scan,
	- shape consistency,
	- basic physical range checks.
5. Version experiments by saving config snapshots with each model artifact.

## Risks

- environment/path mismatch between Windows host and OpenFOAM runtime
- mesh-parameter changes can degrade stability for part of the case sweep
- full parameter sweeps can be expensive in compute time and storage

## Working Rules

- keep workflow changes config-first whenever possible (minimize hardcoded switches)
- track every experiment with explicit config version + output artifacts
- keep a small regression subset (1-3 representative cases) as a pre-check gate