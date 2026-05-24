# CFD Workflow (OpenFOAM)

## Active CFD chain per case

1. `blockMesh`
2. `checkMesh`
3. `decomposePar -force`
4. `simpleFoam -parallel`
5. `reconstructPar -latestTime`

These steps are defined in [configs/solver_config.yaml](../configs/solver_config.yaml) under `case_runner.run_steps`.

## Where configuration is read

- runner logic: [src/case_runner.py](../src/case_runner.py)
- launcher script: [scripts/run_cases.py](../scripts/run_cases.py)

The following are config-driven:
- number of MPI processes (`case_runner.n_procs`)
- required case subdirectories
- exact command list and per-step logs
- parallel worker count for multi-case execution (`run_cases.max_workers`)

## Case locations

- run root: `openfoam_run_root_windows` + `blockmesh_cases_subdir`
- simulation mirror: `openfoam_case_sim_windows` + `blockmesh_cases_subdir`

Both roots come from [configs/paths.yaml](../configs/paths.yaml) + [configs/dataset_config.yaml](../configs/dataset_config.yaml).

## Logs and run summary

For each case:
- `logs/01_blockMesh.log`
- `logs/02_checkMesh.log`
- `logs/03_decomposePar.log`
- `logs/04_simpleFoam.log`
- `logs/05_reconstructPar.log`

Global summary:
- `run_status.csv` in the cases root

## Notes

- `simpleFoam` stays parallel (MPI), as required by current workflow.
- If one step fails for a case, remaining steps for that case are skipped and status is marked as failed.
