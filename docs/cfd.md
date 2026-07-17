# CFD Workflow (OpenFOAM)

## Active CFD chain per case

1. `blockMesh`
2. `checkMesh`
3. `decomposePar -force`
4. `simpleFoam -parallel`
5. `reconstructPar -latestTime`

These steps are defined in [configs/solver_config.yaml](../configs/solver_config.yaml) under `case_runner.run_steps`.

Besides field outputs (`p`, `U`) used by spatial surrogate models,
the CFD workflow also provides force-coefficient history (`forceCoeffs.dat`) used by
the scalar `clcd_mlp` training/evaluation branch.

## CFD base case setup (manual-style summary)

Template used for all generated cases:
- [templates/openfoam_base_case_yPlus1](../templates/openfoam_base_case_yPlus1)
- configured in [configs/dataset_config.yaml](../configs/dataset_config.yaml) via `template_case_relpath`

### Solver and turbulence model

- Solver: `simpleFoam` (steady, incompressible RANS)
- Turbulence framework: `RAS`
- RAS model: `kOmegaSST`
- Molecular kinematic viscosity: `nu = 1.5e-05 m2/s`

Reference files:
- [templates/openfoam_base_case_yPlus1/constant/turbulenceProperties](../templates/openfoam_base_case_yPlus1/constant/turbulenceProperties)
- [templates/openfoam_base_case_yPlus1/constant/transportProperties](../templates/openfoam_base_case_yPlus1/constant/transportProperties)

### Boundary-condition philosophy (low-Re near wall)

The base case is set up as an external aerodynamics domain with velocity inlet and pressure outlet, slip far boundaries, and 2D extrusion (`empty` on front/back).

Velocity and pressure:
- `U`: inlet `fixedValue`, outlet `zeroGradient`, top/bottom/farfield `slip`, airfoil `noSlip`
- `p`: outlet `fixedValue 0`, inlet/top/bottom/farfield/airfoil `zeroGradient`

Turbulence quantities (current template behavior):
- `k`: inlet `fixedValue 0.002`, airfoil `fixedValue 1e-10`
- `omega`: inlet `fixedValue 300`, airfoil `omegaWallFunction` (with very low initial value)
- `nut`: airfoil `nutLowReWallFunction`, other patches `calculated`

This is a low-Re oriented wall treatment strategy (resolved near-wall mesh with y+ target around 1), combined with SST transport equations.

Reference files:
- [templates/openfoam_base_case_yPlus1/0/U](../templates/openfoam_base_case_yPlus1/0/U)
- [templates/openfoam_base_case_yPlus1/0/p](../templates/openfoam_base_case_yPlus1/0/p)
- [templates/openfoam_base_case_yPlus1/0/k](../templates/openfoam_base_case_yPlus1/0/k)
- [templates/openfoam_base_case_yPlus1/0/omega](../templates/openfoam_base_case_yPlus1/0/omega)
- [templates/openfoam_base_case_yPlus1/0/nut](../templates/openfoam_base_case_yPlus1/0/nut)

### Iteration and write policy

The run is fixed to 1000 SIMPLE iterations per case:
- `startTime 0`
- `endTime 1000`
- `deltaT 1`

So the solver advances pseudo-time steps 0 -> 1000, i.e. exactly 1000 SIMPLE outer iterations for each case.

Current disk-saving write setup:
- `writeControl timeStep`
- `writeInterval 1000` (write at the end)
- `purgeWrite 1`
- `writeFormat binary`
- `writeCompression on`

Reference file:
- [templates/openfoam_base_case_yPlus1/system/controlDict](../templates/openfoam_base_case_yPlus1/system/controlDict)

### Numerical schemes used (fvSchemes)

From [templates/openfoam_base_case_yPlus1/system/fvSchemes](../templates/openfoam_base_case_yPlus1/system/fvSchemes):

- Time derivative: `ddtSchemes.default = steadyState`
- Gradient: `gradSchemes.default = Gauss linear`
- Divergence:
	- `div(phi,U) = bounded Gauss linearUpwind grad(U)`
	- `div(phi,k) = bounded Gauss limitedLinear 1`
	- `div(phi,epsilon) = bounded Gauss limitedLinear 1`
	- `div(phi,omega) = bounded Gauss limitedLinear 1`
	- `div(phi,v2) = bounded Gauss limitedLinear 1`
	- `div((nuEff*dev2(T(grad(U))))) = Gauss linear`
	- `div(nonlinearStress) = Gauss linear`
- Laplacian: `laplacianSchemes.default = Gauss linear corrected`
- Interpolation: `interpolationSchemes.default = linear`
- Surface-normal gradient: `snGradSchemes.default = corrected`
- Wall distance: `wallDist.method = meshWave`

### SIMPLE and linear-solver controls (fvSolution)

From [templates/openfoam_base_case_yPlus1/system/fvSolution](../templates/openfoam_base_case_yPlus1/system/fvSolution):

- Pressure solver: `GAMG`, `tolerance 1e-07`, `relTol 0.02`
- `U, k, epsilon, omega, f, v2` solver: `smoothSolver` + `symGaussSeidel`, `tolerance 1e-06`, `relTol 0.05`
- SIMPLE:
	- `nNonOrthogonalCorrectors 2`
	- `consistent yes`
	- residual control: `p 1e-5`, `U 1e-5`, `(k|omega) 1e-5`
- Relaxation factors:
	- fields: `p 0.3`
	- equations: `U 0.5`, `k 0.3`, `omega 0.2`

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

Per-case force coefficients (for scalar branch):
- `postProcessing/forceCoeffs1/0/forceCoeffs.dat`

## Notes

- `simpleFoam` stays parallel (MPI), as required by current workflow.
- If one step fails for a case, remaining steps for that case are skipped and status is marked as failed.
- Scalar Cl/Cd ML pipeline reads only converged finite tail data from `forceCoeffs.dat`; incomplete or corrupted cases are skipped during dataset build.
