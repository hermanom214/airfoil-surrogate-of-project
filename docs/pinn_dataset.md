# Physics-ready RANS dataset

`data/pinn_dataset/` is a separate derived dataset for future physics-informed RANS/SST surrogate experiments. It does not replace or modify the supervised U-Net dataset in `data/flow_fields/`.

Each successful case is stored as `data/pinn_dataset/cases/<case_id>/<case_id>.npz`. Raw physical values are not normalized. Every artifact contains:

- `xy`: `(320, 640, 2)`, metres;
- `fluid_mask`: `(320, 640)`, 1 in fluid and 0 inside the airfoil;
- `airfoil_sdf`: `(320, 640)`, metres, positive in fluid, negative inside the solid, zero on the reconstructed rotated NACA contour;
- `interpolation_valid_mask`: cells successfully sampled from the source CFD mesh and outside the solid;
- `physics_valid_mask`: the preceding masks eroded by two pixels, suitable for centered derivative stencils with radius at most two;
- `p`: kinematic pressure in `m^2/s^2`;
- `U`: `(320, 640, 2)` velocity in `m/s`;
- `k`: turbulent kinetic energy in `m^2/s^2`;
- `omega`: specific dissipation rate in `1/s`;
- `nut`: turbulent kinematic viscosity in `m^2/s`;
- `metadata`: JSON containing case identity, NACA parameters, AoA, `U_inf`, chord, `nu`, Reynolds number, units, source time and validation information.

The Cartesian grid and the `p`, `U`, and mask arrays are read from the existing U-Net NPZ artifact, ensuring exact grid compatibility without rewriting that dataset. Final-time (`1000`) OpenFOAM binary internal fields `k`, `omega`, and `nut` are attached to the archived VTK mesh and sampled to this grid with PyVista. As in the established U-Net extractor, points outside linear interpolation support receive a finite nearest-neighbour value. `interpolation_valid_mask` preserves the original support information, and those filled cells are excluded from `physics_valid_mask`.

The molecular viscosity is parsed from each corresponding generated case's `constant/transportProperties`; it is not taken from documentation or hardcoded. Reynolds number is `Re = U_inf * chord / nu`.

The geometric SDF is distinct from OpenFOAM turbulence-model wall distance. The archived final-time cases do not contain an OpenFOAM `y`/wall-distance field and do not retain `constant/polyMesh`, so wall distance cannot be regenerated from these archives alone. Given a complete original OpenFOAM case including `constant/polyMesh`, it can be generated as post-processing (for example `postProcess -func wallDistance -time 1000`) without rerunning `simpleFoam`. That solver-derived field may reflect OpenFOAM patch selection and wall-distance algorithms; it must not be silently substituted by the analytic airfoil SDF in an SST residual.

Run a representative validation without writing files:

```powershell
.venv\Scripts\python.exe -m scripts.extract_pinn_dataset --validate-only `
  --case-id case_0001_naca0008_aoam4p0_u15p0 `
  --case-id case_0625_naca2416_aoa4p0_u25p0
```

Run the full fixed-split extraction once the subset passes:

```powershell
.venv\Scripts\python.exe -m scripts.extract_pinn_dataset
```

The extractor refuses to target `data/flow_fields`, refuses to reuse an existing output directory, never clips suspicious turbulence values, and fails the full run when any of the 588 persistent split cases is missing or invalid. `manifest.csv`, `validation_summary.json`, and an unchanged copy of the source fixed split document the result.
