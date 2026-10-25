# Phase A review fixes and diagnostic readiness

Verdict: **PASS WITH CONDITIONS** for a short diagnostic GPU run. No GPU run
was performed. CUDA memory use on the GTX 1070 Ti remains unmeasured. The
interpolation investigation found a localized geometry-support limitation;
near-wall physics metrics must therefore be treated as diagnostic estimates.

## Confirmed findings and changes

- `src/unet_pinn.py`: shared `rans_residuals`, corrected modified-pressure
  momentum equation, and direct use of the already eroded stored mask.
- `src/audit_unet_pinn.py`: uses the training operator and mask, reports
  evaluated/region counts and verifies stored radius-two stencil coverage.
- `src/audit_pinn_interpolation.py`: read-only reconstruction of p/U convex-hull
  and solid-intersection support, including all actual derivative dependencies.
- `scripts/evaluate_ml_model.py`: explicit `--model`, model-specific filename,
  PINN dataset, checkpoint identity guard, and PINN physics metrics. Explicit
  experiment directories no longer silently fall back to a legacy checkpoint.
- `src/evaluate_io.py`: reads PINN metadata and constructs U-Net model kwargs.
- `src/ml_data_split.py`: rejects duplicated, overlapping, missing, unknown or
  empty partitions before mapping persisted IDs to indices. Existing membership
  and fingerprint checks remain in force; duplicate current IDs also fail.
- `scripts/train_ml_model.py`: names the supervised selection metric in results
  and checkpoints and avoids transferring unused omega tensors to the GPU.
- `tests/test_unet_pinn.py`: manufactured equations, scaling, mask, split and
  synthetic-checkpoint/evaluator-dataset regression coverage.

Existing user changes in configuration, requirements, plotting and ignore rules
were preserved. This review task did not change the SimpleUNet architecture,
configuration weights, input/output channels, or Cl/Cd model/evaluator.

## Solver evidence and equation

The actual archived case log is
`data/flow_fields/case_0001_naca0008_aoam4p0_u15p0/logs/04_simpleFoam.log`.
It identifies OpenFOAM for Windows 20.09, CFD Support/Symscape packaging,
build `dev-d26757ee1926`, simpleFoam, Newtonian incompressible transport,
RAS kOmegaSST, no MRF models and no finite-volume options.

The installed source is under
`C:/OpenFOAM/20.09/cygwin64/opt/OpenFOAM/OpenFOAM-dev`:

- `applications/solvers/incompressible/simpleFoam/UEqn.H` uses
  `fvm::div(phi,U) + turbulence->divDevReff(U)` and solves against `-grad(p)`.
- `src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C`
  implements the negative Laplacian plus negative divergence of
  `nuEff*dev2(T(grad(U)))`; there is no explicit k gradient.
- The incompressible transport constructor uses unit alpha and rho fields.

Generated case dictionaries are under
`run/airfoil_surrogate_cases/blockmesh_cases/<case_id>`, not repository root.
The archived `system/fvSchemes` sets `steadyState` and the corresponding
`div((nuEff*dev2(T(grad(U)))))` scheme. Pressure dimensions are [0 2 -2],
nu is 1.5e-5 m^2/s. Extractors carry pressure values through unchanged.

Original residual: R = U.grad(U) + grad(p) - div(tau), with
tau = 2*(nu+nut)*dev(S) - 2*k*I/3. This added an erroneous +2*grad(k)/3
relative to the installed solver's pressure/operator pairing.

Corrected residual:

    Rc = du/dx + dv/dy
    Txx = (nu+nut)*(2*du/dx - 2*Rc/3)
    Tyy = (nu+nut)*(2*dv/dy - 2*Rc/3)
    Txy = (nu+nut)*(du/dy + dv/dx)
    Rx = u*du/dx + v*du/dy + dp/dx - dTxx/dx - dTxy/dy
    Ry = u*dv/dx + v*dv/dy + dp/dy - dTxy/dx - dTyy/dy

The trace correction is three-dimensional for U=(u,v,0). Spatially varying nut
remains inside the differentiated stress. No density division or pressure
redefinition is applied. Reference nut is prescribed during physics evaluation;
k and omega remain available in the dataset but neither enters this operator.
Network inference does not require any CFD turbulence fields.

The source is the installed distribution, not a cryptographically verified
reconstruction of the historical executable. Its operator agrees with archived
case schemes and logs. The Cartesian centered operator does not reproduce
OpenFOAM's native face fluxes, bounded convection or mesh corrections.

## Mask and population correction

Extraction erodes fluid AND turbulence interpolation support twice with a 3x3
square, equivalent to radius two. Nested centered derivatives need +/-2 axial
points and +/-1 diagonal points, all inside that square. Training and auditing
previously eroded this mask again, making the effective radius four. They now
use the stored radius-two mask; an explicit two-cell domain-border exclusion
does not remove any valid cells from the actual artifacts.

All three audited stored masks have zero unsafe radius-two stencils.

| Case | Fluid | Physics/evaluated | Near-wall fluid | Old near-wall evaluated | New near-wall evaluated | New wake | New outer |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 201804 | 196872 | 2250 | 54 | 1142 | 20246 | 185311 |
| 0303 | 200334 | 195362 | 2292 | 44 | 1144 | 20241 | 183697 |
| 0624 | 198849 | 193849 | 2344 | 75 | 1168 | 20235 | 182050 |

Regions follow the existing continuity audit: near wall 0<SDF/chord<=0.02,
intermediate through 0.10, outer above 0.10; wake overlaps the distance bands.

## Interpolation investigation

Pressure/velocity use SciPy griddata linear interpolation of projected VTK
source points, with nearest filling outside their convex hull. Turbulence is
attached to mesh cells, converted to point data, then sampled through VTK mesh
cells; vtkValidPointMask records that mesh support. These methods use related
geometry but neither equivalent source associations nor equivalent domains.
Delaunay triangulation can bridge the solid; VTK cell sampling respects mesh
topology. The current PINN mask records the latter support only.

Read-only reconstruction used 733284 source points per requested case:

| Case | Nearest-filled fluid / stencil inputs | Solid-crossing triangle stencil-input points | Evaluated stencils depending on those triangles | Maximum triangle area fraction inside analytic solid |
|---|---:|---:|---:|---:|
| 0001 | 0 / 0 | 69 | 29 | 0.5765 |
| 0303 | 0 / 0 | 72 | 23 | 0.4720 |
| 0624 | 0 / 0 | 102 | 59 | 0.8916 |

No evaluation center itself used a solid-crossing triangle. However, these
geometry-adjacent derivative inputs establish an actual support-provenance gap.
They do not quantify pressure/velocity error or establish that the entire
dataset is corrupted. The comparison uses the same analytic NACA geometry as
the repository masks; native mesh surface discretization can differ slightly.

Current fields/masks were not changed to suppress this evidence. No rebuild is
required for a diagnostic run. Before relying on near-wall PDE residuals, a
separate extraction follow-up should preserve p/U support and simplex geometry
validity (or use consistent mesh-aware sampling), combine support before one
stencil erosion, and rebuild only under separate authorization. This review
does not introduce an unmeasured mask threshold or silently exclude these cells.

## Reference residual audit

Full IDs are case_0001_naca0008_aoam4p0_u15p0,
case_0303_naca1412_aoam4p0_u20p0 and case_0624_naca2416_aoa4p0_u22p5.
All values below are nondimensional RMS: Rc*=L/U*Rc and Ri*=L/U^2*Ri.
Old values are from the original recorded audit.

| Case | Old Rc | New Rc | Old Rx | New Rx | Old Ry | New Ry |
|---|---:|---:|---:|---:|---:|---:|
| 0001 | .032982 | .054106 | .017321 | .036416 | .020334 | .035252 |
| 0303 | .020543 | .035201 | .015227 | .028361 | .019451 | .023670 |
| 0624 | .031679 | .044900 | .027279 | .043892 | .023274 | .023014 |

| Case/region | New Rc | New Rx | New Ry |
|---|---:|---:|---:|
| 0001 near wall | .639205 | .428652 | .386812 |
| 0001 wake | .117517 | .067278 | .036733 |
| 0001 outer | .012495 | .010616 | .004533 |
| 0001 intermediate | .088061 | .053976 | .081955 |
| 0303 near wall | .404785 | .310748 | .197991 |
| 0303 wake | .079951 | .050885 | .043270 |
| 0303 outer | .007114 | .008532 | .005977 |
| 0303 intermediate | .065645 | .056257 | .074281 |
| 0624 near wall | .414108 | .430877 | .170688 |
| 0624 wake | .104950 | .084608 | .041633 |
| 0624 outer | .015931 | .014591 | .007804 |
| 0624 intermediate | .116509 | .105288 | .073579 |

Continuity's equation did not change: its RMS increase is solely a population
change. Momentum changes reflect both population and the removed k term; the
table does not isolate their effects. Near-wall interpolation and resolution
remain major limitations. No reference fields were adjusted.

## Evaluation, selection and split contract

    python -m scripts.evaluate_ml_model --model unet_pinn --experiment-name unet_pinn_phase_a --device cpu

This resolves data/models/unet_pinn_phase_a/unet_pinn_airfoil.pt and loads
data/pinn_dataset. Explicit model/checkpoint disagreement raises an error.
Legacy simple_unet defaults and the separate Cl/Cd evaluator remain supported.

Configuration selection continues to minimize development-validation
loss_supervised. Physics metrics remain independently exposed. Metadata states
that final weights are taken after the configured epochs; there is no newly
introduced best-epoch selector or early stopping. Fixed test is reporting only.

The saved split still contains 446 development and 112 test cases from 558 NPZs.
Membership fingerprint:
5d66d5ce62effb1496fb1947eeb05881a0ab3cf7bbccf02e01185ef72ef777ad.
Split-file SHA256 remains:
6ebe7812880f89cd8001220b81578d3e158446d5c17af94b56a64d4715806d71.

## Testing and GPU readiness

- Focused physics/evaluator/split tests: 17 passed; focused plus experiment
  infrastructure: 41 passed. Full suite: 92 passed.
- Full-grid CPU smoke: DataLoader input (1,5,320,640), output (1,3,320,640),
  small existing U-Net channel configuration [4,8,16]/32, finite supervised,
  continuity and both momentum losses; finite gradients on all parameters.
  No optimizer step or checkpoint write was performed.
- Python compilation and git diff --check passed.
- Synthetic checkpoints were created only in temporary test directories.

The configured architecture remains [32,64,128]/256, batch two, float32.
Transfers are device-consistent; omega stays on CPU because unused. Logged
metrics detach to scalars, backward releases saved graph tensors, and graphs
are not accumulated across batches. Full-resolution convolution activations
and cuDNN workspace dominate memory; physics adds full-grid derivative/stress
graphs. Neither CUDA peak memory nor batch-two fit has been measured. The GPU
diagnostic should establish memory fit before longer runs and may require a
smaller batch. The CPU smoke does not establish full-architecture GPU capacity.

Remaining risks, ranked: (1) geometry/interpolation and near-wall discrete
residual accuracy, (2) unmeasured 8 GB GPU memory fit, (3) conditional reference
nut and absence of a closed turbulence model or boundary-condition loss,
(4) source/binary historical identity not cryptographically established.

No CFD/PINN artifacts, split memberships or existing checkpoints were changed.
No training run, dataset rebuild, deletion or commit was performed.
