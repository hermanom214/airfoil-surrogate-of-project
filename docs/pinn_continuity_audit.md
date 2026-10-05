# PINN dataset continuity audit

## Scope and outcome

This is a read-only audit of all 565 valid cases in `data/pinn_dataset/cases`. It computes the incompressible continuity residual from the stored velocity field without changing the dataset, model code, or training configuration.

The dataset is usable for physics-informed experiments, but continuity residuals are strongly concentrated near the airfoil and in the wake. A uniform pointwise continuity loss would therefore be dominated by near-wall interpolation/discretization error. Experiments should retain the supplied stencil-safe mask and report region-separated metrics; any additional exclusion or weighting must be treated as an explicit modelling choice.

## Stored representation and derivative convention

Inspection of the NPZ files and extraction code established:

- `xy.shape == (320, 640, 2)` and `U.shape == (320, 640, 2)`.
- Array axis 0 is physical y; array axis 1 is physical x.
- `xy[..., 0]` is x and `xy[..., 1]` is y.
- `U[..., 0]` is `Ux` and `U[..., 1]` is `Uy`.
- The grid is uniform, with x in `[-0.75, 1.75]` m, y in `[-0.75, 0.75]` m, `dx = 0.0039123631` m, and `dy = 0.0047021944` m.

The principal dimensional residual is

`R_dim = dUx/dx + dUy/dy` [1/s],

using second-order centered differences. A point is evaluated only when it is in the stored `physics_valid_mask`. Independently reconstructed cross-shaped stencil checks confirmed that every stored physics-mask point is safe for both the second-order radius-1 stencil and the optional fourth-order radius-2 check. There were zero unsafe points across all cases.

Nondimensional residuals use only case metadata:

`R* = (chord / U_inf) R_dim`.

## Regions

Wall distance is the stored signed distance divided by chord; positive values are fluid:

- near wall: `0 < SDF/chord <= 0.02`
- intermediate: `0.02 < SDF/chord <= 0.10`
- outer: `SDF/chord > 0.10`
- wake: an overlapping diagnostic downstream of the AoA-rotated trailing edge through `x/chord = 1.75`, within `|y/chord - y_TE/chord| <= 0.25`

The first three regions partition all evaluated fluid points. The wake intentionally overlaps those distance bands. The near-wall band is approximately 4.3 y-cells or 5.1 x-cells wide for the one-metre chord, so it is resolved by several grid points rather than being a single-cell category.

## Results

Across 110,409,670 evaluated points, the pooled dimensional mean absolute residual is `0.06975 1/s`, RMS is `0.83482 1/s`, and maximum absolute residual is `367.80 1/s`. The pooled nondimensional mean absolute residual is `0.003520`, RMS is `0.04154`, and maximum absolute residual is `15.255`.

| Region | Points | Pooled mean `|R*|` | Pooled RMS `R*` | Median per-case RMS | 95th percentile per-case RMS |
|---|---:|---:|---:|---:|---:|
| Near wall | 650,236 | 0.15387 | 0.42411 | 0.38826 | 0.64503 |
| Intermediate | 5,945,245 | 0.01645 | 0.09799 | 0.08854 | 0.13908 |
| Outer | 103,814,189 | 0.001838 | 0.01259 | 0.01082 | 0.01684 |
| Wake | 11,442,626 | 0.01325 | 0.10348 | 0.09126 | 0.15093 |

Across cases, full-domain nondimensional RMS has mean `0.04045`, median `0.03784`, 95th percentile `0.05604`, minimum `0.02891`, and maximum `0.14161`. The lowest, median, and highest cases are respectively:

- `case_0063_naca0012_aoa0p0_u20p0`: `0.02891`
- `case_0006_naca0008_aoam2p0_u15p0`: `0.03784`
- `case_0227_naca1216_aoam4p0_u17p5`: `0.14161`

Group summaries are stored for NACA code, AoA, inlet speed, and Reynolds number. Mean case RMS varies more with geometry/AoA than inlet speed in this dataset: the AoA group means range from `0.03459` at -2 degrees to `0.04940` at +4 degrees, while inlet-speed group means range only from `0.03974` to `0.04103`. This is descriptive, not a causal claim, because the valid-case design is not perfectly balanced after invalid cases were excluded.

## Derivative sensitivity

A fourth-order centered calculation was compared on the common stored physics mask for the low, median, and high representative cases. Its RMS `R*` values were `0.03292`, `0.03283`, and `0.15413`, versus second-order values `0.02891`, `0.03784`, and `0.14161`. The corresponding RMS differences were `0.01700`, `0.01642`, and `0.04046`.

The broad conclusion—small outer-field residuals and concentrated near-wall/wake residuals—is not an artifact of choosing the second-order stencil. Pointwise extremes and representative RMS values are nevertheless derivative-scheme sensitive, as expected for interpolated CFD fields on a Cartesian grid near a curved solid boundary.

## Interpretation and limitations

The residual is a diagnostic of the archived, interpolated velocity field, not a direct audit of the native OpenFOAM finite-volume flux balance. Centered Cartesian derivatives do not reproduce OpenFOAM face-flux continuity, and interpolation onto the regular grid can introduce divergence, especially near the curved airfoil boundary and steep wake gradients. The maximum values are therefore not evidence that the source CFD solve itself violated mass conservation by the same amount.

For PINN/RANS work, the stored mask is sufficient for centered derivatives and safely excludes the solid, interpolation-invalid points, array boundaries, and a two-pixel neighbourhood around invalid points. Recommended experimental reporting is full-domain plus near-wall, intermediate, outer, and wake metrics. A physics loss should be normalized consistently with `R*`; region balancing, robust penalties, or additional boundary-aware treatment may be worth testing, but this audit deliberately makes no training-code changes.

## Artifacts and reproducibility

- Per-case exact statistics: `data/pinn_dataset/continuity_audit/per_case_metrics.csv`
- Aggregate and grouped machine-readable results: `data/pinn_dataset/continuity_audit/aggregate_summary.json`
- Figures: `data/pinn_dataset/continuity_audit/figures/`
- Reproduction command: `.venv/Scripts/python.exe scripts/audit_pinn_continuity.py --overwrite`

Figures:

- [Per-case RMS histogram](../data/pinn_dataset/continuity_audit/figures/rms_histogram.png)
- [Region boxplots](../data/pinn_dataset/continuity_audit/figures/region_boxplots.png)
- [Residual versus AoA](../data/pinn_dataset/continuity_audit/figures/residual_vs_aoa.png)
- [Residual versus Reynolds number](../data/pinn_dataset/continuity_audit/figures/residual_vs_re.png)
- [Low, median, and high spatial maps](../data/pinn_dataset/continuity_audit/figures/representative_spatial_maps.png)

Spatial maps use the physical x/y aspect ratio, show the zero-SDF airfoil contour, and use a shared symmetric colour limit at the pooled 99th percentile of `|R*|` over the three selected maps (`0.12162`). Values beyond that limit are saturated, not discarded from statistics.
