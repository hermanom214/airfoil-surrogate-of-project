# Controlled GPU preflight ? 2026-10-25

Verdict: **PASS WITH CONDITIONS**. The production architecture, configured batch
size two, full 320x640 grid, float32, all configured losses, backward and one Adam
step completed on the GTX 1070 Ti. Controlled training can be attempted. This
establishes one-step feasibility, not convergence or long-run reliability.

## Scope and cases

The diagnostic did not call the training entrypoint. It used
`scripts/diagnose_unet_pinn_gpu.py`, with isolated output under
`outputs/diagnostics/unet_pinn_gpu_20261025`. Machine-readable stage, gradient,
per-case loss, sensitivity and protected-file results are in `results.json`.

Verified development cases:

- case_0001_naca0008_aoam4p0_u15p0: high near-wall reference residual.
- case_0624_naca2416_aoa4p0_u22p5: wake/loaded cambered case.
- case_0063_naca0012_aoa0p0_u20p0: ordinary symmetric case.

Case 0303 is fixed test and was excluded. Normalization used the default
development holdout's 356 fold-training cases, excluding all fixed-test fields.
The three selected cases are development members; this diagnostic did not
evaluate the fixed-test population or change any partition.

Architecture: exact existing SimpleUNet, encoders [32,64,128], bottleneck 256,
five input channels, three outputs [p,Ux,Uy]. Weights remain 1.0/.1/.1.

## GPU environment and actual memory

- GPU: NVIDIA GeForce GTX 1070 Ti; device cuda:0.
- Total: 8,589,803,520 bytes (7.999878 GiB).
- Actual free before allocations: 7,539,261,440 bytes (7.021484 GiB).
- PyTorch 2.7.1+cu118; CUDA runtime 11.8; CUDA available.
- Baseline allocated/reserved: zero/zero bytes.
- Tested batch size: **2 only**, successful. No OOM occurred; no batch-one
  fallback was needed. No larger size was tested or inferred.

All values below are cumulative synchronized CUDA allocator peaks in MiB.

| Stage, batch 2 | Peak allocated | Peak reserved |
|---|---:|---:|
| CPU dataset batch loaded | 0.000 | 0.000 |
| Model and Adam created | 7.352 | 24.000 |
| Batch copied to GPU | 29.229 | 44.000 |
| Forward | 643.041 | 660.000 |
| Supervised loss, before residuals | 660.232 | 680.000 |
| Continuity and RANS residual tensors | 683.666 | 700.000 |
| Continuity loss reduction | 683.666 | 700.000 |
| Momentum-x loss reduction | 683.666 | 700.000 |
| Momentum-y loss reduction | 683.666 | 700.000 |
| Full combined loss | 683.666 | 700.000 |
| Backward | 744.611 | 1188.000 |
| One Adam step | 744.611 | 1196.000 |

Forward/full-loss/backward-step peaks in bytes are 674277376 / 716875264 /
780781056; maximum reserved is 1254096896 bytes. Free memory after the step
was 6268387328 bytes. The allocator peaks exclude other processes/driver memory;
mem_get_info readings capture actual device free memory as well.

The real implementation builds continuity and momentum residuals together,
then performs their reductions. These are observations of that execution
order, not separately isolated or additive memory costs. Instrumentation
wrapped the existing residual and mean-square functions and synchronized CUDA;
it did not replace the physical equations.

Input [2,5,320,640], target/output [2,3,320,640], scalar masks/turbulence
[2,1,320,640], xy [2,320,640,2], references [2], all float32.
All used tensor inputs and model parameters were verified on cuda:0. Unused
omega intentionally stayed on CPU, matching the production training loop.

## Numerical losses at initial parameters

Batch: cases 0001 and 0624. All values were finite.

| Component | Raw | Weight | Weighted |
|---|---:|---:|---:|
| pressure | 2.09947658 | part of supervised mean | ? |
| Ux | 1.26363420 | part of supervised mean | ? |
| Uy | 1.79695165 | part of supervised mean | ? |
| Supervised | 1.72002077 | 1.0 | 1.72002077 |
| Continuity | .000142231642 | .1 | .000014223165 |
| Momentum-x | .000235982283 | part of momentum sum | ? |
| Momentum-y | .000094807569 | part of momentum sum | ? |
| Momentum sum | .000330789859 | .1 | .000033078988 |
| Combined total | 1.72006798 | ? | 1.72006798 |

Fluid counts were 201804/198849; evaluated physics counts 196872/193849.
The supervised reduction averages channel MSE over fluid points; momentum is
the sum of the two normalized component mean squares over the physics mask.
Mask populations were positive and the total was checked against the explicit
weighted sum. The stored radius-two masks were not changed.

Physical fields were decoded with fold-training mean/std. References were
L=1 m, U=15/22.5 m/s for the batch (20 m/s for ordinary case 0063), with
nu=1.5e-5 m^2/s. Existing Rc*=L/U*Rc and Ri*=L/U^2*Ri scaling was retained.
Manufactured scaling/masking regression tests passed.

At identical fresh-model initialization, per-case supervised/continuity/momentum
losses were .769485/.000221785/.000607162 (0001),
2.684682/.000061437/.000050108 (0624), and
.388819/.000060992/.000088528 (0063). Every channel and weighted component is
recorded in results.json. These values describe an untrained model, not fit
quality or convergence.

## Independent gradients

Raw norms span every trainable parameter. Repeated forward/backward passes used
one unchanged fresh model and the same batch; gradients were cleared between
components, no optimizer ran, and no graph was retained across passes. Gradient
vectors were copied to CPU for cosine measurements.

| Component | Raw gradient L2 | Weighted L2 | Weighted/supervised |
|---|---:|---:|---:|
| Supervised | .374454005 | .374454005 | 1 |
| Continuity | .011533459 | .001153346 | .00308007 |
| Momentum | .022353714 | .002235371 | .00596968 |

Cosines: supervised/continuity .272233, supervised/momentum .276227,
continuity/momentum .899731. Weighted physics gradients are nonzero but weak
relative to supervised (0.308% and 0.597%) at this initialization. Neither
dominates and no numerical instability was observed. The weights were not tuned
and are not established as optimal by this measurement.

Output-gradient norms [p,Ux,Uy] were:

- Supervised: [.001526090, .001183955, .001411864].
- Continuity: [0, .001627853, .000066483]; zero pressure gradient is expected.
- Momentum: [.000953238, .002835537, .000255742].

## Exactly one optimizer step and checkpoint safety

The fresh diagnostic model completed one combined backward, finite parameter
gradients, and exactly one Adam step at configured lr=.001. Parameters changed
and remained finite. No updated diagnostic weights were saved.

A separate temporary synthetic checkpoint with production architecture was
saved, resolved and loaded with exact parameter equality. It resolved to
data/models/unet_pinn_phase_a/unet_pinn_airfoil.pt within a temporary root;
the directory was then removed by its owning TemporaryDirectory context.

The actual evaluator was also exercised with a temporary mismatched model
identity. It raised: Requested model unet_pinn disagrees with checkpoint model
simple_unet. No evaluator run touched real fixed-test fields.

All **12 existing .pt checkpoints**, the YAML and saved PINN split were hashed
before/after and matched. Selected NPZ hashes also matched. The model-specific
filename and output directory isolate simple_unet and clcd_mlp artifacts.

## Interpolation sensitivity (diagnostic masks only)

Affected stencils were reconstructed from the existing audit's actual
Delaunay/analytic-geometry calculation, capturing its affected mask in memory.
Shared production residuals were evaluated on reference fields on CPU; only
the diagnostic reduction mask differed. No reference values or production
physics masks were changed. No nearest-filled fluid/stencil values were found.

All RMS below are nondimensional, current mask -> affected stencils excluded.

| Development case | Valid | Excluded | Retained | Continuity RMS | Momentum-x RMS | Momentum-y RMS |
|---|---:|---:|---:|---:|---:|---:|
| 0001 | 196872 | 29 | 196843 | .05410576 -> .05402655 | .03641576 -> .03626837 | .03525233 -> .03525364 |
| 0624 | 193849 | 59 | 193790 | .04490019 -> .04484514 | .04389224 -> .04375158 | .02301386 -> .02301642 |
| 0063 | 195356 | 32 | 195324 | .02890947 -> .02872783 | .02373820 -> .02358251 | .02014364 -> .02014452 |

All excluded points were near-wall evaluated centers:

| Case | Near-wall valid -> retained | Continuity RMS | Momentum-x RMS | Momentum-y RMS |
|---|---:|---:|---:|---:|
| 0001 | 1142 -> 1113 | .639205 -> .646249 | .428652 -> .431973 | .386812 -> .391799 |
| 0624 | 1168 -> 1109 | .414108 -> .423837 | .430877 -> .439631 | .170688 -> .175148 |
| 0063 | 1146 -> 1114 | .338020 -> .340118 | .222027 -> .222271 | .161586 -> .163874 |

Outer, intermediate and wake counts/RMS were unchanged. Outer counts were
185311/182050/183694. Full regional tables are in the JSON results.

The removed 0.015?0.030% of global points accounted for 0.275?1.269% of
continuity squared residual and 0.670?1.324% of momentum-x squared residual,
but less than .009% of momentum-y squared residual. They contribute more than
average global points to Rc/Rx because they are near the wall. Within the
near-wall population they contribute less than the average retained point;
near-wall RMS increases after exclusion. Maximum global RMS change was about
0.66%; no material domination of these metrics by the flagged stencils was
demonstrated. This does not measure true field error or prove every case safe.
Native mesh/analytic surface differences and interpolation/discretization error
remain uncertainties. These measurements do not justify a dataset rebuild for
the diagnostic or force a production-mask change.

## Integration inspection and recommendation

The unet_pinn loop clears gradients before backward, detaches logged scalars,
does not store graph tensors in history, and validation disables autograd.
It does not retain multiple models simultaneously. Validation computes all
loss components; configuration selection remains supervised development error.
The current loop creates Adam once per run and has no scheduler or resume path;
checkpoints do not save optimizer state. This is an existing limitation for
resuming, not a blocker for a fresh controlled run. No production change was
needed for the diagnostic.

Next action: **Proceed to controlled training**, starting with the tested batch
two and unchanged weights, monitoring validation components and GPU memory.
No larger batch or long-run memory behavior is established. Track whether the
initially weak physics gradients become useful during learning before deciding
on weighting changes. Keep near-wall residuals qualified by interpolation and
resolution limits. No training was started automatically.

## Code, checks and repository safety

Only two files were added by this task:

- scripts/diagnose_unet_pinn_gpu.py: isolated bounded diagnostic.
- docs/unet_pinn_gpu_preflight.md: this report.

Generated JSON outputs are isolated under outputs/diagnostics. Existing
production changes from earlier work remain untouched. Focused regression
tests: 17 passed. Script compilation and git diff --check passed. The prior
92-test full-suite result was not rerun for this diagnostic-only source addition.

No CFD/PINN artifact, inspection result, fixed-test split, model architecture,
loss weight or production checkpoint changed. No full/multi-epoch training,
hyperparameter search, dataset rebuild, CFD execution or commit was performed.
