# Phase A v1 review corrections and CUDA pilots

Date: 2026-10-26 (Europe/Prague). Verdict: PASS WITH CONDITIONS.

Full 30-epoch training is technically ready. Exactly two one-epoch CUDA pilots ran sequentially. One epoch establishes execution, memory fit, gradients, checkpoint selection and comparability; it does not establish a final accuracy benefit. The existing near-wall interpolation-support limitation documented in docs/unet_pinn_review_fixes.md remains a scientific limitation.

## Corrections

- src/unet_pinn.py adds u*div(U) and v*div(U) to momentum convection. The residual now implements conservative continuum div(U tensor U), expressed as (U.grad)U + U div(U). It does not reproduce the discrete OpenFOAM finite-volume convection scheme.
- Kinematic modified pressure, positive pressure gradient, nu+nut inside differentiated stresses, 3D deviatoric trace correction, nondimensionalization, stored masks and derivative stencils are unchanged. No explicit k stress was added.
- scripts/evaluate_ml_model.py now passes dataset.supervised_config through evaluate_pinn_physics. PinnFlowDataset supplies raw-field statistics needed by common generic evaluator metrics without altering improved targets.
- scripts/compare_unet_pinn_fields.py supports --improved and --pilot. Both checkpoint identities are validated; decoding uses saved configuration plus each case's U_inf and p_inf. Raw checkpoints remain supported. The improved default output is outputs/experiments/unet_improved_pinn_fields, optionally /pilot.
- src/unet_pinn_field_evaluation.py derives report interpretation from actual case metrics, with improved/worsened/tied counts, mean and median deltas by region and variable. Tests cover 6 and 8 cases, mixed changes and identical errors.
- The runner records normalization and a conservative-operator identifier, enforces the 558/356/90/112 population and marks skipped training physics with physics_computed=false. Skipped zero placeholders are not measured residuals.
- Pilot checkpoint validation exposed a tuple-versus-JSON-list mismatch in gradient_channel_weights. The validator now compares equivalent serialized metadata consistently. A regression test also confirms genuinely changed coefficients are rejected. The pilot checkpoint itself was not modified.

## Preserved methodology

The original src/ml_supervised.py and src/ml_training.py remain unchanged. Both arms reuse supervised_loss_components: weighted data MSE + 0.1*gradient loss, equal channel weights, weights fluid*(1+3*wall+3*wake). The near-wall distance is <=0.05 chord; wake is 0.9<=x_body/c<=1.75 and abs(y_body)/c<=0.20 using the existing AoA convention. Weights are 1/4/4/7 in ordinary/wall/wake/overlap fluid cells and zero in solids. Physics residuals receive no regional weighting.

Targets remain (p-p_inf)/(0.5*U_inf^2), (Ux-U_inf)/U_inf, Uy/U_inf; the inverse is p_inf+0.5*U_inf^2*p*, U_inf*(ux*+1), U_inf*uy*. Manufactured tests verify actual physical tensors reaching the residual and autograd, including nonzero p_inf and U_inf=7 and 23.

## Pilot evidence

GPU: NVIDIA GeForce GTX 1070 Ti, 8191.875 MiB. PyTorch 2.7.1+cu118, CUDA runtime 11.8.

Both runs completed exactly epoch 1, selected epoch 1, and passed finite-gradient checks on all 178 training batches. Both validation passes computed physics. A physics weights: 0/0; B: 0.1/0.1. Adam lr=0.001, weight_decay=0, batch=2, seed=42, dataloader_seed=42; no scheduler. Architecture: SimpleUNet, 5->3 channels, encoder [32,64,128], bottleneck 256. Checkpoint selection remains improved supervised validation loss.

| Metric | A: improved supervised | B: improved PINN |
|---|---:|---:|
| Training improved supervised loss | 0.130651532 | 0.140719475 |
| Training weighted data MSE | 0.00618567478 | 0.00686606480 |
| Training gradient loss (before factor 0.1) | 1.24465855 | 1.33853407 |
| Training total weighted loss | 0.130651532 | 0.145034895 |
| Training continuity RMS | not computed | 0.0971481390 |
| Training momentum-x RMS | not computed | 0.160629464 |
| Training momentum-y RMS | not computed | 0.0889641443 |
| Training combined momentum RMS | not computed | 0.183620379 |
| Validation improved supervised loss | 0.0942456308 | 0.106630881 |
| Validation weighted data MSE | 0.00330121260 | 0.00509771475 |
| Validation gradient loss (before factor 0.1) | 0.909444163 | 1.01533165 |
| Validation total weighted loss | 0.0942456308 | 0.113296985 |
| Validation continuity RMS | 0.245776340 | 0.121397217 |
| Validation momentum-x RMS | 0.351354513 | 0.184963807 |
| Validation momentum-y RMS | 0.187598168 | 0.133086974 |
| Validation combined momentum RMS | 0.398300222 | 0.227867841 |
| Validation pooled p RMSE [m^2/s^2] | 13.1736323 | 18.5750316 |
| Validation pooled Ux RMSE [m/s] | 0.775479380 | 0.885365380 |
| Validation pooled Uy RMSE [m/s] | 0.825329812 | 0.955884917 |
| Timed training + epoch validation [s] | 66.1021355 | 62.7419783 |
| Peak CUDA allocated [MiB] | 1085.11377 | 1085.11523 |
| Peak CUDA reserved [MiB] | 1348.00000 | 1348.00000 |

Supervised objectives retain the original mean of batch loss reductions. Residual RMS above means sqrt(mean batch masked nondimensional MSE); combined momentum RMS is sqrt(mean Rx*^2 + mean Ry*^2). These are not globally pooled physical residual RMS. Physical field errors labeled pooled use valid-cell counts. Training metrics span changing model weights; validation uses the final selected checkpoint. The duration includes the epoch training and validation pass, excluding preflight and subsequent selected-checkpoint validation/test reporting; CUDA peaks include those later evaluations.

B-A validation supervised loss: 0.0123852503 (13.1414583%). Continuity RMS change: -50.6066300%; combined momentum RMS change: -42.7899288%. Physical full-domain p/Ux/Uy RMSE all worsen after this one epoch. These mixed results do not justify changing weights or claiming final superiority.

## Reproducibility

558 immutable cases; 356 training, 90 validation, 112 fixed test.

- dataset_fingerprint: `fc8158dacfc439833d8e07199fbabffed86c58eeb79817373088095c69ba82ed`
- dataset_membership_fingerprint: `5d66d5ce62effb1496fb1947eeb05881a0ab3cf7bbccf02e01185ef72ef777ad`
- train_fingerprint: `2bd0cd82944c7f945a2c730ab9104cacc072f47692da0848ddfb85f6b362aa20`
- validation_fingerprint: `f72bdf51349061fb851b908f849f52c4d5405a63e2b28e8297dffd64be848199`
- fixed_test_fingerprint: `33d710600637b4da9e51eeed74c6e59b4d21377777846c5f1ca55b051ffe3408`
- initial_state_hash: `0e38c5fd4814d872a2655274491515930f4312b9546ea557aa2566f0586bb8d9`
- Deterministic epoch-one batch order independently reconstructed for both arms (356 distinct cases), SHA256: `52b47700ff3955551cd35c5aa51614bf79bb5b2f4947af5f899c90c702b666eb`.

Both metadata sets have identical normalization, architecture, seeds and memberships. The live post-pilot dataset artifact fingerprint matches preflight. SHA256 checks of 49 pre-existing artifacts (old raw A/B full and pilot outputs, original improved U-Net artifacts and existing split JSONs) show no changes.

## Validation

- Final full suite: 141 passed, no failures (40.46 s). The earlier full suite passed 139 before the serialization and raw-statistics regression tests were added.
- Focused physics, evaluator, improved-loss, field-comparison and reporting tests passed before pilots; the serialization correction passed 14 reporting tests and validated the real A checkpoint before B began.
- Python compileall for src, scripts and tests passed; git diff --check passed (line-ending notices only).
- Improved pilot comparator exited 0 with controlled_comparison=true and compatibility_problems=[]. Both checkpoints load strictly and have finite float32 state.
- Generic evaluator and field decoding are regression-tested for raw and improved targets. No old v0 report was regenerated; no full generic 112-case plotting run was performed.

## Artifacts

- data/models/unet_improved_supervised/pilot/unet_pinn_airfoil.pt
- data/models/unet_improved_pinn/pilot/unet_pinn_airfoil.pt
- Each pilot directory contains resolved_config.json, experiment_metadata.json, split_membership.json, training_history.json, validation_metrics.json and test_metrics.json.
- outputs/experiments/unet_improved_pinn_ab/pilot/comparison.json

Full 30-epoch training was not run. No dataset or split was regenerated. No pre-existing checkpoint was overwritten. No experiment was deleted. No commit was created. Existing unrelated configuration, requirements and ignore-rule changes were preserved.

Repository branch main; HEAD 5e76a2613f1ef9ce5958a6372253506355e7f526.
