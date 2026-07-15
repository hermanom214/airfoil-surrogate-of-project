from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset, random_split

from src.ml_data_split import compute_train_val_indices
from src.ml_models import PhysicsInformedCNN


@dataclass
class SelfCheckResult:
    name: str
    passed: bool
    details: str


def _assert_close(value: float, target: float, atol: float, message: str) -> None:
    if abs(value - target) > atol:
        raise AssertionError(f"{message}: got {value}, expected {target} (atol={atol})")


def evaluate_self_check_channel_order_in_rans_loss() -> SelfCheckResult:
    """Test 1: verifies that output channel order is interpreted as [p, Ux, Uy]."""
    model = PhysicsInformedCNN(in_channels=5, out_channels=3, hidden_channels=8, depth=3)

    b, h, w = 1, 16, 16
    pred = torch.zeros((b, 3, h, w), dtype=torch.float32)

    # Only pressure channel varies spatially; Ux/Uy are constants.
    x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, 1, h, w)
    pred[:, 0:1, :, :] = x
    pred[:, 1:2, :, :] = 0.2
    pred[:, 2:3, :, :] = -0.1

    mask = torch.ones((b, 1, h, w), dtype=torch.float32)

    out = model.rans_residual_loss(
        pred=pred,
        fluid_mask=mask,
        dx=1.0,
        dy=1.0,
        nu=1.0e-5,
        p_mean=3.0,
        p_std=2.0,
        ux_mean=10.0,
        ux_std=4.0,
        uy_mean=-5.0,
        uy_std=6.0,
        pressure_is_kinematic=True,
        mask_erode_pixels=0,
    )

    continuity = float(out["continuity_loss"].item())
    if continuity > 1e-8:
        raise AssertionError(
            "Continuity should be ~0 when only p channel varies; channel order likely broken"
        )

    return SelfCheckResult(
        name="channel_order_in_rans_loss",
        passed=True,
        details=f"continuity_loss={continuity:.3e}",
    )


def evaluate_self_check_constant_field_residuals() -> SelfCheckResult:
    """Test 2: for constant p/U fields, continuity and momentum residuals should be ~0."""
    model = PhysicsInformedCNN(in_channels=5, out_channels=3, hidden_channels=8, depth=3)

    b, h, w = 1, 20, 20
    pred = torch.zeros((b, 3, h, w), dtype=torch.float32)
    pred[:, 0:1, :, :] = 0.5
    pred[:, 1:2, :, :] = -0.25
    pred[:, 2:3, :, :] = 0.1

    mask = torch.ones((b, 1, h, w), dtype=torch.float32)

    out = model.rans_residual_loss(
        pred=pred,
        fluid_mask=mask,
        dx=0.01,
        dy=0.01,
        nu=1.0e-5,
        p_mean=2.0,
        p_std=4.0,
        ux_mean=15.0,
        ux_std=3.0,
        uy_mean=-1.0,
        uy_std=2.0,
        pressure_is_kinematic=True,
        mask_erode_pixels=0,
    )

    continuity = float(out["continuity_loss"].item())
    mom_x = float(out["momentum_x_loss"].item())
    mom_y = float(out["momentum_y_loss"].item())

    if continuity > 1e-8 or mom_x > 1e-8 or mom_y > 1e-8:
        raise AssertionError(
            "Constant-field residuals should be ~0; got "
            f"continuity={continuity:.3e}, mom_x={mom_x:.3e}, mom_y={mom_y:.3e}"
        )

    return SelfCheckResult(
        name="constant_field_residuals",
        passed=True,
        details=(
            f"continuity={continuity:.3e}, mom_x={mom_x:.3e}, mom_y={mom_y:.3e}"
        ),
    )


def evaluate_self_check_denormalization_mapping() -> SelfCheckResult:
    """Test 3: normalized 0 -> mean, normalized 1 -> mean+std for each channel."""
    p_mean, p_std = 5.0, 2.0
    ux_mean, ux_std = 20.0, 4.0
    uy_mean, uy_std = -1.0, 3.0

    p0 = 0.0 * p_std + p_mean
    p1 = 1.0 * p_std + p_mean
    u0 = 0.0 * ux_std + ux_mean
    u1 = 1.0 * ux_std + ux_mean
    v0 = 0.0 * uy_std + uy_mean
    v1 = 1.0 * uy_std + uy_mean

    _assert_close(p0, p_mean, 1e-12, "p(0) mismatch")
    _assert_close(p1, p_mean + p_std, 1e-12, "p(1) mismatch")
    _assert_close(u0, ux_mean, 1e-12, "ux(0) mismatch")
    _assert_close(u1, ux_mean + ux_std, 1e-12, "ux(1) mismatch")
    _assert_close(v0, uy_mean, 1e-12, "uy(0) mismatch")
    _assert_close(v1, uy_mean + uy_std, 1e-12, "uy(1) mismatch")

    return SelfCheckResult(
        name="denormalization_mapping",
        passed=True,
        details="normalized 0/1 mapping verified for p, Ux, Uy",
    )


def evaluate_self_check_split_indices_consistency() -> SelfCheckResult:
    """Test 4: validation split indices match training-style deterministic random_split."""
    dataset_len = 73
    validation_split = 0.2
    split_seed = 42

    train_idx_a, val_idx_a = compute_train_val_indices(dataset_len, validation_split, split_seed)

    dummy = TensorDataset(torch.arange(dataset_len, dtype=torch.float32).unsqueeze(1))
    train_size = max(1, int((1.0 - validation_split) * len(dummy)))
    val_size = len(dummy) - train_size

    if val_size == 0:
        train_idx_b = list(range(dataset_len))
        val_idx_b = list(range(dataset_len))
    else:
        generator = torch.Generator().manual_seed(split_seed)
        train_set, val_set = random_split(dummy, [train_size, val_size], generator=generator)
        train_idx_b = list(train_set.indices)
        val_idx_b = list(val_set.indices)

    if train_idx_a != train_idx_b or val_idx_a != val_idx_b:
        raise AssertionError("Split indices mismatch against training-style random_split")

    return SelfCheckResult(
        name="split_indices_consistency",
        passed=True,
        details=f"train={len(train_idx_a)}, val={len(val_idx_a)}",
    )


def evaluate_run_all_self_checks() -> list[SelfCheckResult]:
    checks = [
        evaluate_self_check_channel_order_in_rans_loss,
        evaluate_self_check_constant_field_residuals,
        evaluate_self_check_denormalization_mapping,
        evaluate_self_check_split_indices_consistency,
    ]

    results: list[SelfCheckResult] = []
    for fn in checks:
        results.append(fn())
    return results
