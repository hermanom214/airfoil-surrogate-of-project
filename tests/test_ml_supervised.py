from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.ml_dataset import AirfoilFlowDataset
from src.ml_supervised import (
    NONDIMENSIONAL_TARGETS,
    RAW_TARGETS,
    body_coordinates,
    build_region_weights,
    gradient_matching_loss,
    normalize_physical_targets,
    physical_to_targets,
    supervised_config_from_checkpoint,
    targets_to_physical,
    wake_region_mask,
)


def test_nondimensional_target_round_trip_and_components() -> None:
    p = np.array([[0.0, 50.0]], dtype=np.float32)
    ux = np.array([[20.0, 10.0]], dtype=np.float32)
    uy = np.array([[2.0, -4.0]], dtype=np.float32)
    target = physical_to_targets(p, ux, uy, 20.0, NONDIMENSIONAL_TARGETS)
    np.testing.assert_allclose(target[0], [[0.0, 0.25]])
    np.testing.assert_allclose(target[1], [[0.0, -0.5]])
    np.testing.assert_allclose(target[2], [[0.1, -0.2]])
    decoded = targets_to_physical(target, 20.0, NONDIMENSIONAL_TARGETS)
    np.testing.assert_allclose(decoded, (p, ux, uy))


def test_raw_representation_is_legacy_identity() -> None:
    arrays = tuple(np.arange(6, dtype=np.float32).reshape(2, 3) + i for i in range(3))
    target = physical_to_targets(*arrays, 15.0, RAW_TARGETS)
    decoded = targets_to_physical(target, 15.0, RAW_TARGETS)
    np.testing.assert_allclose(decoded, arrays)


def test_checkpoint_without_target_metadata_uses_legacy_raw_mode() -> None:
    config = supervised_config_from_checkpoint({"model_state_dict": {}})
    assert config.target_representation == RAW_TARGETS
    assert config.wall_weight == 0.0
    assert config.wake_weight == 0.0
    assert config.gradient_loss_weight == 0.0


def test_common_normalization_is_independent_of_prediction_representation() -> None:
    p = np.array([[10.0, -5.0]], dtype=np.float32)
    ux = np.array([[20.0, 18.0]], dtype=np.float32)
    uy = np.array([[1.0, -1.0]], dtype=np.float32)
    stats = {
        "p_mean": 2.0, "p_std": 4.0,
        "ux_mean": 19.0, "ux_std": 2.0,
        "uy_mean": 0.0, "uy_std": 0.5,
    }
    raw = targets_to_physical(
        physical_to_targets(p, ux, uy, 20.0, RAW_TARGETS), 20.0, RAW_TARGETS,
    )
    nondimensional = targets_to_physical(
        physical_to_targets(p, ux, uy, 20.0, NONDIMENSIONAL_TARGETS),
        20.0,
        NONDIMENSIONAL_TARGETS,
    )
    np.testing.assert_allclose(
        normalize_physical_targets(*raw, stats),
        normalize_physical_targets(*nondimensional, stats),
    )


def test_common_raw_normalization_uses_only_requested_development_indices(
    tmp_path: Path,
) -> None:
    fluid = np.ones((2, 2), dtype=np.float32)
    paths = []
    for name, p_value, u_value in (("development", 2.0, 4.0), ("held_out", 1000.0, 800.0)):
        path = tmp_path / f"{name}.npz"
        np.savez(
            path,
            p=np.full((2, 2), p_value, dtype=np.float32),
            U=np.full((2, 2, 2), u_value, dtype=np.float32),
            fluid_mask=fluid,
        )
        paths.append(path)
    dataset = AirfoilFlowDataset.__new__(AirfoilFlowDataset)
    dataset.files = paths
    stats = dataset.compute_raw_target_normalization([0])
    assert stats["p_mean"] == 2.0
    assert stats["ux_mean"] == 4.0
    assert stats["uy_mean"] == 4.0


def test_wake_mask_orientation_zero_and_nonzero_aoa() -> None:
    xb = np.array([[[1.0, 0.0], [1.4, 0.1], [0.5, 0.0]]])
    mask = np.ones((1, 3), dtype=bool)
    wake0 = wake_region_mask(xb, mask, 1.0, 0.0, 0.9, 1.75, 0.2)
    assert wake0.tolist() == [[True, True, False]]

    theta = np.deg2rad(-20.0)
    x, y = xb[..., 0] - 0.25, xb[..., 1]
    rotated = np.stack((0.25 + x * np.cos(theta) - y * np.sin(theta),
                        x * np.sin(theta) + y * np.cos(theta)), axis=-1)
    wake_rotated = wake_region_mask(rotated, mask, 1.0, 20.0, 0.9, 1.75, 0.2)
    np.testing.assert_array_equal(wake_rotated, wake0)
    recovered = body_coordinates(rotated, 1.0, 20.0)
    np.testing.assert_allclose(recovered, (xb[..., 0], xb[..., 1]), atol=1e-7)


def test_region_weights_and_solid_exclusion() -> None:
    fluid = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    wall = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    wake = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    weights = build_region_weights(fluid, wall, wake, 3.0, 2.0)
    torch.testing.assert_close(weights, torch.tensor([[[[4.0, 3.0], [0.0, 1.0]]]]))


def test_gradient_loss_zero_identical_positive_for_perturbation() -> None:
    target = torch.zeros((1, 3, 7, 7))
    mask = torch.ones((1, 1, 7, 7))
    assert gradient_matching_loss(target, target, mask, 0.5, 0.25).item() == 0.0
    pred = target.clone()
    pred[:, 0, 3, 3] = 1.0
    assert gradient_matching_loss(pred, target, mask, 0.5, 0.25).item() > 0.0


def test_gradient_stencil_does_not_cross_solid() -> None:
    target = torch.zeros((1, 3, 7, 7))
    pred = target.clone()
    pred[:, :, 3, 3] = 100.0
    mask = torch.ones((1, 1, 7, 7))
    mask[:, :, 3, 3] = 0.0
    assert gradient_matching_loss(pred, target, mask, 1.0, 1.0).item() == 0.0
