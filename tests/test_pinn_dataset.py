from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pinn_dataset import (
    airfoil_sdf_from_geometry,
    build_physics_valid_mask,
    parse_openfoam_uniform_nu,
    validate_arrays,
    verify_split_mapping,
)


def test_sdf_sign_and_rotation() -> None:
    xs = np.linspace(-0.2, 1.2, 141)
    ys = np.linspace(-0.3, 0.3, 121)
    x, y = np.meshgrid(xs, ys)
    xy = np.stack([x, y], axis=-1)
    sdf_zero = airfoil_sdf_from_geometry(xy, "0012", 1.0, 0.0)
    sdf_rotated = airfoil_sdf_from_geometry(xy, "0012", 1.0, 8.0)
    assert sdf_zero[ys.searchsorted(0.0), xs.searchsorted(0.5)] < 0.0
    assert sdf_zero[0, 0] > 0.0
    assert not np.array_equal(sdf_zero < 0.0, sdf_rotated < 0.0)
    assert float(np.min(np.abs(sdf_zero))) < 0.01


def test_physics_mask_respects_two_pixel_stencil() -> None:
    fluid = np.ones((9, 9), dtype=np.uint8)
    fluid[4, 4] = 0
    valid = np.ones_like(fluid)
    result = build_physics_valid_mask(fluid, valid, stencil_radius=2)
    assert not result[2:7, 2:7].any()
    assert not result[[0, 1, -2, -1], :].any()
    assert not result[:, [0, 1, -2, -1]].any()


def test_viscosity_and_reynolds_number(tmp_path: Path) -> None:
    transport = tmp_path / "transportProperties"
    transport.write_text("nu [0 2 -1 0 0 0 0] 1.5e-05;", encoding="utf-8")
    nu = parse_openfoam_uniform_nu(transport)
    assert nu == pytest.approx(1.5e-5)
    assert 20.0 * 1.0 / nu == pytest.approx(1_333_333.3333333333)


def _valid_arrays() -> dict[str, np.ndarray]:
    shape = (5, 7)
    return {
        "xy": np.zeros((*shape, 2), dtype=np.float32),
        "U": np.zeros((*shape, 2), dtype=np.float32),
        "p": np.zeros(shape, dtype=np.float32),
        "k": np.ones(shape, dtype=np.float32),
        "omega": np.ones(shape, dtype=np.float32),
        "nut": np.ones(shape, dtype=np.float32),
        "fluid_mask": np.ones(shape, dtype=np.uint8),
        "airfoil_sdf": np.ones(shape, dtype=np.float32),
        "interpolation_valid_mask": np.ones(shape, dtype=np.uint8),
        "physics_valid_mask": np.ones(shape, dtype=np.uint8),
    }


def test_invalid_turbulence_values_are_reported_not_clipped() -> None:
    arrays = _valid_arrays()
    arrays["k"][0, 0] = -1.0
    arrays["omega"][0, 1] = 0.0
    arrays["nut"][0, 2] = -1.0
    issues = validate_arrays(arrays)
    assert any("k: negative" in issue for issue in issues)
    assert any("omega: non-positive" in issue for issue in issues)
    assert any("nut: negative" in issue for issue in issues)
    assert arrays["k"][0, 0] == -1.0


def test_split_mapping(tmp_path: Path) -> None:
    split = {
        "dataset_case_ids": ["a", "b", "c"],
        "development_case_ids": ["a", "b"],
        "test_case_ids": ["c"],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    mapping = verify_split_mapping(["a", "b", "c"], path)
    assert mapping["all_mapped"] is True
    assert mapping["development_count"] == 2
    assert mapping["test_count"] == 1


def test_output_guard_targets_separate_directory() -> None:
    source = Path("data/flow_fields").resolve()
    output = Path("data/pinn_dataset").resolve()
    assert source != output
    assert source not in output.parents
