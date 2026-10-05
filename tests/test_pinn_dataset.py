from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.pinn_dataset import (
    airfoil_sdf_from_geometry,
    build_physics_valid_mask,
    parse_openfoam_uniform_nu,
    validate_arrays,
)
from scripts.extract_pinn_dataset import clean_generated_output, inspection_candidates, verify_final_artifacts
from src.ml_quality_filter import load_inspection_membership
from src.pinn_dataset import CaseResult


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


def _write_inspection(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_name", "overall_status", "overall_reason"])
        writer.writeheader()
        for case_id, status in rows:
            writer.writerow({"case_name": case_id, "overall_status": status, "overall_reason": status})


def test_pinn_candidates_come_from_inspection_not_split(tmp_path: Path) -> None:
    source = tmp_path / "flow_fields"
    for case_id in ("case_a", "case_b"):
        (source / case_id / "VTK").mkdir(parents=True)
        (source / case_id / "system").mkdir()
    _write_inspection(tmp_path / "pictures_inspect_flow" / "inspect_plausibility.csv", [
        ("case_a", "OK"), ("case_b", "nOK")
    ])
    archived, membership = inspection_candidates(source)
    assert archived == ["case_a", "case_b"]
    assert membership.ok_case_ids == ["case_a"]


def test_missing_inspection_csv_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="refusing to construct an unfiltered"):
        load_inspection_membership(tmp_path / "flow_fields")


def test_malformed_inspection_csv_fails(tmp_path: Path) -> None:
    path = tmp_path / "pictures_inspect_flow" / "inspect_plausibility.csv"
    path.parent.mkdir(parents=True)
    path.write_text("case_name,overall_status\ncase_a,BAD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_inspection_membership(tmp_path / "flow_fields")


def test_duplicate_inspection_cases_fail(tmp_path: Path) -> None:
    _write_inspection(
        tmp_path / "pictures_inspect_flow" / "inspect_plausibility.csv",
        [("case_a", "OK"), ("case_a", "nOK")],
    )
    with pytest.raises(ValueError, match="Duplicate inspection record"):
        load_inspection_membership(tmp_path / "flow_fields")


def test_inspection_membership_fingerprint_is_order_independent(tmp_path: Path) -> None:
    path = tmp_path / "pictures_inspect_flow" / "inspect_plausibility.csv"
    _write_inspection(path, [("case_b", "nOK"), ("case_a", "OK")])
    first = load_inspection_membership(tmp_path / "flow_fields")
    path.unlink()
    _write_inspection(path, [("case_a", "OK"), ("case_b", "nOK")])
    second = load_inspection_membership(tmp_path / "flow_fields")
    assert first.ok_case_ids == ["case_a"]
    assert first.nok_case_ids == ["case_b"]
    assert first.membership_fingerprint == second.membership_fingerprint


def test_clean_rebuild_rejects_unexpected_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="may only target"):
        clean_generated_output(tmp_path / "wrong", tmp_path / "expected")


def test_final_artifacts_reject_stale_case(tmp_path: Path) -> None:
    stale = tmp_path / "cases" / "case_stale"
    stale.mkdir(parents=True)
    np.savez(stale / "case_stale.npz", x=np.ones(1))
    with pytest.raises(RuntimeError, match="stale_or_unknown"):
        verify_final_artifacts(tmp_path, ["case_a"], [CaseResult("case_a", "invalid")])


def test_output_guard_targets_separate_directory() -> None:
    source = Path("data/flow_fields").resolve()
    output = Path("data/pinn_dataset").resolve()
    assert source != output
    assert source not in output.parents
