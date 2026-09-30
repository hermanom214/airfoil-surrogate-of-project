from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.fix_force_coeff_reference_area import (
    BACKUP_SUFFIX,
    DATA_RELPATH,
    METADATA_NAME,
    MigrationError,
    correction_factor,
    migrate,
    transform_force_coeffs,
)
from src.file_editors import force_reference_area, update_force_coeffs_file
from src.ml_data_split import dataset_fingerprint, load_or_create_fixed_test_split


HEADER = """# Force coefficients
# Time          \tCm              \tCd              \tCl              \tCl(f)           \tCl(r)           
"""


def _case(root: Path, name: str = "case_0001_naca0012_aoa0p0_u20p0") -> Path:
    case = root / name
    data = case / DATA_RELPATH
    data.parent.mkdir(parents=True)
    data.write_text(
        HEADER
        + "0               \t1.00000000e-02\t2.00000000e-03\t3.00000000e-02\t4.00000000e-02\t-1.00000000e-02\n",
        encoding="utf-8",
    )
    system = case / "system"
    system.mkdir()
    (system / "forceCoeffs").write_text(
        "magUInf 20;\nlRef 1.0;\nAref 1.0;\n", encoding="utf-8"
    )
    (case / "params.json").write_text(json.dumps({"chord": 1.0}), encoding="utf-8")
    return case


def test_reference_area_uses_chord_and_full_span() -> None:
    assert force_reference_area(1.0, 0.05) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        force_reference_area(0.0, 0.05)


def test_generated_force_coeffs_receives_geometry_references(tmp_path: Path) -> None:
    path = tmp_path / "forceCoeffs"
    path.write_text("magUInf 15;\nlRef 9;\nAref 9;\n", encoding="utf-8")
    update_force_coeffs_file(path, 20.0, chord=1.2, z_half=0.05)
    text = path.read_text(encoding="utf-8")
    assert "magUInf         20.000000;" in text
    assert "lRef            1.2;" in text
    assert "Aref            0.12;" in text


def test_transform_scales_only_coefficient_columns_and_preserves_header() -> None:
    original = (
        HEADER
        + "100\t1.00000000e-02\t2.00000000e-03\t3.00000000e-02\t4.00000000e-02\t-1.00000000e-02\n"
    ).encode()
    transformed = transform_force_coeffs(original, correction_factor(1.0, 0.1)).decode()
    lines = transformed.splitlines()
    assert lines[:2] == original.decode().splitlines()[:2]
    fields = lines[2].split()
    assert fields[0] == "100"
    assert np.allclose([float(value) for value in fields[1:]], [0.1, 0.02, 0.3, 0.4, -0.1])


def test_dry_run_is_zero_write_and_apply_is_idempotent(tmp_path: Path) -> None:
    case = _case(tmp_path)
    data = case / DATA_RELPATH
    before = data.read_bytes()
    dry = migrate(tmp_path, apply=False)
    assert dry.eligible == 1 and dry.migrated == 0
    assert data.read_bytes() == before
    assert not data.with_name(data.name + BACKUP_SUFFIX).exists()

    first = migrate(tmp_path, apply=True)
    after_first = data.read_bytes()
    assert first.migrated == 1
    assert data.with_name(data.name + BACKUP_SUFFIX).read_bytes() == before
    assert data.with_name(METADATA_NAME).is_file()
    params = json.loads((case / "params.json").read_text())
    assert params["Aref"] == pytest.approx(0.1)
    assert params["span"] == pytest.approx(0.1)

    second = migrate(tmp_path, apply=True)
    assert second.skipped == 1 and second.migrated == 0
    assert data.read_bytes() == after_first


def test_unexpected_header_fails_before_any_write(tmp_path: Path) -> None:
    case = _case(tmp_path)
    data = case / DATA_RELPATH
    data.write_text("# Time Cd Cl\n0 0.1 0.2\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="preflight failed"):
        migrate(tmp_path, apply=True)
    assert not data.with_name(data.name + BACKUP_SUFFIX).exists()


def test_membership_fingerprint_and_fixed_test_ignore_target_values(tmp_path: Path) -> None:
    ids = ["case_a", "case_b", "case_c"]
    before_targets = np.array([[0.01, 0.001], [0.02, 0.002], [0.03, 0.003]])
    after_targets = before_targets * 10.0
    split_path = tmp_path / "fixed_test_split.json"
    before_split = load_or_create_fixed_test_split(ids, split_path, 1 / 3, 42)
    assert not np.array_equal(before_targets, after_targets)
    assert dataset_fingerprint(ids) == dataset_fingerprint(ids)
    after_split = load_or_create_fixed_test_split(ids, split_path, 1 / 3, 42)
    assert after_split.test_case_ids == before_split.test_case_ids


def test_corrected_naca0012_lift_curve_slope_is_physical() -> None:
    alpha_deg = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    old_cl = np.array([-0.0424, -0.0214, 0.0, 0.0215, 0.0420])
    corrected_cl = old_cl * correction_factor(1.0, 0.1)
    slope_per_rad = np.polyfit(np.deg2rad(alpha_deg), corrected_cl, 1)[0]
    assert 4.5 < slope_per_rad < 7.5
    assert not 0.45 < slope_per_rad < 0.75
