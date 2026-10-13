from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.file_editors import force_reference_area, update_force_coeffs_file
from src.ml_data_split import dataset_fingerprint, load_or_create_fixed_test_split

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