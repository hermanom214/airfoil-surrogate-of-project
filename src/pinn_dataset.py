from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion
from shapely import contains_xy, distance, points

from src.flow_extractor import generate_naca4_polygon


GRID_SHAPE = (320, 640)
REQUIRED_FIELDS = ("p", "U", "k", "omega", "nut")
FIELD_UNITS = {
    "xy": "m",
    "p": "m^2/s^2 (kinematic pressure)",
    "U": "m/s",
    "k": "m^2/s^2",
    "omega": "1/s",
    "nut": "m^2/s",
    "airfoil_sdf": "m; positive in fluid, negative in solid",
    "nu": "m^2/s",
    "Re": "dimensionless",
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    output_file: str = ""
    message: str = ""


def parse_openfoam_uniform_nu(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\bnu\s+\[\s*0\s+2\s+-1\s+0\s+0\s+0\s+0\s*\]\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;",
        text,
    )
    if not match:
        raise ValueError(f"Uniform kinematic viscosity 'nu' not found in {path}")
    value = float(match.group(1))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid molecular kinematic viscosity in {path}: {value}")
    return value


def read_openfoam_binary_internal_field(path: Path, components: int = 1) -> np.ndarray:
    """Read a little-endian OpenFOAM binary nonuniform scalar/vector internal field."""
    raw = path.read_bytes()
    header_end = raw.find(b"internalField")
    if header_end < 0:
        raise ValueError(f"internalField not found in {path}")
    header = raw[: header_end + 512].decode("ascii", errors="ignore")
    expected_type = b"List<scalar>" if components == 1 else b"List<vector>"
    type_at = raw.find(expected_type, header_end)
    if type_at < 0:
        raise ValueError(f"{expected_type.decode()} internal field not found in {path}")
    count_match = re.match(rb"\s*(\d+)\s*\(\s*", raw[type_at + len(expected_type) :])
    if count_match is None:
        raise ValueError(f"Cannot parse internal-field count in {path}")
    count = int(count_match.group(1))
    data_start = type_at + len(expected_type) + count_match.end()
    n_values = count * components
    n_bytes = n_values * np.dtype("<f8").itemsize
    if data_start + n_bytes > len(raw):
        raise ValueError(f"Truncated binary internal field in {path}")
    values = np.frombuffer(raw, dtype="<f8", count=n_values, offset=data_start).copy()
    if components > 1:
        values = values.reshape(count, components)
    return values


def airfoil_sdf_from_geometry(
    xy: np.ndarray, naca_code: str, chord: float, aoa_deg: float
) -> np.ndarray:
    polygon = generate_naca4_polygon(naca_code, chord=chord, aoa_deg=aoa_deg)
    x = np.asarray(xy[..., 0], dtype=np.float64)
    y = np.asarray(xy[..., 1], dtype=np.float64)
    grid_points = points(x.ravel(), y.ravel())
    unsigned = np.asarray(distance(grid_points, polygon.boundary)).reshape(x.shape)
    inside = np.asarray(contains_xy(polygon, x, y), dtype=bool)
    signed = np.where(inside, -unsigned, unsigned)
    return signed.astype(np.float32)


def build_physics_valid_mask(
    fluid_mask: np.ndarray,
    interpolation_valid_mask: np.ndarray,
    stencil_radius: int = 2,
) -> np.ndarray:
    if fluid_mask.shape != interpolation_valid_mask.shape:
        raise ValueError("fluid and interpolation masks must have identical shapes")
    if stencil_radius < 0:
        raise ValueError("stencil_radius must be non-negative")
    base = (fluid_mask > 0.5) & (interpolation_valid_mask > 0.5)
    if stencil_radius == 0:
        return base.astype(np.uint8)
    structure = np.ones((3, 3), dtype=bool)
    return binary_erosion(base, structure=structure, iterations=stencil_radius, border_value=0).astype(
        np.uint8
    )


def validate_arrays(arrays: dict[str, np.ndarray]) -> list[str]:
    issues: list[str] = []
    expected_2d = arrays["fluid_mask"].shape
    for key in ("xy", "U"):
        if arrays[key].shape != (*expected_2d, 2):
            issues.append(f"{key}: unexpected shape {arrays[key].shape}")
    for key in ("p", "k", "omega", "nut", "airfoil_sdf", "interpolation_valid_mask", "physics_valid_mask"):
        if arrays[key].shape != expected_2d:
            issues.append(f"{key}: unexpected shape {arrays[key].shape}")
    for key in ("xy", "p", "U", "k", "omega", "nut", "airfoil_sdf"):
        if not np.all(np.isfinite(arrays[key])):
            issues.append(f"{key}: contains NaN or Inf")
    fluid = arrays["fluid_mask"] > 0.5
    if np.any(arrays["k"][fluid] < 0.0):
        issues.append("k: negative values in fluid")
    if np.any(arrays["omega"][fluid] <= 0.0):
        issues.append("omega: non-positive values in fluid")
    if np.any(arrays["nut"][fluid] < 0.0):
        issues.append("nut: negative values in fluid")
    sdf_fluid_mismatch = np.count_nonzero((arrays["airfoil_sdf"] > 0.0) != fluid)
    if sdf_fluid_mismatch:
        issues.append(f"airfoil_sdf/fluid_mask sign mismatch at {sdf_fluid_mismatch} cells")
    if np.any((arrays["physics_valid_mask"] > 0) & ~fluid):
        issues.append("physics_valid_mask includes solid cells")
    if np.any((arrays["physics_valid_mask"] > 0) & ~(arrays["interpolation_valid_mask"] > 0)):
        issues.append("physics_valid_mask includes interpolation-invalid cells")
    return issues


def summary_for_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    fluid = arrays["fluid_mask"] > 0.5
    selected = {
        "p": arrays["p"][fluid],
        "Ux": arrays["U"][..., 0][fluid],
        "Uy": arrays["U"][..., 1][fluid],
        "k": arrays["k"][fluid],
        "omega": arrays["omega"][fluid],
        "nut": arrays["nut"][fluid],
        "airfoil_sdf": arrays["airfoil_sdf"],
    }
    return {
        key: {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values, dtype=np.float64)),
        }
        for key, values in selected.items()
    }


def save_case_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing PINN dataset artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, sort_keys=True))


def verify_split_mapping(case_ids: Iterable[str], split_path: Path) -> dict[str, object]:
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    available = set(case_ids)
    dataset_ids = set(str(v) for v in raw["dataset_case_ids"])
    return {
        "development_count": len(raw["development_case_ids"]),
        "test_count": len(raw["test_case_ids"]),
        "missing_from_new_dataset": sorted(dataset_ids - available),
        "extra_in_new_dataset": sorted(available - dataset_ids),
        "all_mapped": dataset_ids == available,
    }


def case_result_dict(result: CaseResult) -> dict[str, str]:
    return asdict(result)
