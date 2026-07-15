from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Heuristics for detecting unrealistically constant fields in the central flow region.
PLAUSIBILITY_REL_THRESHOLD = 0.03
PLAUSIBILITY_P_ABS_THRESHOLD = 0.5
PLAUSIBILITY_U_ABS_THRESHOLD = 0.15


@dataclass(frozen=True)
class PlausibilityResult:
    case_name: str
    plausibility: str
    reason: str
    p_min: float
    p_max: float
    p_median: float
    u_min: float
    u_max: float
    u_median: float


def validate_shapes(xy: np.ndarray, p: np.ndarray, u: np.ndarray, fluid_mask: np.ndarray) -> None:
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise RuntimeError(f"Invalid xy shape: {xy.shape}")

    if p.ndim != 2:
        raise RuntimeError(f"Invalid p shape: {p.shape}")

    if u.ndim != 3 or u.shape[2] != 2:
        raise RuntimeError(f"Invalid U shape: {u.shape}")

    if fluid_mask.ndim != 2:
        raise RuntimeError(f"Invalid fluid_mask shape: {fluid_mask.shape}")

    ny, nx = p.shape

    if xy.shape[:2] != (ny, nx):
        raise RuntimeError("xy and p shape mismatch")

    if u.shape[:2] != (ny, nx):
        raise RuntimeError("U and p shape mismatch")

    if fluid_mask.shape != (ny, nx):
        raise RuntimeError("fluid_mask and p shape mismatch")


def _compute_scalar_stats(values: np.ndarray) -> tuple[float, float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.min(finite_values)),
        float(np.max(finite_values)),
        float(np.median(finite_values)),
    )


def _compute_central_plausibility(
    p: np.ndarray,
    u_mag: np.ndarray,
    fluid_binary: np.ndarray,
) -> tuple[str, str]:
    ny, nx = p.shape
    y0, y1 = int(0.25 * ny), int(0.75 * ny)
    x0, x1 = int(0.25 * nx), int(0.75 * nx)

    roi_fluid = fluid_binary[y0:y1, x0:x1]
    roi_p = p[y0:y1, x0:x1][roi_fluid]
    roi_u = u_mag[y0:y1, x0:x1][roi_fluid]
    global_p = p[fluid_binary]
    global_u = u_mag[fluid_binary]

    if roi_p.size < 200 or roi_u.size < 200:
        return "nOK", "too_few_fluid_points_in_central_roi"

    roi_p_span = float(np.percentile(roi_p, 95.0) - np.percentile(roi_p, 5.0))
    roi_u_span = float(np.percentile(roi_u, 95.0) - np.percentile(roi_u, 5.0))
    global_p_span = float(np.percentile(global_p, 99.0) - np.percentile(global_p, 1.0))
    global_u_span = float(np.percentile(global_u, 99.0) - np.percentile(global_u, 1.0))

    p_threshold = max(
        PLAUSIBILITY_P_ABS_THRESHOLD,
        PLAUSIBILITY_REL_THRESHOLD * max(global_p_span, 1e-12),
    )
    u_threshold = max(
        PLAUSIBILITY_U_ABS_THRESHOLD,
        PLAUSIBILITY_REL_THRESHOLD * max(global_u_span, 1e-12),
    )

    p_constant = roi_p_span <= p_threshold
    u_constant = roi_u_span <= u_threshold
    if p_constant and u_constant:
        return (
            "nOK",
            (
                f"constant_central_region:p_span={roi_p_span:.6g}<=thr={p_threshold:.6g};"
                f"u_span={roi_u_span:.6g}<=thr={u_threshold:.6g}"
            ),
        )

    return "OK", ""


def evaluate_plausibility(
    *,
    xy: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    fluid_mask: np.ndarray,
    case_name: str,
) -> PlausibilityResult:
    validate_shapes(xy, p, u, fluid_mask)

    fluid_binary = fluid_mask > 0.5
    u_mag = np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)

    p_min, p_max, p_median = _compute_scalar_stats(p[fluid_binary])
    u_min, u_max, u_median = _compute_scalar_stats(u_mag[fluid_binary])
    plausibility, reason = _compute_central_plausibility(p, u_mag, fluid_binary)

    return PlausibilityResult(
        case_name=case_name,
        plausibility=plausibility,
        reason=reason,
        p_min=p_min,
        p_max=p_max,
        p_median=p_median,
        u_min=u_min,
        u_max=u_max,
        u_median=u_median,
    )


def evaluate_plausibility_from_npz(file_path: Path) -> PlausibilityResult:
    with np.load(file_path, allow_pickle=True) as data:
        required = ["xy", "p", "U", "fluid_mask"]
        for key in required:
            if key not in data:
                raise RuntimeError(f"Missing key '{key}' in {file_path}")

        return evaluate_plausibility(
            xy=data["xy"],
            p=data["p"],
            u=data["U"],
            fluid_mask=data["fluid_mask"],
            case_name=file_path.name,
        )


def write_plausibility_csv(rows: list[PlausibilityResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_name",
        "plausibility",
        "reason",
        "p_min",
        "p_max",
        "p_median",
        "u_min",
        "u_max",
        "u_median",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "case_name": row.case_name,
                    "plausibility": row.plausibility,
                    "reason": row.reason,
                    "p_min": f"{row.p_min:.10g}",
                    "p_max": f"{row.p_max:.10g}",
                    "p_median": f"{row.p_median:.10g}",
                    "u_min": f"{row.u_min:.10g}",
                    "u_max": f"{row.u_max:.10g}",
                    "u_median": f"{row.u_median:.10g}",
                }
            )
