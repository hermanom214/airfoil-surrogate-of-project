"""Reusable, evaluation-only helpers for the Phase A U-Net field comparison."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src.ml_supervised import (
    NONDIMENSIONAL_TARGETS, body_coordinates, supervised_config_from_checkpoint,
    targets_to_physical,
)


PRESSURE_UNIT = "m^2/s^2 (OpenFOAM kinematic pressure)"
VELOCITY_UNIT = "m/s"
NORMALIZATION_KEYS = (
    "aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std",
    "p_mean", "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_normalization(checkpoint: Mapping[str, object]) -> dict[str, float]:
    missing = [key for key in NORMALIZATION_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint lacks training normalization: {missing}")
    result = {key: float(checkpoint[key]) for key in NORMALIZATION_KEYS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Checkpoint normalization contains non-finite values")
    if any(result[key] <= 0 for key in ("aoa_std", "inlet_u_std", "p_std", "ux_std", "uy_std")):
        raise ValueError("Checkpoint normalization standard deviations must be positive")
    return result


def assert_matching_normalization(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, float]:
    a, b = checkpoint_normalization(left), checkpoint_normalization(right)
    if a != b:
        raise ValueError("A/B checkpoints use different training normalization")
    if supervised_config_from_checkpoint(left) != supervised_config_from_checkpoint(right):
        raise ValueError("A/B checkpoints use different target/preprocessing configuration")
    return a


def denormalize_prediction(prediction: np.ndarray, normalization: Mapping[str, float]) -> dict[str, np.ndarray]:
    values = np.asarray(prediction)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("Prediction must have shape [3, height, width]")
    return {
        "p": values[0] * normalization["p_std"] + normalization["p_mean"],
        "Ux": values[1] * normalization["ux_std"] + normalization["ux_mean"],
        "Uy": values[2] * normalization["uy_std"] + normalization["uy_mean"],
    }


def decode_checkpoint_prediction(prediction, checkpoint, inlet_velocity):
    """Decode using saved metadata and per-case U_inf, never directory names."""
    config = supervised_config_from_checkpoint(checkpoint)
    norm = checkpoint_normalization(checkpoint)
    if config.target_representation == NONDIMENSIONAL_TARGETS:
        # Original improved training uses no further target standardization.
        if any(norm[f"{c}_mean"] != 0 or norm[f"{c}_std"] != 1 for c in ("p", "ux", "uy")):
            raise ValueError("Improved checkpoint has unsupported additional target normalization")
        values = targets_to_physical(prediction, inlet_velocity, config.target_representation, config.p_inf)
        return dict(zip(("p", "Ux", "Uy"), values))
    return denormalize_prediction(prediction, norm)


def summarize_field_metrics(rows):
    """Case-level summaries; each case contributes once to its region/field."""
    summary = []
    for region, variable in sorted({(r["region"], r["variable"]) for r in rows}):
        subset = [r for r in rows if r["region"] == region and r["variable"] == variable]
        for metric in ("MAE", "RMSE"):
            a = np.asarray([r[f"supervised_{metric}"] for r in subset], dtype=float)
            b = np.asarray([r[f"physics_{metric}"] for r in subset], dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("Non-finite field metrics")
            deltas = b - a
            summary.append(dict(region=region, variable=variable, metric=metric, case_count=len(subset),
                physics_improved_cases=int((deltas < 0).sum()), physics_worsened_cases=int((deltas > 0).sum()),
                ties=int((deltas == 0).sum()), mean_B_minus_A=float(deltas.mean()),
                median_B_minus_A=float(np.median(deltas)), mean_supervised_error=float(a.mean()),
                mean_physics_error=float(b.mean())))
    return summary


def field_interpretation(summary):
    """Describe actual counts, including ties, without a fixed performance claim."""
    lines = []
    for s in summary:
        lines.append(f"{s['region']} {s['variable']} {s['metric']}: physics-informed error improved in "
            f"{s['physics_improved_cases']} of {s['case_count']} selected cases, worsened in "
            f"{s['physics_worsened_cases']}, and tied in {s['ties']}; mean B-A "
            f"{s['mean_B_minus_A']:.6g}, median B-A {s['median_B_minus_A']:.6g}.")
    return lines


def case_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        meta["inspection_ok"] = not meta.get("validation_issues")
        meta["pinn_valid"] = bool(np.any(data["physics_valid_mask"] > 0.5))
    meta["path"] = str(path)
    return meta


def select_representative_cases(cases: Iterable[Mapping[str, object]], count: int = 8) -> list[dict[str, object]]:
    """Greedy metadata-only selection with categorical then maximin coverage."""
    eligible = [dict(c) for c in cases if c.get("inspection_ok") and c.get("pinn_valid")]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible fixed-test cases for requested {count}")
    eligible.sort(key=lambda c: str(c["case_id"]))
    features = np.asarray([
        [float(c["camber_percent"]), float(c["thickness_percent"]), float(c["aoa_deg"]), float(c["U_inf"])]
        for c in eligible
    ])
    lo, hi = features.min(0), features.max(0)
    scaled = (features - lo) / np.where(hi > lo, hi - lo, 1.0)
    def categories(row: np.ndarray) -> set[str]:
        camber, thickness, aoa, speed = row
        return {
            "symmetric" if camber == 0 else "cambered",
            "aoa_negative" if aoa < 0 else ("aoa_positive" if aoa > 0 else "aoa_zero"),
            *( ["speed_low"] if speed == lo[3] else []),
            *( ["speed_high"] if speed == hi[3] else []),
            *( ["thickness_low"] if thickness == lo[1] else []),
            *( ["thickness_high"] if thickness == hi[1] else []),
        }
    required = {"symmetric", "cambered", "aoa_negative", "aoa_zero", "aoa_positive",
                "speed_low", "speed_high", "thickness_low", "thickness_high"}
    # Begin at the smallest parameter-space corner. Prefer unmet categorical
    # requirements, then maximize distance; ties follow sorted case_id order.
    selected = [int(np.argmin(np.sum(scaled, axis=1)))]
    while len(selected) < count:
        distances = np.min(np.linalg.norm(scaled[:, None, :] - scaled[selected][None, :, :], axis=2), axis=1)
        distances[selected] = -1.0
        covered = set().union(*(categories(features[i]) for i in selected))
        best = max(range(len(eligible)), key=lambda i: (
            len(categories(features[i]) & (required - covered)),
            round(float(distances[i]), 12), -i,
        ) if i not in selected else (-1, -1.0, -i))
        selected.append(best)
    covered = set().union(*(categories(features[i]) for i in selected))
    available = set().union(*(categories(row) for row in features))
    missing = (required & available) - covered
    if missing:
        raise ValueError(f"Representative selection failed available coverage categories: {sorted(missing)}")
    return [eligible[i] for i in selected]


def region_masks(xy: np.ndarray, valid_mask: np.ndarray, chord: float, aoa_deg: float) -> dict[str, np.ndarray]:
    """Physical ROIs in chord-aligned coordinates; never pixel-index based."""
    valid = np.asarray(valid_mask, dtype=bool)
    xb, yb = body_coordinates(xy, chord, aoa_deg)
    xc, yc = xb / chord, yb / chord
    near = valid & (xc >= -0.20) & (xc <= 1.20) & (np.abs(yc) <= 0.30)
    wake = valid & (xc >= 0.90) & (xc <= 1.75) & (np.abs(yc) <= 0.20)
    return {"full_domain": valid, "near_airfoil": near, "wake": wake, "outer_flow": valid & ~near & ~wake}


def masked_metrics(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("Metric mask contains no finite valid cells")
    error = np.asarray(prediction)[valid] - np.asarray(reference)[valid]
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error))))


def shared_value_limits(reference: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(reference)[np.asarray(mask, dtype=bool) & np.isfinite(reference)]
    if not values.size:
        raise ValueError("Cannot determine limits from an empty reference mask")
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not hi > lo:
        lo, hi = float(values.min()), float(values.max())
    if not hi > lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def shared_error_limit(error_a: np.ndarray, error_b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    values = np.concatenate((np.abs(np.asarray(error_a)[valid]), np.abs(np.asarray(error_b)[valid])))
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Cannot determine error limits from an empty mask")
    return max(float(np.percentile(values, 99.0)), np.finfo(float).eps)

