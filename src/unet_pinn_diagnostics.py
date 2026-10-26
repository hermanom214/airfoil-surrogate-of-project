"""Testable helpers for post-hoc improved U-Net/PINN diagnostics."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping

import numpy as np


def error_statistics(prediction, reference, mask):
    valid = np.asarray(mask, bool) & np.isfinite(prediction) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("Diagnostic mask contains no finite cells")
    error = np.asarray(prediction)[valid] - np.asarray(reference)[valid]
    absolute = np.abs(error)
    return {"count": int(error.size), "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
        "max_abs": float(absolute.max()), "p95_abs": float(np.percentile(absolute, 95))}


def residual_statistics(residual, mask):
    valid = np.asarray(mask, bool) & np.isfinite(residual)
    values = np.asarray(residual)[valid]
    if not values.size:
        raise ValueError("Residual mask contains no finite cells")
    return {"count": int(values.size), "rms": float(np.sqrt(np.mean(values ** 2))),
            "bias": float(values.mean()), "p95_abs": float(np.percentile(np.abs(values), 95))}


def gradient_magnitude(field, xy, mask):
    values, coords = np.asarray(field, float), np.asarray(xy, float)
    dx = float(coords[0, 1, 0] - coords[0, 0, 0])
    dy = float(coords[1, 0, 1] - coords[0, 0, 1])
    gx = np.zeros_like(values); gy = np.zeros_like(values)
    gx[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2 * dx)
    gy[1:-1, :] = (values[2:, :] - values[:-2, :]) / (2 * dy)
    base = np.asarray(mask, bool)
    valid = np.zeros_like(base); valid[1:-1, 1:-1] = (base[1:-1, 1:-1]
        & base[1:-1, :-2] & base[1:-1, 2:] & base[:-2, 1:-1] & base[2:, 1:-1])
    return np.hypot(gx, gy), valid


def aggregate_error_rows(rows: Iterable[Mapping[str, object]]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["region"], row["field"])].append(row)
    result = []
    for (region, field), subset in sorted(groups.items()):
        item = {"region": region, "field": field, "case_count": len(subset)}
        for model in ("supervised", "pinn"):
            for metric in ("mae", "rmse", "bias", "max_abs", "p95_abs"):
                values = np.asarray([r[f"{model}_{metric}"] for r in subset], float)
                item[f"case_mean_{model}_{metric}"] = float(values.mean())
                item[f"case_median_{model}_{metric}"] = float(np.median(values))
            count = sum(int(r["valid_cells"]) for r in subset)
            item[f"pooled_{model}_mae"] = sum(float(r[f"{model}_abs_sum"]) for r in subset) / count
            item[f"pooled_{model}_rmse"] = math.sqrt(sum(float(r[f"{model}_sq_sum"]) for r in subset) / count)
        deltas = np.asarray([r["pinn_mae"] - r["supervised_mae"] for r in subset])
        item.update(improved_cases=int((deltas < 0).sum()), worsened_cases=int((deltas > 0).sum()),
                    tied_cases=int((deltas == 0).sum()))
        result.append(item)
    return result


def select_diagnostic_cases(rows, metadata, count_each=2):
    """Outcome-driven, unique selection using dimensionless within-field comparisons."""
    by_key = {(r["case_id"], r["region"], r["field"]): r for r in rows}
    ids = sorted(metadata)
    near_p = sorted(ids, key=lambda cid: by_key[cid, "near_airfoil", "p"]["pinn_mae"]
                    - by_key[cid, "near_airfoil", "p"]["supervised_mae"], reverse=True)
    wake_ux = sorted(ids, key=lambda cid: by_key[cid, "wake", "Ux"]["pinn_mae"]
                     - by_key[cid, "wake", "Ux"]["supervised_mae"])
    scale = {field: np.median([by_key[cid, "full_domain", field]["supervised_mae"]
                              for cid in ids]) for field in ("p", "Ux", "Uy")}
    similar = sorted(ids, key=lambda cid: max(abs(by_key[cid, "full_domain", field]["pinn_mae"]
        - by_key[cid, "full_domain", field]["supervised_mae"]) / max(scale[field], 1e-12)
        for field in ("p", "Ux", "Uy")))
    similarity_scores = {cid: max(abs(by_key[cid, "full_domain", field]["pinn_mae"]
        - by_key[cid, "full_domain", field]["supervised_mae"]) / max(scale[field], 1e-12)
        for field in ("p", "Ux", "Uy")) for cid in ids}
    selected, used = [], set()
    for category, ordered in (("largest near-airfoil pressure deterioration", near_p),
                              ("largest wake Ux improvement", wake_ux),
                              ("most similar overall field errors", similar)):
        added = 0
        for cid in ordered:
            if cid in used:
                continue
            used.add(cid); added += 1
            r = by_key[cid, "near_airfoil" if "pressure" in category else
                       "wake" if "wake" in category else "full_domain",
                       "p" if "pressure" in category else "Ux"]
            selected.append({**metadata[cid], "selection_category": category,
                "selection_supervised_mae": r["supervised_mae"],
                "selection_pinn_mae": r["pinn_mae"],
                "selection_delta_mae": r["pinn_mae"] - r["supervised_mae"],
                **({"similarity_score_max_abs_delta_over_population_median_A_mae": similarity_scores[cid]}
                   if "similar" in category else {})})
            if added == count_each:
                break
    if len(selected) != 3 * count_each:
        raise ValueError("Insufficient unique cases for diagnostic selection")
    return selected


def body_to_global(x_body, y_body, chord, aoa_deg):
    angle = np.deg2rad(aoa_deg); cx = .25 * chord
    dx = np.asarray(x_body) - cx; yb = np.asarray(y_body)
    return cx + dx * np.cos(angle) + yb * np.sin(angle), -dx * np.sin(angle) + yb * np.cos(angle)
