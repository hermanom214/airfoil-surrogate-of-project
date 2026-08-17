from __future__ import annotations

import csv
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_paths
from src.inspect_plausibility import (
    PlausibilityResult,
    evaluate_plausibility,
)


SAVE_FIGURES = True
PICTURES_SUBDIR = Path("data") / "pictures_inspect_flow"
PLAUSIBILITY_CSV_NAME = "inspect_plausibility.csv"
RESIDUALS_SUBDIR = "residuals"
RESIDUAL_FIELDS = ("Ux", "Uy", "p", "omega", "k")
RESIDUAL_CHECK_FIELDS = ("Ux", "Uy", "p")
RESIDUAL_LIMIT = 6.0e-3
RESIDUAL_STRONG_GROWTH_FACTOR = 2.0
FIELD_UX_MIN = -20.0
FIELD_UX_MAX = 40.0
FIELD_P_MIN = -1000.0
FIELD_P_MAX = 500.0
MANUALLY_EXCLUDED_CASES = {
    "case_0354_naca1416_aoam4p0_u22p5": "visually_invalid_flow_field",
    "case_0610_naca2416_aoam2p0_u25p0": "visually_invalid_flow_field",
}

RESIDUAL_RE = re.compile(
    r"Solving for (Ux|Uy|p|omega|k),\s+Initial residual =\s*"
    r"([0-9.eE+\-]+)"
)
TIME_RE = re.compile(r"^Time\s*=\s*(\S+)")


def print_stats(name: str, arr: np.ndarray) -> None:
    print(f"\n{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  min:   {np.nanmin(arr):.6f}")
    print(f"  max:   {np.nanmax(arr):.6f}")
    print(f"  mean:  {np.nanmean(arr):.6f}")
    print(f"  std:   {np.nanstd(arr):.6f}")
    print(f"  NaN count: {np.isnan(arr).sum()}")
    print(f"  Inf count: {np.isinf(arr).sum()}")


def read_initial_residuals(log_path: Path) -> dict[str, list[float]]:
    residuals = {field: [] for field in RESIDUAL_FIELDS}
    current_time = ""
    occurrences_at_time = {field: 0 for field in RESIDUAL_FIELDS}

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            time_match = TIME_RE.match(line.strip())
            if time_match:
                current_time = time_match.group(1)
                occurrences_at_time = {field: 0 for field in RESIDUAL_FIELDS}
                continue

            match = RESIDUAL_RE.search(line)
            if not match:
                continue
            field, value = match.groups()
            occurrences_at_time[field] += 1
            target_occurrence = 3 if field == "p" else 1
            if current_time and occurrences_at_time[field] != target_occurrence:
                continue
            residuals[field].append(float(value))

    return residuals


def check_residual_series(values: list[float]) -> tuple[str, str]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 20:
        return "nOK", f"too_few_residual_points={arr.size}"

    window = max(20, int(math.ceil(0.1 * arr.size)))
    tail = arr[-window:]
    reasons: list[str] = []

    if float(np.max(tail)) >= RESIDUAL_LIMIT:
        reasons.append(f"tail_max={np.max(tail):.6g}>={RESIDUAL_LIMIT:.0e}")
    half = window // 2
    tail_start_median = float(np.median(tail[:half]))
    tail_end_median = float(np.median(tail[half:]))
    if tail_end_median > RESIDUAL_STRONG_GROWTH_FACTOR * tail_start_median:
        reasons.append(
            f"strongly_increasing(tail_ratio={tail_end_median / tail_start_median:.3g})"
        )

    return ("nOK", ";".join(reasons)) if reasons else ("OK", "")


def evaluate_residuals(log_path: Path) -> tuple[dict[str, list[float]], str, str]:
    if not log_path.is_file():
        return {field: [] for field in RESIDUAL_FIELDS}, "nOK", "missing_04_simpleFoam.log"

    residuals = read_initial_residuals(log_path)
    failures: list[str] = []
    for field in RESIDUAL_CHECK_FIELDS:
        status, reason = check_residual_series(residuals[field])
        if status != "OK":
            failures.append(f"{field}:{reason}")
    return residuals, ("nOK" if failures else "OK"), " | ".join(failures)


def save_residual_figure(
    case_name: str,
    residuals: dict[str, list[float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for field in RESIDUAL_FIELDS:
        values = np.asarray(residuals[field], dtype=np.float64)
        if values.size:
            ax.semilogy(np.arange(1, values.size + 1), values, label=field, linewidth=1.0)
    ax.axhline(
        RESIDUAL_LIMIT,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"{RESIDUAL_LIMIT:g} limit",
    )
    ax.set_xlabel("SIMPLE iteration")
    ax.set_ylabel("Initial residual")
    ax.set_title(f"Initial residuals: {case_name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_inspection_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fields = [
        "case_name", "filename", "overall_status", "overall_reason",
        "residuals_status", "residuals_reason", "field_range_status", "field_range_reason",
        "manual_status", "manual_reason",
        "ux_residual_last", "uy_residual_last", "p_residual_last",
        "omega_residual_last", "k_residual_last",
        "ux_min", "ux_max", "p_min", "p_max", "plausibility", "plausibility_reason",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def evaluate_file(file_path: Path) -> dict[str, object]:
    with np.load(file_path, allow_pickle=True) as data:
        required = ["xy", "p", "U", "fluid_mask"]
        for key in required:
            if key not in data:
                raise RuntimeError(f"Missing key '{key}' in {file_path}")

        xy = data["xy"]
        p = data["p"]
        u = data["U"]
        fluid_mask = data["fluid_mask"]

        plausibility: PlausibilityResult = evaluate_plausibility(
            xy=xy,
            p=p,
            u=u,
            fluid_mask=fluid_mask,
            case_name=file_path.name,
        )

        fluid_binary = fluid_mask > 0.5
        solid_fraction = 1.0 - float(np.mean(fluid_binary.astype(np.float32)))
        ux_values = u[:, :, 0][fluid_binary]
        p_values = p[fluid_binary]
        ux_min = float(np.nanmin(ux_values))
        ux_max = float(np.nanmax(ux_values))
        field_p_min = float(np.nanmin(p_values))
        field_p_max = float(np.nanmax(p_values))
        range_reasons: list[str] = []
        if ux_min < FIELD_UX_MIN or ux_max > FIELD_UX_MAX:
            range_reasons.append(f"Ux=[{ux_min:.6g},{ux_max:.6g}] outside [{FIELD_UX_MIN:g},{FIELD_UX_MAX:g}]")
        if field_p_min < FIELD_P_MIN or field_p_max > FIELD_P_MAX:
            range_reasons.append(f"p=[{field_p_min:.6g},{field_p_max:.6g}] outside [{FIELD_P_MIN:g},{FIELD_P_MAX:g}]")

        return {
            "filename": file_path.name,
            "case_name": file_path.parent.name,
            "shape": tuple(int(v) for v in p.shape),
            "solid_fraction": solid_fraction,
            "plausibility": plausibility.plausibility,
            "plausibility_reason": plausibility.reason,
            "p_min": plausibility.p_min,
            "p_max": plausibility.p_max,
            "p_median": plausibility.p_median,
            "u_min": plausibility.u_min,
            "u_max": plausibility.u_max,
            "u_median": plausibility.u_median,
            "p_mean": float(np.nanmean(p)),
            "p_std": float(np.nanstd(p)),
            "ux_min": ux_min,
            "ux_max": ux_max,
            "field_range_status": "nOK" if range_reasons else "OK",
            "field_range_reason": ";".join(range_reasons),
            "ux_mean": float(np.nanmean(u[:, :, 0])),
            "ux_std": float(np.nanstd(u[:, :, 0])),
            "uy_mean": float(np.nanmean(u[:, :, 1])),
            "uy_std": float(np.nanstd(u[:, :, 1])),
            "nan_count_total": float(
                np.isnan(p).sum() + np.isnan(u).sum() + np.isnan(fluid_mask).sum()
            ),
            "inf_count_total": float(
                np.isinf(p).sum() + np.isinf(u).sum() + np.isinf(fluid_mask).sum()
            ),
        }


def save_diagnostic_figure(file_path: Path, output_path: Path) -> None:
    with np.load(file_path, allow_pickle=True) as data:
        xy = data["xy"]
        p = data["p"]
        u = data["U"]
        fluid_mask = data["fluid_mask"] > 0.5

    x = xy[:, :, 0]
    y = xy[:, :, 1]
    u_mag = np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)

    p_plot = np.where(fluid_mask, p, np.nan)
    u_plot = np.where(fluid_mask, u_mag, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(
        p_plot,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        cmap="turbo",
    )
    axes[0].set_title("Pressure p")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_plot,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        cmap="turbo",
    )
    axes[1].set_title("|U|")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        fluid_mask.astype(np.float32),
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title("fluid_mask")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    fig.suptitle(file_path.name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    paths = load_paths(config_path)
    data_dir = paths.flow_fields_output
    pictures_dir = PROJECT_ROOT / PICTURES_SUBDIR
    residuals_dir = pictures_dir / RESIDUALS_SUBDIR

    if SAVE_FIGURES:
        if pictures_dir.exists():
            shutil.rmtree(pictures_dir)
        pictures_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recreated clean figures directory: {pictures_dir}")

    csv_output_path = (
        pictures_dir / PLAUSIBILITY_CSV_NAME
        if SAVE_FIGURES
        else data_dir.parent / PLAUSIBILITY_CSV_NAME
    )

    files = sorted(data_dir.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"No flow-field .npz files found in case directories under {data_dir}")

    print(f"[INFO] Scanning directory: {data_dir}")
    print(f"[INFO] Found files: {len(files)}")

    ok_results: list[dict[str, object]] = []
    failed: list[tuple[str, str]] = []
    plausibility_nok: list[dict[str, object]] = []
    saved_figures = 0

    for idx, file_path in enumerate(files, start=1):
        try:
            result = evaluate_file(file_path)
            residuals, residuals_status, residuals_reason = evaluate_residuals(
                file_path.parent / "logs" / "04_simpleFoam.log"
            )
            result["residuals_status"] = residuals_status
            result["residuals_reason"] = residuals_reason
            manual_reason = MANUALLY_EXCLUDED_CASES.get(file_path.parent.name, "")
            result["manual_status"] = "nOK" if manual_reason else "OK"
            result["manual_reason"] = manual_reason
            for field in RESIDUAL_FIELDS:
                result[f"{field.lower()}_residual_last"] = (
                    residuals[field][-1] if residuals[field] else ""
                )
            overall_reasons: list[str] = []
            if str(result["plausibility"]) != "OK":
                overall_reasons.append(f"plausibility:{result['plausibility_reason']}")
            if residuals_status != "OK":
                overall_reasons.append(f"residuals:{residuals_reason}")
            if str(result["field_range_status"]) != "OK":
                overall_reasons.append(f"field_range:{result['field_range_reason']}")
            if str(result["manual_status"]) != "OK":
                overall_reasons.append(f"manual:{result['manual_reason']}")
            result["overall_status"] = "nOK" if overall_reasons else "OK"
            result["overall_reason"] = " | ".join(overall_reasons)
            ok_results.append(result)
            if str(result["plausibility"]) != "OK":
                plausibility_nok.append(result)

            if SAVE_FIGURES:
                output_png = pictures_dir / f"{file_path.stem}.png"
                save_diagnostic_figure(file_path, output_png)
                save_residual_figure(
                    file_path.parent.name,
                    residuals,
                    residuals_dir / f"{file_path.parent.name}.png",
                )
                saved_figures += 1

            print(f"[{idx}/{len(files)}] OK  {file_path.name}")
        except Exception as exc:
            failed.append((file_path.name, str(exc)))
            print(f"[{idx}/{len(files)}] FAIL {file_path.name} | {exc}")

    if not ok_results:
        raise RuntimeError("All files failed validation.")

    write_inspection_csv(ok_results, csv_output_path)

    p_means = np.array([float(r["p_mean"]) for r in ok_results], dtype=np.float64)
    p_stds = np.array([float(r["p_std"]) for r in ok_results], dtype=np.float64)
    ux_means = np.array([float(r["ux_mean"]) for r in ok_results], dtype=np.float64)
    uy_means = np.array([float(r["uy_mean"]) for r in ok_results], dtype=np.float64)
    solid_fractions = np.array([float(r["solid_fraction"]) for r in ok_results], dtype=np.float64)

    unique_shapes = sorted({tuple(r["shape"]) for r in ok_results})

    total_nan = int(sum(float(r["nan_count_total"]) for r in ok_results))
    total_inf = int(sum(float(r["inf_count_total"]) for r in ok_results))

    print("\n[SUMMARY]")
    print(f"Processed successfully: {len(ok_results)}")
    print(f"Failed: {len(failed)}")
    print(f"Plausibility nOK: {len(plausibility_nok)}")
    print(
        "Overall inspection nOK (excluded from training): "
        f"{sum(str(row['overall_status']) != 'OK' for row in ok_results)}"
    )
    print(
        "Manually excluded: "
        f"{sum(str(row['manual_status']) != 'OK' for row in ok_results)}"
    )
    print(f"Unique grid shapes: {unique_shapes}")
    print(
        "Solid fraction (min/mean/max): "
        f"{solid_fractions.min():.6f} / {solid_fractions.mean():.6f} / {solid_fractions.max():.6f}"
    )
    print(
        "p mean (min/mean/max): "
        f"{p_means.min():.6f} / {p_means.mean():.6f} / {p_means.max():.6f}"
    )
    print(
        "p std (min/mean/max): "
        f"{p_stds.min():.6f} / {p_stds.mean():.6f} / {p_stds.max():.6f}"
    )
    print(
        "Ux mean (min/mean/max): "
        f"{ux_means.min():.6f} / {ux_means.mean():.6f} / {ux_means.max():.6f}"
    )
    print(
        "Uy mean (min/mean/max): "
        f"{uy_means.min():.6f} / {uy_means.mean():.6f} / {uy_means.max():.6f}"
    )
    print(f"Total NaN count across all files: {total_nan}")
    print(f"Total Inf count across all files: {total_inf}")
    if SAVE_FIGURES:
        print(f"Saved/updated figures: {saved_figures}")
        print(f"Figures directory: {pictures_dir}")
        print(f"Residual figures directory: {residuals_dir}")
    print(f"Plausibility CSV: {csv_output_path}")

    # Print detailed stats for one representative valid file.
    representative = ok_results[0]
    representative_path = next(
        file_path
        for file_path in files
        if file_path.name == str(representative["filename"])
    )
    with np.load(representative_path, allow_pickle=True) as data:
        print(f"\n[REPRESENTATIVE FILE] {representative_path.name}")
        print_stats("Pressure p", data["p"])
        print_stats("Velocity Ux", data["U"][:, :, 0])
        print_stats("Velocity Uy", data["U"][:, :, 1])
        print_stats("fluid_mask", data["fluid_mask"])

    if failed:
        print("\n[FAILED FILES]")
        max_to_show = min(20, len(failed))
        for name, error in failed[:max_to_show]:
            print(f"- {name}: {error}")
        remaining = len(failed) - max_to_show
        if remaining > 0:
            print(f"... and {remaining} more")

    if plausibility_nok:
        print("\n[PLAUSIBILITY nOK CASES]")
        for row in plausibility_nok:
            print(f"- {row['filename']}: {row['plausibility_reason']}")

    if total_nan > 0 or total_inf > 0:
        print("\n[WARN] Some files contain NaN/Inf values.")
    else:
        print("\n[INFO] No NaN/Inf values detected in processed files.")


if __name__ == "__main__":
    main()
