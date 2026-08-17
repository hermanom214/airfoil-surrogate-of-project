from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_paths
from src.inspect_plausibility import (
    PlausibilityResult,
    evaluate_plausibility,
    write_plausibility_csv,
)


SAVE_FIGURES = True
PICTURES_SUBDIR = Path("data") / "pictures_inspect_flow"
PLAUSIBILITY_CSV_NAME = "inspect_plausibility.csv"


def print_stats(name: str, arr: np.ndarray) -> None:
    print(f"\n{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  min:   {np.nanmin(arr):.6f}")
    print(f"  max:   {np.nanmax(arr):.6f}")
    print(f"  mean:  {np.nanmean(arr):.6f}")
    print(f"  std:   {np.nanstd(arr):.6f}")
    print(f"  NaN count: {np.isnan(arr).sum()}")
    print(f"  Inf count: {np.isinf(arr).sum()}")


def evaluate_file(file_path: Path) -> dict[str, float | str | tuple[int, ...]]:
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

        return {
            "filename": file_path.name,
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

    ok_results: list[dict[str, float | str | tuple[int, ...]]] = []
    failed: list[tuple[str, str]] = []
    plausibility_nok: list[dict[str, float | str | tuple[int, ...]]] = []
    saved_figures = 0

    for idx, file_path in enumerate(files, start=1):
        try:
            result = evaluate_file(file_path)
            ok_results.append(result)
            if str(result["plausibility"]) != "OK":
                plausibility_nok.append(result)

            if SAVE_FIGURES:
                output_png = pictures_dir / f"{file_path.stem}.png"
                save_diagnostic_figure(file_path, output_png)
                saved_figures += 1

            print(f"[{idx}/{len(files)}] OK  {file_path.name}")
        except Exception as exc:
            failed.append((file_path.name, str(exc)))
            print(f"[{idx}/{len(files)}] FAIL {file_path.name} | {exc}")

    if not ok_results:
        raise RuntimeError("All files failed validation.")

    write_plausibility_csv(
        [
            PlausibilityResult(
                case_name=str(row["filename"]),
                plausibility=str(row["plausibility"]),
                reason=str(row["plausibility_reason"]),
                p_min=float(row["p_min"]),
                p_max=float(row["p_max"]),
                p_median=float(row["p_median"]),
                u_min=float(row["u_min"]),
                u_max=float(row["u_max"]),
                u_median=float(row["u_median"]),
            )
            for row in ok_results
        ],
        csv_output_path,
    )

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
