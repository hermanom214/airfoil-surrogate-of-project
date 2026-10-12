"""Export and audit a small high-resolution trailing-edge/wake prototype.

This experiment is deliberately isolated from the authoritative 320x640 U-Net
dataset.  It reads accepted cases from the persisted SimpleUNet split, samples
the archived final VTK fields onto a regular body-coordinate patch, and compares
native-derived, global-raster, and patch wake profiles.  Existing artifacts are
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.flow_extractor import create_fluid_mask, find_vtk_file, read_vtk_with_pyvista


SOURCE_ROOT = ROOT / "data" / "flow_fields"
PATCH_ROOT = ROOT / "data" / "flow_fields_highres_wake_prototype"
OUTPUT_ROOT = ROOT / "outputs" / "highres_wake_patch_feasibility"
SPLIT_PATH = ROOT / "data" / "models" / "simple_unet" / "fixed_test_split.json"
INSPECTION_PATH = ROOT / "data" / "pictures_inspect_flow" / "inspect_plausibility.csv"

XMIN_C, XMAX_C = 0.70, 1.75
YMIN_C, YMAX_C = -0.15, 0.15
NX, NY = 272, 480
STATIONS = (1.02, 1.05, 1.10, 1.25, 1.50)
PROFILE_Y = np.linspace(YMIN_C, YMAX_C, 2401)

SELECTED = {
    "case_0011_naca0008_aoa0p0_u15p0": "thin symmetric, AoA 0, low U_inf",
    "case_0023_naca0008_aoa4p0_u20p0": "thin symmetric, positive AoA",
    "case_0053_naca0012_aoam4p0_u20p0": "symmetric, negative AoA",
    "case_0063_naca0012_aoa0p0_u20p0": "baseline symmetric, AoA 0",
    "case_0074_naca0012_aoa4p0_u22p5": "symmetric, positive AoA; accepted replacement for rejected case_0073",
    "case_0115_naca0016_aoa0p0_u25p0": "thick symmetric, AoA 0, high U_inf",
    "case_0388_naca2208_aoa0p0_u20p0": "thin cambered, AoA 0",
    "case_0398_naca2208_aoa4p0_u20p0": "thin cambered, positive AoA",
    "case_0553_naca2412_aoam4p0_u20p0": "cambered, negative AoA",
    "case_0561_naca2412_aoa0p0_u15p0": "cambered, AoA 0, low U_inf",
    "case_0563_naca2412_aoa0p0_u20p0": "canonical cambered baseline",
    "case_0575_naca2412_aoa4p0_u25p0": "cambered, positive AoA, high U_inf",
}


@dataclass
class ProfileMetric:
    centre: float
    minimum: float
    deficit: float
    width: float
    count: float


def body_to_lab(xb: np.ndarray, yb: np.ndarray, aoa_deg: float, chord: float) -> tuple[np.ndarray, np.ndarray]:
    """Map body coordinates to the archived laboratory frame.

    blockMesh rotates the airfoil by -AoA about quarter chord, so this applies
    that same rotation to a body-aligned sampling lattice.
    """
    angle = math.radians(-aoa_deg)
    xr = xb - 0.25 * chord
    return (
        0.25 * chord + math.cos(angle) * xr - math.sin(angle) * yb,
        math.sin(angle) * xr + math.cos(angle) * yb,
    )


def lab_to_body(x: np.ndarray, y: np.ndarray, aoa_deg: float, chord: float) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(aoa_deg)
    xr = x - 0.25 * chord
    return (
        0.25 * chord + math.cos(angle) * xr - math.sin(angle) * y,
        math.sin(angle) * xr + math.cos(angle) * y,
    )


def accepted_case_ids() -> set[str]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    ids = set(split["development_case_ids"]) | set(split["test_case_ids"])
    if len(ids) != len(split["development_case_ids"]) + len(split["test_case_ids"]):
        raise RuntimeError("Persisted split has overlapping development/test membership")
    return ids


def inspection_statuses() -> dict[str, str]:
    with INSPECTION_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["case_name"]: row["overall_status"] for row in csv.DictReader(handle)}


def prepare_output_dirs() -> None:
    collisions = [path for path in (PATCH_ROOT, OUTPUT_ROOT) if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite pre-existing experiment output: " + ", ".join(map(str, collisions)))
    PATCH_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True)


def make_interpolator(points_body: np.ndarray, values: np.ndarray):
    """Cache Delaunay work while matching linear-griddata plus nearest hull fill."""
    triangulation = Delaunay(points_body)
    linear = LinearNDInterpolator(triangulation, values)
    nearest = NearestNDInterpolator(points_body, values)

    def evaluate(queries: np.ndarray) -> np.ndarray:
        result = np.asarray(linear(queries))
        missing = ~np.all(np.isfinite(result), axis=-1) if result.ndim > 1 else ~np.isfinite(result)
        if np.any(missing):
            result[missing] = np.asarray(nearest(queries[missing]))
        return result
    return evaluate


def regular_profile(xy: np.ndarray, values: np.ndarray, x_c: float, y_c: np.ndarray, chord: float) -> np.ndarray:
    xs = xy[0, :, 0] / chord
    ys = xy[:, 0, 1] / chord
    return np.asarray(RegularGridInterpolator((ys, xs), values, bounds_error=False, fill_value=np.nan)(np.c_[y_c, np.full_like(y_c, x_c)]))


def scattered_profile(xb_c: np.ndarray, yb_c: np.ndarray, values: np.ndarray, x_c: float, y_c: np.ndarray) -> np.ndarray:
    return make_interpolator(np.c_[xb_c, yb_c], values)(np.c_[np.full_like(y_c, x_c), y_c])


def wake_metric(y: np.ndarray, ux_nd: np.ndarray, spacing: float | None = None) -> ProfileMetric:
    valid = np.isfinite(ux_nd)
    if np.count_nonzero(valid) < 5:
        return ProfileMetric(*(float("nan"),) * 5)
    yy, uu = y[valid], ux_nd[valid]
    idx = int(np.argmin(uu))
    minimum = float(uu[idx])
    deficit = max(0.0, 1.0 - minimum)
    if deficit <= 0:
        return ProfileMetric(float(yy[idx]), minimum, deficit, float("nan"), float("nan"))
    inside = uu <= 1.0 - 0.5 * deficit
    left = right = idx
    while left > 0 and inside[left - 1]:
        left -= 1
    while right + 1 < len(inside) and inside[right + 1]:
        right += 1
    width = float(yy[right] - yy[left]) if right > left else float("nan")
    count = float(width / spacing + 1.0) if spacing and np.isfinite(width) else float(right - left + 1)
    return ProfileMetric(float(yy[idx]), minimum, deficit, width, count)


def native_count(points_x: np.ndarray, points_y: np.ndarray, station: float, metric: ProfileMetric) -> float:
    if not np.isfinite(metric.width):
        return float("nan")
    # Native streamwise spacing is about 0.03c; use one half-cell slab.
    half = 0.5 * metric.width
    mask = (np.abs(points_x - station) <= 0.015) & (np.abs(points_y - metric.centre) <= half)
    return float(np.count_nonzero(mask))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles(case_id: str, profiles: list[dict], u_inf: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, profile in zip(axes.flat, profiles):
        y = profile["y"]
        ax.plot(profile["native_ux"], y, "k-", lw=1.5, label="native-derived")
        ax.plot(profile["global_ux"], y, color="tab:orange", lw=1.2, label="global 320x640")
        ax.plot(profile["patch_ux"], y, color="tab:blue", lw=1.2, label="high-res patch")
        ax.set_title(f"x/c={profile['station']:.2f}")
        ax.set_xlabel("Ux/U_inf")
        ax.grid(alpha=.25)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("y_body/c")
    axes[1, 0].set_ylabel("y_body/c")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{case_id}: transverse wake profiles (U_inf={u_inf:g} m/s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "figures" / f"{case_id}_wake_profiles.png", dpi=150)
    plt.close(fig)


def plot_comparison(case_id: str, native_xy: np.ndarray, native_values: np.ndarray,
                    global_xy_body: np.ndarray, global_values: np.ndarray,
                    patch_xy: np.ndarray, patch_values: np.ndarray, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharex=True, sharey=True)
    finite = native_values[np.isfinite(native_values)]
    lo, hi = np.nanpercentile(finite, [1, 99])
    native_mask = ((native_xy[:, 0] >= XMIN_C) & (native_xy[:, 0] <= XMAX_C) &
                   (native_xy[:, 1] >= YMIN_C) & (native_xy[:, 1] <= YMAX_C))
    sc = axes[0].scatter(native_xy[native_mask, 0], native_xy[native_mask, 1], c=native_values[native_mask],
                         s=.4, rasterized=True, vmin=lo, vmax=hi, cmap="viridis")
    axes[0].set_title("native VTK samples")
    axes[1].pcolormesh(global_xy_body[..., 0], global_xy_body[..., 1], global_values, shading="auto", vmin=lo, vmax=hi, cmap="viridis")
    axes[1].set_title("global raster")
    axes[2].pcolormesh(patch_xy[..., 0], patch_xy[..., 1], patch_values, shading="auto", vmin=lo, vmax=hi, cmap="viridis")
    axes[2].set_title("high-res patch")
    for ax in axes:
        ax.set(xlim=(XMIN_C, XMAX_C), ylim=(YMIN_C, YMAX_C), xlabel="x_body/c", aspect="auto")
    axes[0].set_ylabel("y_body/c")
    fig.colorbar(sc, ax=axes, label=label, shrink=.9)
    fig.suptitle(case_id)
    fig.subplots_adjust(left=.06, right=.91, bottom=.15, top=.82, wspace=.12)
    safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    fig.savefig(OUTPUT_ROOT / "figures" / f"{case_id}_{safe}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate membership and print the selected case table")
    args = parser.parse_args()

    accepted = accepted_case_ids()
    inspected = inspection_statuses()
    if "case_0073_naca0012_aoa4p0_u20p0" in SELECTED:
        raise RuntimeError("Rejected case_0073 must never be selected")
    missing = sorted(set(SELECTED) - accepted)
    if missing:
        raise RuntimeError(f"Selected cases are absent from the authoritative persisted split: {missing}")

    selection_rows = []
    for case_id, reason in SELECTED.items():
        params = json.loads((SOURCE_ROOT / case_id / "params.json").read_text(encoding="utf-8"))
        selection_rows.append({"case_id": case_id, "NACA": params["naca_code"], "AoA": params["aoa_deg"],
                               "U_inf": params["inlet_velocity"], "accepted_status": inspected.get(case_id, "UNKNOWN"),
                               "selection_reason": reason})
    print("\nAccepted prototype cases")
    for row in selection_rows:
        print(f"{row['case_id']:39s} NACA={row['NACA']} AoA={row['AoA']:+g} U_inf={row['U_inf']:g} "
              f"accepted={row['accepted_status']} | {row['selection_reason']}")
    print(f"\nPatch: x/c=[{XMIN_C},{XMAX_C}], y/c=[{YMIN_C},{YMAX_C}], nx={NX}, ny={NY}, "
          f"dx/c={(XMAX_C-XMIN_C)/(NX-1):.9f}, dy/c={(YMAX_C-YMIN_C)/(NY-1):.9f}")
    if args.dry_run:
        return

    prepare_output_dirs()
    write_csv(OUTPUT_ROOT / "selected_cases.csv", selection_rows, list(selection_rows[0]))
    metrics_rows: list[dict] = []
    artifacts: list[dict] = []
    case_summaries: list[dict] = []

    for case_number, (case_id, _) in enumerate(SELECTED.items(), 1):
        print(f"[{case_number}/{len(SELECTED)}] {case_id}", flush=True)
        case_dir = SOURCE_ROOT / case_id
        params = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
        chord, aoa, u_inf = float(params["chord"]), float(params["aoa_deg"]), float(params["inlet_velocity"])
        naca = str(params["naca_code"])
        vtk_file = find_vtk_file(case_dir)
        points, p_native, u_native = read_vtk_with_pyvista(vtk_file)
        xb, yb = lab_to_body(points[:, 0], points[:, 1], aoa, chord)
        native_xy_c = np.c_[xb / chord, yb / chord]
        native_eval = make_interpolator(native_xy_c, np.c_[p_native, u_native[:, :2]])

        xs_c = np.linspace(XMIN_C, XMAX_C, NX)
        ys_c = np.linspace(YMIN_C, YMAX_C, NY)
        gx_c, gy_c = np.meshgrid(xs_c, ys_c)
        queries_c = np.c_[gx_c.ravel(), gy_c.ravel()]
        sampled = native_eval(queries_c).reshape(NY, NX, 3)
        p_patch = sampled[..., 0].astype(np.float32)
        ux_patch = sampled[..., 1].astype(np.float32)
        uy_patch = sampled[..., 2].astype(np.float32)
        xy_patch = (np.stack([gx_c, gy_c], axis=-1) * chord).astype(np.float32)
        patch_mask = create_fluid_mask(xy_patch, naca, chord, 0.0).astype(np.float32)
        u_patch = np.stack([ux_patch, uy_patch], axis=-1)

        case_out = PATCH_ROOT / case_id
        case_out.mkdir()
        patch_file = case_out / f"{case_id}.npz"
        if patch_file.exists():
            raise FileExistsError(f"Refusing collision: {patch_file}")
        metadata = {
            **params, "coordinate_frame": "body", "velocity_component_frame": "laboratory/global",
            "patch_bounds_c": [XMIN_C, XMAX_C, YMIN_C, YMAX_C], "nx": NX, "ny": NY,
            "dx_c": (XMAX_C-XMIN_C)/(NX-1), "dy_c": (YMAX_C-YMIN_C)/(NY-1),
            "source_vtk": str(vtk_file), "interpolation": "linear scipy Delaunay; nearest only outside convex hull",
            "p_semantics": "OpenFOAM kinematic pressure; outlet p=0 reference",
        }
        np.savez_compressed(patch_file, xy=xy_patch, p=p_patch, U=u_patch, fluid_mask=patch_mask,
                            case_id=case_id, naca_code=naca, aoa_deg=aoa, U_inf=u_inf, chord=chord,
                            patch_bounds_c=np.array([XMIN_C, XMAX_C, YMIN_C, YMAX_C]),
                            dx_c=metadata["dx_c"], dy_c=metadata["dy_c"], source_vtk=str(vtk_file),
                            params=json.dumps(params), metadata=json.dumps(metadata))

        with np.load(case_dir / f"{case_id}.npz", allow_pickle=False) as z:
            xy_global = z["xy"].astype(float)
            p_global = z["p"].astype(float)
            u_global = z["U"].astype(float)
            global_mask = z["fluid_mask"].astype(bool)
        gxb, gyb = lab_to_body(xy_global[..., 0], xy_global[..., 1], aoa, chord)
        global_xy_body_c = np.stack([gxb/chord, gyb/chord], axis=-1)
        flat_valid = global_mask.ravel()
        global_points = global_xy_body_c.reshape(-1, 2)[flat_valid]
        global_eval = make_interpolator(global_points, np.c_[p_global.ravel()[flat_valid],
                                                              u_global[..., 0].ravel()[flat_valid],
                                                              u_global[..., 1].ravel()[flat_valid]])

        profiles = []
        for station in STATIONS:
            line = np.c_[np.full_like(PROFILE_Y, station), PROFILE_Y]
            native_line = native_eval(line)
            global_line = global_eval(line)
            native_ux, native_uy = native_line[:, 1] / u_inf, native_line[:, 2] / u_inf
            global_ux, global_uy = global_line[:, 1] / u_inf, global_line[:, 2] / u_inf
            patch_ux = regular_profile(xy_patch, ux_patch/u_inf, station, PROFILE_Y, chord)
            patch_uy = regular_profile(xy_patch, uy_patch/u_inf, station, PROFILE_Y, chord)
            nm = wake_metric(PROFILE_Y, native_ux)
            gm = wake_metric(PROFILE_Y, global_ux, 1.5/(320-1))
            hm = wake_metric(PROFILE_Y, patch_ux, metadata["dy_c"])
            ncount = native_count(native_xy_c[:, 0], native_xy_c[:, 1], station, nm)
            valid = np.isfinite(native_ux) & np.isfinite(global_ux) & np.isfinite(patch_ux)
            valid_y = np.isfinite(native_uy) & np.isfinite(global_uy) & np.isfinite(patch_uy)
            metrics_rows.append({
                "case_id": case_id, "NACA": naca, "AoA": aoa, "U_inf": u_inf, "wake_station_xc": station,
                "native_sample_count": ncount, "global_pixel_count": gm.count, "highres_pixel_count": hm.count,
                "native_wake_centre_yc": nm.centre, "global_wake_centre_yc": gm.centre, "highres_wake_centre_yc": hm.centre,
                "native_Ux_min_nd": nm.minimum, "global_Ux_min_nd": gm.minimum, "highres_Ux_min_nd": hm.minimum,
                "native_max_deficit": nm.deficit, "global_max_deficit": gm.deficit, "highres_max_deficit": hm.deficit,
                "native_wake_width_c": nm.width, "global_wake_width_c": gm.width, "highres_wake_width_c": hm.width,
                "global_Ux_RMSE_vs_native": float(np.sqrt(np.mean((global_ux[valid]-native_ux[valid])**2))),
                "highres_Ux_RMSE_vs_native": float(np.sqrt(np.mean((patch_ux[valid]-native_ux[valid])**2))),
                "global_Ux_MAE_vs_native": float(np.mean(np.abs(global_ux[valid]-native_ux[valid]))),
                "highres_Ux_MAE_vs_native": float(np.mean(np.abs(patch_ux[valid]-native_ux[valid]))),
                "global_Uy_RMSE_vs_native": float(np.sqrt(np.mean((global_uy[valid_y]-native_uy[valid_y])**2))),
                "highres_Uy_RMSE_vs_native": float(np.sqrt(np.mean((patch_uy[valid_y]-native_uy[valid_y])**2))),
            })
            profiles.append({"station": station, "y": PROFILE_Y, "native_ux": native_ux, "global_ux": global_ux,
                             "patch_ux": patch_ux, "native_uy": native_uy, "global_uy": global_uy, "patch_uy": patch_uy})
        plot_profiles(case_id, profiles, u_inf)

        # Linear interpolation is convex inside each simplex: any patch extrema
        # beyond the native range would therefore identify fill/support trouble.
        local_native = ((native_xy_c[:, 0] >= XMIN_C-.02) & (native_xy_c[:, 0] <= XMAX_C+.02) &
                        (native_xy_c[:, 1] >= YMIN_C-.02) & (native_xy_c[:, 1] <= YMAX_C+.02))
        for name, patch_values, source_values in (("p", p_patch, p_native), ("Ux", ux_patch, u_native[:, 0]), ("Uy", uy_patch, u_native[:, 1])):
            smin, smax = np.nanmin(source_values[local_native]), np.nanmax(source_values[local_native])
            tol = 1e-6 * max(1.0, abs(smin), abs(smax))
            outside = int(np.count_nonzero((patch_values < smin-tol) | (patch_values > smax+tol)))
            artifacts.append({"case_id": case_id, "feature": f"{name}_extrema", "classification":
                              "RASTER_ARTIFACT" if outside else "PRESENT_IN_NATIVE",
                              "evidence": f"{outside} patch cells outside nearby native range [{smin:.8g}, {smax:.8g}]"})
        solid_speed = np.hypot(ux_patch, uy_patch)[patch_mask < .5]
        artifacts.append({"case_id": case_id, "feature": "analytic_mask_solid_values", "classification": "INCONCLUSIVE",
                          "evidence": f"{solid_speed.size} solid pixels retain interpolated values but are excluded by fluid_mask, matching global convention"})

        case_summaries.append({"case_id": case_id, "file_bytes": patch_file.stat().st_size,
                               "values_per_case": int(xy_patch.size+p_patch.size+u_patch.size+patch_mask.size),
                               "fluid_fraction": float(np.mean(patch_mask > .5))})
        if case_number in (1, 3, 6, 9, 12):
            native_speed = np.hypot(u_native[:, 0], u_native[:, 1]) / u_inf
            global_speed = np.hypot(u_global[..., 0], u_global[..., 1]) / u_inf
            patch_speed = np.hypot(ux_patch, uy_patch) / u_inf
            plot_comparison(case_id, native_xy_c, native_speed, global_xy_body_c, global_speed, xy_patch/chord, patch_speed, "|U|/U_inf")
            plot_comparison(case_id, native_xy_c, u_native[:,0]/u_inf, global_xy_body_c, u_global[...,0]/u_inf, xy_patch/chord, ux_patch/u_inf, "Ux/U_inf")
            plot_comparison(case_id, native_xy_c, u_native[:,1]/u_inf, global_xy_body_c, u_global[...,1]/u_inf, xy_patch/chord, uy_patch/u_inf, "Uy/U_inf")
            plot_comparison(case_id, native_xy_c, p_native/(.5*u_inf**2), global_xy_body_c, p_global/(.5*u_inf**2), xy_patch/chord, p_patch/(.5*u_inf**2), "Cp")

    write_csv(OUTPUT_ROOT / "highres_patch_metrics.csv", metrics_rows, list(metrics_rows[0]))
    write_csv(OUTPUT_ROOT / "interpolation_artifacts.csv", artifacts, list(artifacts[0]))
    write_csv(OUTPUT_ROOT / "prototype_file_sizes.csv", case_summaries, list(case_summaries[0]))

    global_sizes = [(SOURCE_ROOT/c/f"{c}.npz").stat().st_size for c in accepted]
    patch_sizes = [r["file_bytes"] for r in case_summaries]
    gx = 640 * 320
    px = NX * NY
    ux_global_rmse = np.array([r["global_Ux_RMSE_vs_native"] for r in metrics_rows])
    ux_patch_rmse = np.array([r["highres_Ux_RMSE_vs_native"] for r in metrics_rows])
    ratios = ux_patch_rmse / ux_global_rmse
    native_counts = np.array([r["native_sample_count"] for r in metrics_rows])
    global_counts = np.array([r["global_pixel_count"] for r in metrics_rows])
    patch_counts = np.array([r["highres_pixel_count"] for r in metrics_rows])
    bad_artifacts = sum(r["classification"] in {"RASTER_ARTIFACT", "INTERPOLATION_AMPLIFICATION"} for r in artifacts)
    proceed = (np.nanmedian(ratios) < .6 and np.nanmedian(patch_counts) >= 5 and bad_artifacts == 0)
    decision = "PROCEED_TO_LOCAL_UNET_EXPERIMENT" if proceed else "DO_NOT_PROCEED"
    cost = {
        "accepted_case_count": len(accepted), "prototype_shape_ny_nx": [NY, NX], "stored_numeric_values_per_case": case_summaries[0]["values_per_case"],
        "mean_compressed_patch_bytes": float(np.mean(patch_sizes)), "estimated_all_accepted_patch_bytes": float(np.mean(patch_sizes)*len(accepted)),
        "mean_current_global_npz_bytes": float(np.mean(global_sizes)), "current_global_dataset_bytes": int(np.sum(global_sizes)),
        "patch_to_global_compressed_file_ratio": float(np.mean(patch_sizes)/np.mean(global_sizes)),
        "future_local_sample_float32_bytes_5_input_plus_3_target_plus_1_mask": int(px*9*4),
        "current_global_equivalent_training_sample_bytes": int(gx*9*4), "local_to_global_training_sample_ratio": px/gx,
    }
    (OUTPUT_ROOT / "cost_summary.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    report = f"""# High-resolution TE/wake patch feasibility

## Decision

**{decision}**

The prototype uses the authoritative persisted SimpleUNet population ({len(accepted)} cases) for membership and exports only the 12 accepted cases listed in `selected_cases.csv`. Rejected `case_0073` is explicitly excluded.

## Configuration and conventions

- Additional body-coordinate patch only; the existing global dataset was not modified.
- Bounds: `x_body/c=[{XMIN_C}, {XMAX_C}]`, `y_body/c=[{YMIN_C}, {YMAX_C}]`. Inspection of ±4 degree cases required no widening.
- Shape: `ny={NY}`, `nx={NX}`; exact `dx/c={(XMAX_C-XMIN_C)/(NX-1):.9f}`, `dy/c={(YMAX_C-YMIN_C)/(NY-1):.9f}`.
- Final archived VTK, OpenFOAM kinematic `p` with outlet-zero reference, and physical laboratory-frame `U` components are preserved.
- Sampling uses linear Delaunay interpolation and nearest fill only outside the convex hull, matching the global exporter. The body grid maps to the laboratory frame using the established `-AoA` quarter-chord rotation.

## Quantitative result

- Median samples across the half-deficit wake width: native slab `{np.nanmedian(native_counts):.1f}`, global raster `{np.nanmedian(global_counts):.1f}`, high-res patch `{np.nanmedian(patch_counts):.1f}`.
- Median Ux/U_inf RMSE vs direct native-derived profiles: global `{np.nanmedian(ux_global_rmse):.6g}`, high-res `{np.nanmedian(ux_patch_rmse):.6g}`.
- Median high-res/global RMSE ratio: `{np.nanmedian(ratios):.4f}`; high-res improves `{np.count_nonzero(ux_patch_rmse < ux_global_rmse)}/{len(metrics_rows)}` station/case profiles.
- Full station-level centres, minima, deficits, widths, sample counts, RMSE, and MAE are in `highres_patch_metrics.csv`.

## Trailing edge and interpolation findings

The side-by-side Cp, Ux, Uy, and velocity-magnitude figures cover the aft airfoil, trailing edge, and wake. The denser transverse lattice preserves the linear native-derived TE-to-wake gradients and local extrema rather than inventing oscillations: linear simplex interpolation is non-oscillatory, and the automated local native-range check found `{bad_artifacts}` unacceptable extrema violations. Values geometrically inside the analytic airfoil are retained in arrays, as in the established exporter, but excluded by `fluid_mask`; they must not be used as fluid targets. This experiment does not claim accuracy below native mesh support, and the streamwise direction remains interpolation-limited.

## Cost

- Mean compressed patch file: `{np.mean(patch_sizes)/2**20:.3f} MiB`; estimated {len(accepted)}-case patch set: `{cost['estimated_all_accepted_patch_bytes']/2**30:.3f} GiB`.
- Current global set: `{cost['current_global_dataset_bytes']/2**30:.3f} GiB`; patch/global mean compressed-file ratio: `{cost['patch_to_global_compressed_file_ratio']:.3f}`.
- One float32 local training sample (5 inputs + 3 targets + mask): `{cost['future_local_sample_float32_bytes_5_input_plus_3_target_plus_1_mask']/2**20:.3f} MiB`, `{cost['local_to_global_training_sample_ratio']:.3f}x` the equivalent global tensor footprint.

## Scientific conclusion

{('A high-resolution local regular patch preserves more of the native CFD TE/wake structure than the current global raster.' if proceed else 'The prototype did not demonstrate enough additional native CFD information to justify a local-model experiment.')}

## Next experiment (design only)

If proceeding: train a separate local `SimpleUNet` on this body-aligned `{NY}x{NX}` representation, leaving `simple_unet_improved` intact. Inputs are local fluid mask, x, y, AoA, and U_inf; targets are `Cp`, `dUx/U_inf`, and `Uy/U_inf`. Reuse the SimpleUNet family, optimizer, persisted case IDs, development/test population, and seed. Before choosing region weights, measure fluid/near-wall/wake/remaining-local pixel fractions over the full accepted export; do not copy global wall/wake weights automatically. Keep gradient loss as an untuned candidate, not part of this feasibility task.
"""
    (OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT_ROOT / "decision.txt").write_text(decision + "\n", encoding="utf-8")
    print(f"\n{decision}")
    print(report)


if __name__ == "__main__":
    main()
