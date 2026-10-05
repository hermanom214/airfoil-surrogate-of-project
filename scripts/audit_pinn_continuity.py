from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.continuity_audit import (  # noqa: E402
    STAT_NAMES, continuity_residual, inspect_uniform_xy, nondimensionalize_residual,
    region_masks, residual_stats, stencil_valid_mask,
)

DATASET_ROOT = PROJECT_ROOT / "data" / "pinn_dataset"
OUTPUT_ROOT = DATASET_ROOT / "continuity_audit"
REGIONS = ("near_wall", "intermediate", "outer", "wake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only continuity audit of the PINN dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace only prior continuity-audit outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Development/testing case limit.")
    return parser.parse_args()


def flatten_stats(row: dict, prefix: str, stats: dict) -> None:
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value


def summarize_numbers(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def grouped(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: {"case_count": len(items), "nondim_rms": summarize_numbers([r["nd_rms"] for r in items]),
                   "nondim_mean_abs": summarize_numbers([r["nd_mean_abs"] for r in items])}
            for name, items in sorted(groups.items())}


def plot_outputs(rows: list[dict], output: Path) -> None:
    figdir = output / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    rms = np.asarray([r["nd_rms"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(rms, bins=35, edgecolor="white")
    ax.set(xlabel="Per-case RMS of R*", ylabel="Cases", title="Continuity residual distribution")
    fig.tight_layout(); fig.savefig(figdir / "rms_histogram.png", dpi=170); plt.close(fig)

    region_values = [[r[f"{region}_rms"] for r in rows] for region in REGIONS]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.boxplot(region_values, tick_labels=REGIONS, showfliers=False)
    ax.set_yscale("log"); ax.set(ylabel="Per-case RMS of R*", title="Continuity residual by region")
    fig.tight_layout(); fig.savefig(figdir / "region_boxplots.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter([r["aoa_deg"] for r in rows], rms, s=12, alpha=.55)
    ax.set_yscale("log"); ax.set(xlabel="Angle of attack [deg]", ylabel="RMS of R*", title="Residual versus angle of attack")
    fig.tight_layout(); fig.savefig(figdir / "residual_vs_aoa.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); sc = ax.scatter([r["Re"] for r in rows], rms, c=[r["aoa_deg"] for r in rows], s=14, alpha=.65, cmap="coolwarm")
    ax.set_yscale("log"); ax.set(xlabel="Re", ylabel="RMS of R*", title="Residual versus Reynolds number")
    fig.colorbar(sc, ax=ax, label="AoA [deg]"); fig.tight_layout(); fig.savefig(figdir / "residual_vs_re.png", dpi=170); plt.close(fig)


def spatial_maps(selected: list[dict], output: Path) -> None:
    loaded = []
    for row in selected:
        with np.load(row["path"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item())); xy=z["xy"]; mask=z["physics_valid_mask"].astype(bool)
            r = nondimensionalize_residual(continuity_residual(z["U"], xy, mask), float(meta["chord"]), float(meta["U_inf"]))
            loaded.append((row, xy, z["airfoil_sdf"].copy(), r))
    all_values = np.concatenate([np.abs(item[3][np.isfinite(item[3])]) for item in loaded])
    clip = float(np.percentile(all_values, 99))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.8), constrained_layout=True)
    last = None
    for label, ax, (row, xy, sdf, residual) in zip(("low", "median", "high"), axes, loaded):
        last = ax.pcolormesh(xy[...,0], xy[...,1], residual, shading="auto", cmap="RdBu_r", vmin=-clip, vmax=clip)
        ax.contour(xy[...,0], xy[...,1], sdf, levels=[0], colors="black", linewidths=.7)
        ax.set_aspect("equal"); ax.set_ylabel("y [m]"); ax.set_title(f"{label}: {row['case_id']} (RMS={row['nd_rms']:.4g})", fontsize=9)
    axes[-1].set_xlabel("x [m]"); fig.colorbar(last, ax=axes, label="R* (clipped at pooled 99th |R*| percentile)")
    fig.savefig(output / "figures" / "representative_spatial_maps.png", dpi=180); plt.close(fig)
    return clip


def main() -> None:
    args = parse_args(); dataset = args.dataset_root.resolve(); output = args.output_root.resolve()
    if dataset == output or dataset not in output.parents:
        raise ValueError("output-root must be a dedicated child of dataset-root")
    existing = [output / "per_case_metrics.csv", output / "aggregate_summary.json", output / "figures"]
    if any(p.exists() for p in existing) and not args.overwrite:
        raise FileExistsError("Audit outputs exist; pass --overwrite to replace only those outputs")
    output.mkdir(parents=True, exist_ok=True); (output / "figures").mkdir(exist_ok=True)
    paths = sorted((dataset / "cases").glob("*/*.npz"))
    if args.limit is not None: paths = paths[:args.limit]
    if not paths: raise FileNotFoundError("No case NPZ files found")
    rows: list[dict] = []; grid_reference = None; mask_failures = []
    pooled = {name: {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0, "max_abs": 0.0} for name in ("dim", "nd", *REGIONS)}
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item())); xy=z["xy"]; info=inspect_uniform_xy(xy)
            signature=(info.ny,info.nx,info.dx,info.dy,info.x_min,info.x_max,info.y_min,info.y_max)
            if grid_reference is None: grid_reference=signature
            if not np.allclose(signature, grid_reference, rtol=2e-5, atol=1e-9): raise ValueError(f"grid mismatch: {path}")
            base=z["fluid_mask"].astype(bool) & z["interpolation_valid_mask"].astype(bool)
            physics=z["physics_valid_mask"].astype(bool); required2=stencil_valid_mask(base,2); required4=stencil_valid_mask(base,4)
            bad2=int(np.count_nonzero(physics & ~required2)); bad4=int(np.count_nonzero(physics & ~required4))
            if bad2 or bad4: mask_failures.append({"case_id":meta["case_id"],"order2":bad2,"order4":bad4})
            rdim=continuity_residual(z["U"],xy,physics,2); rnd=nondimensionalize_residual(rdim,float(meta["chord"]),float(meta["U_inf"]))
            row={"case_id":meta["case_id"],"path":str(path),"naca_code":meta["naca_code"],"aoa_deg":float(meta["aoa_deg"]),"U_inf":float(meta["U_inf"]),"chord":float(meta["chord"]),"nu":float(meta["nu"]),"Re":float(meta["Re"]),"physics_mask_count":int(physics.sum()),"order2_unsafe_count":bad2,"order4_unsafe_count":bad4}
            flatten_stats(row,"dim",residual_stats(rdim)); flatten_stats(row,"nd",residual_stats(rnd))
            regions=region_masks(xy,z["airfoil_sdf"],physics,float(meta["chord"]),float(meta["aoa_deg"]))
            for region, mask in regions.items(): flatten_stats(row,region,residual_stats(rnd[mask]))
            rows.append(row)
            for name, vals in (("dim",rdim),("nd",rnd),*((region,rnd[mask]) for region,mask in regions.items())):
                vals=np.asarray(vals); vals=vals[np.isfinite(vals)]; a=np.abs(vals); p=pooled[name]
                p["count"]+=int(vals.size); p["sum_abs"]+=float(a.sum()); p["sum_sq"]+=float(np.dot(vals,vals)); p["max_abs"]=max(p["max_abs"],float(a.max(initial=0)))
        if index % 25 == 0 or index == len(paths): print(f"[{index}/{len(paths)}] audited")
    if mask_failures: raise RuntimeError(f"Stored physics mask is unsafe: {mask_failures[:3]}")
    for p in pooled.values():
        p["mean_abs"]=p.pop("sum_abs")/p["count"]; p["rms"]=(p.pop("sum_sq")/p["count"])**.5
    csv_rows=[{k:v for k,v in row.items() if k != "path"} for row in rows]
    with (output/"per_case_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    ordered=sorted(rows,key=lambda r:r["nd_rms"]); selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    comparison=[]
    for row in selected:
        with np.load(row["path"],allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"].item())); mask=z["physics_valid_mask"].astype(bool)
            r2=nondimensionalize_residual(continuity_residual(z["U"],z["xy"],mask,2),meta["chord"],meta["U_inf"])
            r4=nondimensionalize_residual(continuity_residual(z["U"],z["xy"],mask,4),meta["chord"],meta["U_inf"])
            common=np.isfinite(r2)&np.isfinite(r4); comparison.append({"case_id":row["case_id"],"order2":residual_stats(r2[common]),"order4":residual_stats(r4[common]),"difference":residual_stats((r4-r2)[common])})
    plot_outputs(rows,output); clip=spatial_maps(selected,output)
    region_summary={region:{"pooled":pooled[region],"case_rms":summarize_numbers([r[f"{region}_rms"] for r in rows])} for region in REGIONS}
    summary={"case_count":len(rows),"method":{"main":"second-order centered differences","dimensional_units":"1/s","nondimensionalization":"R*=chord/U_inf*R_dim","axis_mapping":"array axis 0=y; axis 1=x; U channel 0=Ux; channel 1=Uy"},"grid":{"shape":[grid_reference[0],grid_reference[1]],"dx":grid_reference[2],"dy":grid_reference[3],"extent":{"x":[grid_reference[4],grid_reference[5]],"y":[grid_reference[6],grid_reference[7]]}},"mask_validation":{"failures":mask_failures,"all_safe_for_order2":True,"all_safe_for_order4":True},"pooled":{"dimensional":pooled["dim"],"nondimensional":pooled["nd"]},"regions":region_summary,"groups":{"naca_code":grouped(rows,"naca_code"),"aoa_deg":grouped(rows,"aoa_deg"),"U_inf":grouped(rows,"U_inf"),"Re":grouped(rows,"Re")},"representative_cases":[{k:r[k] for k in ("case_id","nd_rms","naca_code","aoa_deg","U_inf","Re")} for r in selected],"order2_vs_order4":comparison,"map_symmetric_clip_abs_rstar_p99":clip,"region_definitions":{"near_wall":"0<SDF/chord<=0.02","intermediate":"0.02<SDF/chord<=0.10","outer":"SDF/chord>0.10","wake":"overlapping: x/chord downstream of rotated trailing edge through 1.75 and |y/chord-y_TE/chord|<=0.25"}}
    (output/"aggregate_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Wrote audit to {output}")


if __name__ == "__main__": main()
