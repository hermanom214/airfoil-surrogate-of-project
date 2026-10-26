"""Evaluate completed Phase A U-Net checkpoints on representative fixed-test fields."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_models import build_model
from src.ml_supervised import supervised_config_from_checkpoint
from src.unet_pinn import PinnFlowDataset
from src.unet_pinn_ab import dataset_artifact_sha256
from src.unet_pinn_field_evaluation import (
    PRESSURE_UNIT, VELOCITY_UNIT, assert_matching_normalization, case_metadata,
    decode_checkpoint_prediction, field_interpretation, summarize_field_metrics,
    file_sha256, masked_metrics, region_masks,
    select_representative_cases, shared_error_limit, shared_value_limits,
)
from src.unet_pinn_reporting import validate_pair, validate_run


ROOTS = {
    "supervised": PROJECT_ROOT / "data/models/unet_pinn_supervised",
    "physics": PROJECT_ROOT / "data/models/unet_pinn_physics",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/experiments/unet_pinn_ab_fields"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--improved", action="store_true", help="Select Phase A v1 improved checkpoints")
    parser.add_argument("--pilot", action="store_true", help="Evaluate completed bounded pilots")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--case-count", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (PROJECT_ROOT / "outputs/experiments/unet_improved_pinn_fields"
                           if args.improved else DEFAULT_OUTPUT)
        if args.pilot:
            args.output_dir = args.output_dir / "pilot"
    return args


def roi_plot_bounds(xy, roi):
    x, y = xy[..., 0][roi], xy[..., 1][roi]
    return (float(x.min()), float(x.max())), (float(y.min()), float(y.max()))


def plot_comparison(path, xy, mask, roi, reference, pred_a, pred_b, variable, unit, title):
    arrays = [reference, pred_a, pred_b, pred_a-reference, pred_b-reference]
    vmin, vmax = shared_value_limits(reference, roi)
    emax = shared_error_limit(arrays[3], arrays[4], roi)
    shown = [np.where(mask, value, np.nan) for value in arrays]
    extent = [float(xy[...,0].min()), float(xy[...,0].max()), float(xy[...,1].min()), float(xy[...,1].max())]
    xlim, ylim = roi_plot_bounds(xy, roi)
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), constrained_layout=True)
    labels = ("CFD reference", "Supervised", "Physics-informed", "Supervised - CFD", "Physics - CFD")
    for index, (ax, values, label) in enumerate(zip(axes, shown, labels)):
        image = ax.imshow(values, origin="lower", extent=extent, aspect="equal",
                          cmap="turbo" if index < 3 else "RdBu_r",
                          vmin=vmin if index < 3 else -emax, vmax=vmax if index < 3 else emax)
        ax.set(xlim=xlim, ylim=ylim, title=label, xlabel="x [m]")
        if index == 0: ax.set_ylabel("y [m]")
        else: ax.set_yticklabels([])
        fig.colorbar(image, ax=ax, shrink=.78, label=f"{variable} [{unit}]")
    fig.suptitle(title)
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return {"value_limits": [vmin, vmax], "error_limits": [-emax, emax], "x_limits": list(xlim), "y_limits": list(ylim)}


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    if not 6 <= args.case_count <= 8:
        raise ValueError("Representative evaluation requires 6 to 8 cases")

    profiles = {"supervised": "unet_improved_supervised", "physics": "unet_improved_pinn"} if args.improved else {
        "supervised": "unet_pinn_supervised", "physics": "unet_pinn_physics"}
    roots = {name: PROJECT_ROOT / "data/models" / profile for name, profile in profiles.items()}
    if args.pilot:
        roots = {name: root / "pilot" for name, root in roots.items()}
    if args.improved and (output == DEFAULT_OUTPUT.resolve() or DEFAULT_OUTPUT.resolve() in output.parents):
        raise ValueError("Improved field comparison requires a separate output directory")
    validated = {name: validate_run(roots[name], profile, args.pilot) for name, profile in profiles.items()}
    validate_pair(validated["supervised"][0], validated["physics"][0])
    checkpoints = {name: value[3] for name, value in validated.items()}
    normalization = assert_matching_normalization(checkpoints["supervised"], checkpoints["physics"])

    protected = [roots[name] / "unet_pinn_airfoil.pt" for name in roots]
    protected += [roots[name] / "split_membership.json" for name in roots]
    protected += [PROJECT_ROOT / "data/models/unet_pinn/fixed_test_split.json"]
    hashes_before = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in protected}

    dataset = PinnFlowDataset(PROJECT_ROOT / "data/pinn_dataset",
        supervised_config=supervised_config_from_checkpoint(checkpoints["supervised"]))
    for key, value in normalization.items(): setattr(dataset, key, value)
    live_fingerprint = dataset_artifact_sha256(dataset.files, dataset.root)
    if any(run[0]["dataset_fingerprint"] != live_fingerprint for run in validated.values()):
        raise ValueError("Live PINN dataset fingerprint differs from checkpoint metadata")
    by_id = {path.parent.name: path for path in dataset.files}
    fixed_ids = validated["supervised"][0]["fixed_test_case_ids"]
    candidates = [case_metadata(by_id[case_id]) for case_id in fixed_ids]
    selected = select_representative_cases(candidates, args.case_count)

    requested = args.device
    device = torch.device("cuda" if (requested == "cuda" or requested == "auto" and torch.cuda.is_available()) else "cpu")
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    models = {}
    for name, checkpoint in checkpoints.items():
        model = build_model("unet_pinn", **checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[name] = model.eval()

    for region in ("full_domain", "near_airfoil", "wake"):
        (output / "figures" / region).mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir(parents=True)

    rows, plot_metadata = [], {}
    units = {"p": PRESSURE_UNIT, "Ux": VELOCITY_UNIT, "Uy": VELOCITY_UNIT}
    reference_keys = {"p": lambda z: z["p"], "Ux": lambda z: z["U"][...,0], "Uy": lambda z: z["U"][...,1]}
    id_to_index = {path.parent.name: i for i, path in enumerate(dataset.files)}
    with torch.no_grad():
        for meta in selected:
            case_id = str(meta["case_id"]); index = id_to_index[case_id]
            sample = dataset[index]
            inp = sample["input"].unsqueeze(0).to(device)
            predictions = {name: decode_checkpoint_prediction(model(inp)[0].cpu().numpy(),
                checkpoints[name], float(meta["U_inf"])) for name, model in models.items()}
            with np.load(by_id[case_id], allow_pickle=False) as data:
                xy = data["xy"].astype(np.float64)
                mask = (data["fluid_mask"] > .5) & (data["interpolation_valid_mask"] > .5)
                refs = {key: np.asarray(loader(data), dtype=np.float64) for key, loader in reference_keys.items()}
            regions = region_masks(xy, mask, float(meta["chord"]), float(meta["aoa_deg"]))
            plot_metadata[case_id] = {}
            for variable in ("p", "Ux", "Uy"):
                for region_name in ("full_domain", "near_airfoil", "wake"):
                    roi = regions[region_name]
                    a_mae, a_rmse = masked_metrics(predictions["supervised"][variable], refs[variable], roi)
                    b_mae, b_rmse = masked_metrics(predictions["physics"][variable], refs[variable], roi)
                    delta_mae, delta_rmse = b_mae-a_mae, b_rmse-a_rmse
                    rows.append({"case_id":case_id,"NACA":meta["naca_code"],"AoA":meta["aoa_deg"],"U_inf":meta["U_inf"],
                        "region":region_name,"variable":variable,"unit":units[variable],"valid_cells":int(roi.sum()),
                        "supervised_MAE":a_mae,"physics_MAE":b_mae,"supervised_RMSE":a_rmse,"physics_RMSE":b_rmse,
                        "mae_absolute_difference":delta_mae,"mae_percentage_difference":100*delta_mae/a_mae if a_mae else None,
                        "rmse_absolute_difference":delta_rmse,"rmse_percentage_difference":100*delta_rmse/a_rmse if a_rmse else None,
                        "absolute_difference":delta_rmse,"percentage_difference":100*delta_rmse/a_rmse if a_rmse else None})
                    filename = f"{case_id}_{variable}.png"
                    plot_metadata[case_id][f"{region_name}_{variable}"] = plot_comparison(
                        output/"figures"/region_name/filename, xy, mask, roi, refs[variable],
                        predictions["supervised"][variable], predictions["physics"][variable], variable, units[variable],
                        f"{case_id} | NACA {meta['naca_code']} | AoA={float(meta['aoa_deg']):g} deg | U_inf={float(meta['U_inf']):g} m/s | {region_name.replace('_',' ')}")

    fieldnames = list(rows[0])
    with (output/"tables/selected_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    summary = summarize_field_metrics(rows)
    with (output/"tables/summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

    selected_payload={"selection_basis":"Deterministic metadata-only categorical coverage followed by normalized maximin coverage over camber, thickness, AoA, and U_inf; no predictions or errors used.",
        "population":"112-case persisted fixed-test split","case_count":len(selected),"cases":[{k:m[k] for k in ("case_id","naca_code","camber_percent","thickness_percent","aoa_deg","U_inf","inspection_ok","pinn_valid")} for m in selected]}
    (output/"selected_cases.json").write_text(json.dumps(selected_payload,indent=2),encoding="utf-8")
    (output/"plot_metadata.json").write_text(json.dumps(plot_metadata,indent=2),encoding="utf-8")
    (output/"checkpoint_verification.json").write_text(json.dumps({"device":str(device),"dataset_fingerprint":live_fingerprint,"normalization":normalization,
        "checkpoints":{name:{"best_validation_epoch":validated[name][0]["best_validation_epoch"],"initial_state_hash":validated[name][0]["initial_state_hash"],"model_config":validated[name][0]["model_config"]} for name in validated},"protected_hashes":hashes_before},indent=2),encoding="utf-8")
    hashes_after={str(path.relative_to(PROJECT_ROOT)):file_sha256(path) for path in protected}
    if hashes_before != hashes_after: raise RuntimeError("A protected checkpoint or split artifact changed")

    lines=["# Phase A U-Net field comparison","",f"Completed on {len(selected)} representative cases from the immutable 112-case fixed-test split using {device} inference.","",
        "## Selection","",selected_payload["selection_basis"],""]
    lines += [f"- `{m['case_id']}` — NACA {m['naca_code']}, AoA {float(m['aoa_deg']):g} deg, U_inf {float(m['U_inf']):g} m/s" for m in selected]
    lines += ["","## Region and mask definitions","","Metrics use only fluid cells with valid interpolation. The production supervised loss used the fluid mask; evaluation additionally excludes invalid interpolation cells. Near-airfoil ROI: chord-aligned -0.20 <= x/c <= 1.20 and |y|/c <= 0.30. Wake ROI: chord-aligned 0.90 <= x/c <= 1.75 and |y|/c <= 0.20. Coordinates are rotated by the case AoA convention about quarter chord.","",
        "## Results","","Negative B-A means physics-informed improved. This representative subset does not characterize the full 112-case test population.","",
        "| Region | Variable | Metric | Mean supervised | Mean physics | Improved | Worsened | Mean B-A | Median B-A |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary: lines.append(f"| {s['region']} | {s['variable']} | {s['metric']} | {s['mean_supervised_error']:.6g} | {s['mean_physics_error']:.6g} | {s['physics_improved_cases']} | {s['physics_worsened_cases']} | {s['mean_B_minus_A']:.6g} | {s['median_B_minus_A']:.6g} |")
    lines += ["", "## Interpretation", "", *field_interpretation(summary)]
    lines += ["","## Artifacts","","- `figures/full_domain/`: CFD, supervised, physics-informed, and shared-scale signed errors for p, Ux, Uy.","- `figures/near_airfoil/`: identical five-panel comparisons in the near-airfoil ROI.","- `figures/wake/`: identical five-panel comparisons in the chord-aligned wake ROI.","- `tables/selected_case_metrics.csv`: per-case physical-unit MAE/RMSE.","- `tables/summary_metrics.csv`: per-region and per-variable case summary.","","Pressure is OpenFOAM kinematic pressure in m^2/s^2, not Pa. Shared prediction limits use the 1st–99th percentiles of the CFD reference within each displayed ROI; both signed-error panels use the same symmetric 99th-percentile magnitude from both models.","",
        "## Validation","",f"Selected checkpoint epochs: supervised {validated['supervised'][0]['best_validation_epoch']}, physics-informed {validated['physics'][0]['best_validation_epoch']}. Architecture, experiment identity, initialization hash, dataset fingerprint, split membership, checkpoint precision, and persisted training normalization passed strict validation. Protected checkpoint and split hashes were unchanged. No training or full-test recomputation was performed."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"selected_cases":[m["case_id"] for m in selected],"device":str(device)},indent=2))


if __name__ == "__main__": main()

