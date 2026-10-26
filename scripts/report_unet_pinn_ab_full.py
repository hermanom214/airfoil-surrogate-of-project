"""Read-only validation reevaluation and report for two completed A/B runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ml_models import build_model
from src.unet_pinn import PinnFlowDataset, evaluation_mask, unet_pinn_losses
from src.unet_pinn_ab import configure_determinism, dataset_artifact_sha256
from src.unet_pinn_reporting import validate_run, validate_pair, pooled_mse, plot_histories


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(dataset, meta, checkpoint, device):
    for key in ("aoa_mean","aoa_std","inlet_u_mean","inlet_u_std","p_mean","p_std","ux_mean","ux_std","uy_mean","uy_std"):
        setattr(dataset, key, checkpoint[key])
    lookup = {p.parent.name:i for i,p in enumerate(dataset.files)}
    loader = DataLoader(Subset(dataset, [lookup[c] for c in meta["validation_case_ids"]]), batch_size=2, shuffle=False)
    model = build_model("unet_pinn", **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    norm = tuple(checkpoint[k] for k in ("p_mean","p_std","ux_mean","ux_std","uy_mean","uy_std"))
    vals = {k:[] for k in ("loss_p","loss_ux","loss_uy","raw_continuity_loss","raw_momentum_x_loss","raw_momentum_y_loss")}
    fluid_counts, physics_counts = [], []
    with torch.no_grad():
        for raw in loader:
            b = {k:v.to(device) if torch.is_tensor(v) and k != "omega" else v for k,v in raw.items()}
            pred = model(b["input"])
            losses = unet_pinn_losses(pred,b["target"],b["fluid_mask"],b["physics_mask"],b["xy"],b["k"],b["nut"],b["nu"],b["u_ref"],b["l_ref"],norm)
            for key in vals:
                vals[key].append(float(losses[key]))
            fluid_counts.append(int(b["fluid_mask"].sum()))
            physics_counts.append(int(evaluation_mask(b["physics_mask"]).sum()))
    pooled = {key:pooled_mse(value, fluid_counts if key.startswith("loss_") else physics_counts) for key,value in vals.items()}
    pooled["supervised_loss"] = sum(pooled[k] for k in ("loss_p","loss_ux","loss_uy"))/3
    for short, key in (("continuity_rms","raw_continuity_loss"),("momentum_x_rms","raw_momentum_x_loss"),("momentum_y_rms","raw_momentum_y_loss")):
        pooled[short] = math.sqrt(pooled[key])
    return {"aggregation":"valid-cell pooled masked MSE; residual RMS nondimensional", "case_count":90,
        "fluid_valid_cells":sum(fluid_counts),"physics_valid_cells":sum(physics_counts),
        "metrics":pooled,"batch_sufficient_statistics":{"metrics":vals,"fluid_counts":fluid_counts,"physics_counts":physics_counts}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output_dir}")
    roots=[Path("data/models/unet_pinn_supervised"),Path("data/models/unet_pinn_physics")]
    runs=[validate_run(root, profile) for root,profile in zip(roots,("unet_pinn_supervised","unet_pinn_physics"))]
    validate_pair(runs[0][0],runs[1][0])
    preserved={str(p):sha(p) for root in roots for p in root.rglob("*") if p.is_file()}
    split=Path("data/models/unet_pinn/fixed_test_split.json")
    preserved[str(split)]=sha(split)
    configure_determinism(42)
    dataset=PinnFlowDataset(Path("data/pinn_dataset"))
    fingerprint=dataset_artifact_sha256(dataset.files,dataset.root)
    if any(run[0]["dataset_fingerprint"] != fingerprint for run in runs):
        raise ValueError("Live dataset fingerprint differs")
    device=torch.device("cuda")
    labels=("supervised","physics_informed")
    pooled={label:evaluate(dataset,run[0],run[3],device) for label,run in zip(labels,runs)}
    if any(sha(p) != v for p,v in preserved.items()):
        raise RuntimeError("Input artifact changed during reporting")
    args.output_dir.mkdir(parents=True)
    def write(name,payload):
        (args.output_dir/name).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    write("validation_pooled_metrics.json",pooled)
    write("preserved_artifact_hashes.json",preserved)
    histories={label:run[1] for label,run in zip(labels,runs)}
    write("epoch_history.json",histories)
    write("resolved_configs.json",{label:run[0] for label,run in zip(labels,runs)})
    plot_paths=plot_histories(histories,{label:run[0]["best_validation_epoch"] for label,run in zip(labels,runs)},args.output_dir)
    with (args.output_dir/"epoch_history.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["experiment","epoch","train_supervised","validation_supervised","continuity_sqrt_mean_batch_MSE","momentum_x_sqrt_mean_batch_MSE","momentum_y_sqrt_mean_batch_MSE","learning_rate","elapsed_seconds"])
        for label,hist in histories.items():
            for e in hist:
                v=e["validation"]
                writer.writerow([label,e["epoch"],e["train"]["supervised_loss"],v["supervised_loss"],*[math.sqrt(v[k]) for k in ("raw_continuity_loss","raw_momentum_x_loss","raw_momentum_y_loss")],e["learning_rate"],e["elapsed_training_seconds"]])
    lines=["# Full Phase A U-Net PINN A/B comparison", "", "Both runs completed 30 epochs. Strict artifact and checkpoint compatibility validation passed. Experiment B had completed before recovery; no training restart was needed.","", "## Best supervised-validation checkpoints", "", "| Metric | Supervised A | Physics B | B − A | Change |", "|---|---:|---:|---:|---:|"]
    comparisons={}
    rows=[("Selected epoch",*[run[0]["best_validation_epoch"] for run in runs])]
    rows += [("Validation "+k+" (mean batch MSE)",runs[0][2][k],runs[1][2][k]) for k in ("supervised_loss","loss_p","loss_ux","loss_uy")]
    rows += [(k+" (pooled, nondimensional)",pooled[labels[0]]["metrics"][k],pooled[labels[1]]["metrics"][k]) for k in ("continuity_rms","momentum_x_rms","momentum_y_rms")]
    rows += [(k,*[run[0][k] for run in runs]) for k in ("training_duration_seconds","peak_cuda_memory_allocated_mib","peak_cuda_memory_reserved_mib")]
    for name,a,b in rows:
        delta=b-a; pct=100*delta/a if a else None
        comparisons[name]={"supervised":a,"physics_informed":b,"absolute_difference":delta,"percentage_difference":pct}
        lines.append(f"| {name} | {a:.9g} | {b:.9g} | {delta:+.9g} | {pct:+.3f}% |")
    write("selected_checkpoint_comparison.json",comparisons)
    lines += ["", "## Aggregation and interpretation", "", "Prediction losses above retain the original normalized mean-of-batch masked MSE used for selection. Historical physics curves are sqrt(mean batch masked MSE), nondimensional. Final physical residual RMS is computed by weighting each validation batch MSE by its actual valid-cell count; original experiment artifacts and histories are preserved. Pooled prediction losses are also available in validation_pooled_metrics.json.", "", "The test populations were excluded from optimization and checkpoint selection. Existing automatic test metrics are preserved separately in the run directories; no test reevaluation was performed.", "", "## History figures", ""]
    lines += [f"- [{p}]({p})" for p in plot_paths]
    lines += ["", "Stars mark each run's minimum supervised-validation epoch. One paired seed supports descriptive comparisons, not statistical significance.", "", "## Review findings and verification", "", "The comparator now rejects incomplete/full-versus-pilot runs, wrong profiles/weights, missing checkpoints and metrics, invalid selection epochs, incompatible environments, memberships, and checkpoint precision/identity. Pilot epoch overrides require pilot mode. Existing persisted split validation already covers duplicates, overlaps and coverage. No source physics equations or optimization behavior was changed.", "", "All input run artifacts, including pilot checkpoints and the fixed split, were hashed before and after reporting with identical hashes. The live dataset fingerprint matches both completed runs. No new commit was created."]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(comparisons,indent=2))


if __name__ == "__main__":
    main()
