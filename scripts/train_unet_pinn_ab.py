"""Run one controlled Phase-A U-Net A/B experiment (never both implicitly)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_ml_models_config, load_paths
from src.ml_data_split import build_holdout_fold, load_or_create_fixed_test_split
from src.ml_device import loader_device_kwargs, resolve_device
from src.ml_models import build_model
from src.unet_pinn import PinnFlowDataset, unet_pinn_losses
from src.unet_pinn_ab import (ALL_AB_EXPERIMENTS, configure_determinism,
    cuda_peak_memory, dataset_artifact_sha256, environment_metadata, membership_sha256,
    seed_worker, state_dict_sha256, validate_finite_gradients, write_json)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=ALL_AB_EXPERIMENTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--pilot", action="store_true", help="Use configured bounded pilot epochs and an isolated pilot directory.")
    parser.add_argument("--pilot-epochs", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.pilot_epochs is not None and not args.pilot:
        parser.error("--pilot-epochs requires --pilot")
    return args


def resolve_experiment(cfg, name: str, pilot: bool, pilot_epochs: int | None) -> dict[str, Any]:
    if pilot_epochs is not None and not pilot:
        raise ValueError("--pilot-epochs requires --pilot")
    exp = cfg.experiment_for(name)
    raw = exp.training_overrides
    required = {"batch_size", "epochs", "learning_rate", "weight_decay", "optimizer",
                "scheduler", "seed", "dataloader_seed", "pilot_epochs"}
    missing = required.difference(raw)
    if missing or set(exp.pinn_loss_weights) != {"supervised_weight", "continuity_weight", "momentum_weight"}:
        raise RuntimeError(f"Incomplete controlled experiment configuration: {sorted(missing)}")
    epochs = int(pilot_epochs if pilot_epochs is not None else raw["pilot_epochs"] if pilot else raw["epochs"])
    if epochs <= 0 or (pilot and epochs > int(raw["epochs"])):
        raise ValueError("Pilot epochs must be positive and no greater than full epochs")
    if str(raw["optimizer"]).lower() != "adam" or str(raw["scheduler"]).lower() != "none":
        raise RuntimeError("Controlled Phase-A runner currently supports optimizer=adam and scheduler=none")
    return {
        "experiment_name": name, "output_name": exp.output_name, "dataset": exp.dataset,
        "model_name": "unet_pinn", "model_config": asdict(cfg.model.params.unet_pinn),
        "training_config": {k: raw[k] for k in required if k != "pilot_epochs"} | {"epochs": epochs},
        "loss_weights": dict(exp.pinn_loss_weights),
        "supervised_loss_config": exp.supervised_loss.metadata(),
        "selection_metric": exp.selection_metric,
        "validation_split": cfg.training.validation_split, "test_fraction": exp.data_split.test_fraction,
        "split_seed": exp.data_split.seed, "pilot": pilot,
    }


def _move(batch, device):
    return {k: (v.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(v) and k != "omega" else v)
            for k, v in batch.items()}


def run_loader(model, loader, dataset, device, weights, supervised_config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    gradient_batches_checked = 0
    pooled: dict[str, float] = {}
    case_values: dict[str, list[float]] = {}
    norm = (dataset.p_mean, dataset.p_std, dataset.ux_mean, dataset.ux_std,
            dataset.uy_mean, dataset.uy_std)
    calculate_physics = (not training or weights["continuity_weight"] != 0.0
                         or weights["momentum_weight"] != 0.0)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = _move(raw, device)
            pred = model(batch["input"])
            losses = unet_pinn_losses(pred, batch["target"],
                batch["fluid_mask"], batch["physics_mask"], batch["xy"], batch["k"],
                batch["nut"], batch["nu"], batch["u_ref"], batch["l_ref"], norm,
                weights["supervised_weight"], weights["continuity_weight"],
                weights["momentum_weight"], compute_physics=calculate_physics,
                supervised_config=supervised_config, regions=batch.get("regions"))
            if not torch.isfinite(losses["loss_total"]):
                raise RuntimeError("Non-finite U-Net PINN loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                validate_finite_gradients(model)
                gradient_batches_checked += 1
                optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            with torch.no_grad():
                if supervised_config.target_representation == "nondimensional_perturbation":
                    scale = batch["u_ref"].view(-1, 1, 1, 1)
                    def physical(values):
                        return torch.cat((values[:, 0:1] * (.5 * scale.square()) + supervised_config.p_inf,
                            scale * (1 + values[:, 1:2]), scale * values[:, 2:3]), 1)
                    pred_physical, target_physical = physical(pred), physical(batch["target"])
                else:
                    means = pred.new_tensor(norm[::2]).view(1, 3, 1, 1)
                    stds = pred.new_tensor(norm[1::2]).view(1, 3, 1, 1)
                    pred_physical, target_physical = pred * stds + means, batch["target"] * stds + means
                error = pred_physical - target_physical
                masks = {"full": batch["fluid_mask"]}
                if "regions" in batch:
                    masks.update(near_airfoil=batch["regions"][:, 0:1], wake=batch["regions"][:, 1:2])
                for region, mask in masks.items():
                    for channel, field in enumerate(("p", "ux", "uy")):
                        values = error[:, channel:channel+1]
                        count = float(mask.sum())
                        prefix = f"{region}_{field}"
                        pooled[prefix + "_abs_sum"] = pooled.get(prefix + "_abs_sum", 0.) + float((values.abs()*mask).sum())
                        pooled[prefix + "_sq_sum"] = pooled.get(prefix + "_sq_sum", 0.) + float((values.square()*mask).sum())
                        pooled[prefix + "_count"] = pooled.get(prefix + "_count", 0.) + count
                        per_case = ((values.abs()*mask).flatten(1).sum(1) /
                                    mask.flatten(1).sum(1).clamp_min(1))
                        case_values.setdefault(prefix + "_mae", []).extend(per_case.cpu().tolist())
    metrics = {key: value / len(loader) for key, value in totals.items()}
    metrics["physics_computed"] = calculate_physics
    for prefix in sorted({key.rsplit("_", 2)[0] for key in pooled if key.endswith("_abs_sum")}):
        count = pooled[prefix + "_count"]
        metrics[f"pooled_{prefix}_mae"] = pooled[prefix + "_abs_sum"] / count
        metrics[f"pooled_{prefix}_rmse"] = (pooled[prefix + "_sq_sum"] / count) ** .5
        values = case_values[prefix + "_mae"]
        metrics[f"case_mean_{prefix}_mae"] = sum(values) / len(values)
    if training:
        metrics["gradient_batches_checked"] = gradient_batches_checked
        metrics["finite_gradients"] = True
    return metrics


def main(argv=None):
    args = parse_args(argv)
    cfg = load_ml_models_config(ROOT / "configs" / "ml_models_config.yaml")
    paths = load_paths(ROOT / "configs" / "paths.yaml")
    resolved = resolve_experiment(cfg, args.experiment, args.pilot, args.pilot_epochs)
    if resolved["dataset"] != "pinn_dataset" or resolved["selection_metric"] not in {
        "supervised_loss", "improved_supervised_loss"
    }:
        raise RuntimeError("Controlled experiment requires pinn_dataset and a supervised selection metric")
    tc = resolved["training_config"]
    seed = int(tc["seed"])
    configure_determinism(seed)
    device = resolve_device(args.device or cfg.device)
    exp = cfg.experiment_for(args.experiment)
    improved = resolved["selection_metric"] == "improved_supervised_loss"
    dataset = PinnFlowDataset(paths.project_root / "data" / "pinn_dataset",
        supervised_config=exp.supervised_loss, return_regions=improved)
    case_ids = [path.parent.name for path in dataset.files]
    fixed_path = paths.project_root / cfg.output.models_subdir / "unet_pinn" / "fixed_test_split.json"
    if not fixed_path.exists():
        raise FileNotFoundError(f"Existing fixed-test split is required: {fixed_path}")
    split = load_or_create_fixed_test_split(case_ids, fixed_path, resolved["test_fraction"], resolved["split_seed"])
    fold = build_holdout_fold(split.development_indices, case_ids, resolved["validation_split"], resolved["split_seed"])
    if (len(case_ids), len(fold.train_indices), len(fold.val_indices), len(split.test_indices)) != (558,356,90,112):
        raise RuntimeError("Controlled experiment requires 558 cases and 356/90/112 membership")
    output = paths.project_root / cfg.output.models_subdir / str(resolved["output_name"])
    if args.pilot:
        output = output / "pilot"
    if output.exists():
        existing = list(output.iterdir())
        full_run_with_only_preserved_pilot = not args.pilot and all(
            item.name == "pilot" and item.is_dir() for item in existing
        )
        if not full_run_with_only_preserved_pilot:
            raise FileExistsError(f"Refusing to overwrite experiment output: {output}")
    dataset.fit_condition_normalization(fold.train_indices)
    dataset.fit_target_normalization(fold.train_indices)
    metadata = {
        **resolved, "dataset_fingerprint": dataset_artifact_sha256(dataset.files, dataset.root),
        "physics_operator": "conservative_continuum_modified_pressure_v1",
        "normalization": {k: float(getattr(dataset, k)) for k in (
            "aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std", "p_mean", "p_std",
            "ux_mean", "ux_std", "uy_mean", "uy_std")},
        "dataset_membership_fingerprint": split.dataset_fingerprint,
        "development_fingerprint": membership_sha256(split.development_case_ids),
        "train_fingerprint": membership_sha256(fold.train_case_ids),
        "validation_fingerprint": membership_sha256(fold.val_case_ids),
        "fixed_test_fingerprint": membership_sha256(split.test_case_ids),
        "development_case_ids": split.development_case_ids, "train_case_ids": fold.train_case_ids,
        "validation_case_ids": fold.val_case_ids, "fixed_test_case_ids": split.test_case_ids,
        "development_count": len(split.development_case_ids), "train_count": len(fold.train_case_ids),
        "validation_count": len(fold.val_case_ids), "fixed_test_count": len(split.test_case_ids),
    }
    configure_determinism(seed)
    model = build_model("unet_pinn", **resolved["model_config"])
    initial_state = copy.deepcopy(model.state_dict())
    metadata["initial_state_hash"] = state_dict_sha256(initial_state)
    verifier = build_model("unet_pinn", **resolved["model_config"])
    verifier.load_state_dict(initial_state)
    if state_dict_sha256(verifier.state_dict()) != metadata["initial_state_hash"]:
        raise RuntimeError("Initial parameter state verification failed")
    metadata["environment"] = environment_metadata(device)
    print(json.dumps({k: v for k, v in metadata.items() if not k.endswith("case_ids")}, indent=2))
    if args.preflight_only:
        return metadata

    common = dict(batch_size=int(tc["batch_size"]), **loader_device_kwargs(device))
    generator = torch.Generator().manual_seed(int(tc["dataloader_seed"]))
    train_loader = DataLoader(Subset(dataset, fold.train_indices), shuffle=True, generator=generator,
                              worker_init_fn=seed_worker, **common)
    val_loader = DataLoader(Subset(dataset, fold.val_indices), shuffle=False, **common)
    test_loader = DataLoader(Subset(dataset, split.test_indices), shuffle=False, **common)
    model.load_state_dict(initial_state)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]),
                                 weight_decay=float(tc["weight_decay"]))
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, int(tc["epochs"]) + 1):
        train_metrics = run_loader(model, train_loader, dataset, device, resolved["loss_weights"], exp.supervised_loss, optimizer)
        val_metrics = run_loader(model, val_loader, dataset, device, resolved["loss_weights"], exp.supervised_loss)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_training_seconds": time.perf_counter() - started,
                        "train": train_metrics, "validation": val_metrics})
        if val_metrics["supervised_loss"] < best_loss:
            best_loss = val_metrics["supervised_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch={epoch} validation_supervised_loss={val_metrics['supervised_loss']:.8e}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    validation_metrics = run_loader(model, val_loader, dataset, device, resolved["loss_weights"], exp.supervised_loss)
    test_metrics = run_loader(model, test_loader, dataset, device, resolved["loss_weights"], exp.supervised_loss)
    memory = cuda_peak_memory(device)
    metadata.update({"best_validation_epoch": best_epoch, "training_duration_seconds": duration,
                     "configured_batch_size": int(tc["batch_size"]),
                     "finite_gradient_validation": True,
                     "gradient_batches_checked": sum(item["train"]["gradient_batches_checked"] for item in history),
                     **memory,
                     # Backward-compatible name retained for existing pilot/report consumers.
                     "peak_cuda_memory_bytes": memory["peak_cuda_memory_allocated_bytes"]})
    # The collision guard above permits an existing parent only when it contains
    # the preserved pilot directory and no full-run artifacts.
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "experiment_metadata.json", metadata)
    write_json(output / "split_membership.json", {k: metadata[k] for k in (
        "dataset_fingerprint", "dataset_membership_fingerprint", "development_fingerprint", "train_fingerprint",
        "validation_fingerprint", "fixed_test_fingerprint", "development_case_ids",
        "train_case_ids", "validation_case_ids", "fixed_test_case_ids")})
    write_json(output / "training_history.json", {"epochs": history})
    write_json(output / "validation_metrics.json", validation_metrics)
    write_json(output / "test_metrics.json", {"used_for_selection": False, **test_metrics})
    checkpoint = {"model_name": "unet_pinn", "model_config": resolved["model_config"],
        "model_state_dict": best_state, "optimizer_config": {"name": "adam", "scheduler": "none"},
        "selection_metric": resolved["selection_metric"], "best_validation_epoch": best_epoch,
        "initial_state_hash": metadata["initial_state_hash"], "dataset_fingerprint": metadata["dataset_fingerprint"],
        "dataset_membership_fingerprint": split.dataset_fingerprint,
        "train_case_ids": fold.train_case_ids, "val_case_ids": fold.val_case_ids,
        "development_case_ids": split.development_case_ids, "test_case_ids": split.test_case_ids,
        "loss_weights": resolved["loss_weights"],
        "physics_operator": metadata["physics_operator"],
        "target_representation": exp.supervised_loss.target_representation,
        "supervised_loss_config": resolved["supervised_loss_config"], **{k: float(getattr(dataset, k)) for k in (
            "aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std", "p_mean", "p_std",
            "ux_mean", "ux_std", "uy_mean", "uy_std")}}
    torch.save(checkpoint, output / "unet_pinn_airfoil.pt")
    return metadata


if __name__ == "__main__":
    main()
