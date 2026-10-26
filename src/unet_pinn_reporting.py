"""Strict completed-run validation and explicitly weighted reporting utilities."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.unet_pinn_ab import compare_metadata, membership_sha256
from src.ml_models import build_model


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_run(root, profile, pilot=False):
    """Validate persisted evidence before comparing or evaluating a checkpoint."""
    root = Path(root)
    meta = load_json(root / "experiment_metadata.json")
    resolved = load_json(root / "resolved_config.json")
    history = load_json(root / "training_history.json")["epochs"]
    validation = load_json(root / "validation_metrics.json")
    test = load_json(root / "test_metrics.json")
    checkpoint = torch.load(root / "unet_pinn_airfoil.pt", map_location="cpu", weights_only=True)
    supported = ("unet_pinn_supervised", "unet_pinn_physics",
                 "unet_improved_supervised", "unet_improved_pinn")
    supervised_arm = profile.endswith("supervised")
    improved = profile.startswith("unet_improved_")
    expected_weights = dict(supervised_weight=1.0,
        continuity_weight=0.0 if supervised_arm else 0.1,
        momentum_weight=0.0 if supervised_arm else 0.1)
    def require(condition, message):
        if not condition:
            raise ValueError(f"{root}: {message}")
    require(profile in supported, "unknown profile")
    require(meta["experiment_name"] == profile and meta["output_name"] == profile, "profile identity mismatch")
    require(meta["pilot"] is pilot, "pilot/full identity mismatch")
    require(all(meta[k] == v for k, v in resolved.items()), "resolved config differs from metadata")
    require(meta["loss_weights"] == expected_weights, "unexpected A/B loss weights")
    tc = meta["training_config"]
    expected_tc = dict(batch_size=2, learning_rate=.001, weight_decay=0., optimizer="adam",
                       scheduler="none", seed=42, dataloader_seed=42)
    require(all(tc.get(k) == v for k, v in expected_tc.items()), "unexpected training settings")
    epochs = tc["epochs"]
    require((pilot and 0 < epochs <= 30) or (not pilot and epochs == 30), "expected exactly 30 full epochs")
    require([e["epoch"] for e in history] == list(range(1, epochs + 1)), "incomplete epoch history")
    best = min(history, key=lambda e: e["validation"]["supervised_loss"])["epoch"]
    require(meta["best_validation_epoch"] == best, "incorrect selected epoch")
    expected_selection = "improved_supervised_loss" if improved else "supervised_loss"
    require(meta["selection_metric"] == expected_selection, "incorrect selection policy")
    if improved:
        expected_supervision = dict(target_representation="nondimensional_perturbation",
            wall_weight=3.0, wake_weight=3.0, wall_distance_fraction=.05,
            wake_x_start=.9, wake_x_end=1.75, wake_half_height=.2,
            gradient_loss_weight=.1, gradient_channel_weights=[1.,1.,1.], p_inf=0.)
        require(meta.get("supervised_loss_config") == expected_supervision,
                "unexpected improved supervised configuration")
    require(meta["model_config"] == dict(in_channels=5, out_channels=3,
        encoder_channels=[32,64,128], bottleneck_channels=256), "unexpected architecture")
    require(meta["dataset"] == "pinn_dataset" and meta["model_name"] == "unet_pinn", "dataset/model identity mismatch")
    environment = meta["environment"]
    environment_keys = ("python_version", "pytorch_version", "cuda_version", "gpu_model",
                        "deterministic_algorithms", "cudnn_benchmark")
    require(all(k in environment for k in environment_keys), "incomplete runtime environment")
    if not pilot:
        require(environment.get("gpu_total_memory_bytes", 0) > 0, "missing total VRAM")
    require(environment["deterministic_algorithms"] is True and environment["cudnn_benchmark"] is False,
        "unexpected determinism settings")
    ids = []
    for key, count, fingerprint in (("train_case_ids",356,"train_fingerprint"),
            ("validation_case_ids",90,"validation_fingerprint"), ("fixed_test_case_ids",112,"fixed_test_fingerprint")):
        values = meta[key]
        require(len(values) == count and len(set(values)) == count, "invalid membership count/uniqueness")
        require(membership_sha256(values) == meta[fingerprint], "membership hash mismatch")
        ids.extend(values)
    require(len(set(ids)) == 558, "overlapping memberships")
    require(set(meta["development_case_ids"]) == set(meta["train_case_ids"] + meta["validation_case_ids"]), "invalid development membership")
    require(membership_sha256(meta["development_case_ids"]) == meta["development_fingerprint"], "development hash mismatch")
    require(membership_sha256(ids) == meta["dataset_membership_fingerprint"], "dataset membership hash mismatch")
    require(test["used_for_selection"] is False, "test used for selection")
    for metric in ("supervised_loss", "loss_p", "loss_ux", "loss_uy", "raw_continuity_loss",
                   "raw_momentum_x_loss", "raw_momentum_y_loss"):
        require(math.isfinite(validation[metric]) and validation[metric] >= 0, "invalid validation metrics")
        require(math.isclose(validation[metric], history[best-1]["validation"][metric], rel_tol=1e-6, abs_tol=1e-9), "selected validation/history mismatch")
        require(math.isfinite(test[metric]), "nonfinite test metric")
        require(all(math.isfinite(e["validation"][metric]) for e in history), "nonfinite history")
    checkpoint_identity_keys = ["model_name", "model_config", "loss_weights", "initial_state_hash",
        "dataset_fingerprint", "dataset_membership_fingerprint", "best_validation_epoch", "selection_metric"]
    if improved:
        checkpoint_identity_keys.append("supervised_loss_config")
        if "physics_operator" in meta:
            checkpoint_identity_keys.append("physics_operator")
    for key in checkpoint_identity_keys:
        # torch.save preserves tuples; JSON represents the same sequences as lists.
        require(json.dumps(checkpoint[key], sort_keys=True) == json.dumps(meta[key], sort_keys=True),
                f"checkpoint identity mismatch: {key}")
    for ck, mk in (("train_case_ids","train_case_ids"),("val_case_ids","validation_case_ids"),
                   ("test_case_ids","fixed_test_case_ids")):
        require(checkpoint[ck] == meta[mk], "checkpoint membership mismatch")
    require(checkpoint["optimizer_config"] == {"name":"adam", "scheduler":"none"}, "checkpoint optimizer mismatch")
    if "normalization" in meta:
        require(all(checkpoint.get(k) == v for k,v in meta["normalization"].items()),
                "checkpoint normalization mismatch")
    if improved and pilot:
        require(meta["finite_gradient_validation"] is True and
                meta["gradient_batches_checked"] == 178*epochs, "incomplete pilot gradient diagnostics")
        require(all(e["train"].get("finite_gradients") is True and
                    e["train"].get("gradient_batches_checked") == 178 for e in history),
                "incomplete pilot epoch gradient checks")
    require(bool(checkpoint["model_state_dict"]), "empty checkpoint state")
    require(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in checkpoint["model_state_dict"].values()), "checkpoint precision/nonfinite state")
    model = build_model("unet_pinn", **meta["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not pilot:
        require(meta["finite_gradient_validation"] is True and meta["gradient_batches_checked"] == 5340, "incomplete gradient diagnostics")
        require(all(e["train"]["finite_gradients"] is True and e["train"]["gradient_batches_checked"] == 178 for e in history), "incomplete epoch gradient checks")
        require(all(e["learning_rate"] == .001 for e in history), "learning rate changed")
        require(all(math.isfinite(e["elapsed_training_seconds"]) and e["elapsed_training_seconds"] > 0 for e in history), "invalid elapsed history")
        require(all(math.isfinite(v) for e in history for v in e["train"].values()), "nonfinite training history")
        for key in ("training_duration_seconds", "peak_cuda_memory_allocated_mib", "peak_cuda_memory_reserved_mib"):
            require(math.isfinite(meta[key]) and meta[key] > 0, f"missing diagnostic: {key}")
    return meta, history, validation, checkpoint


def validate_pair(left, right):
    problems = compare_metadata(left, right)
    for key in ("dataset_membership_fingerprint", "development_fingerprint", "environment",
                "dataset", "model_name", "pilot", "split_seed", "test_fraction", "validation_split"):
        if left.get(key) != right.get(key) or key not in left:
            problems.append(f"{key} differs or is missing")
    if problems:
        raise ValueError("Incompatible A/B runs: " + "; ".join(problems))


def pooled_mse(batch_mses, counts):
    """Combine batch masked MSEs by their actual number of valid cells."""
    if len(batch_mses) != len(counts) or not counts or sum(counts) <= 0:
        raise ValueError("Invalid pooled sufficient statistics")
    return math.fsum(m*c for m,c in zip(batch_mses, counts)) / sum(counts)


def plot_histories(histories, best_epochs, output, improved=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    specs = [("supervised_loss", "Supervised validation loss", False)]
    if improved:
        specs += [
            ("supervised_weighted_data_loss", "Region-weighted data loss", False),
            ("supervised_gradient_loss", "Gradient-matching loss", False),
        ]
    specs += [
        ("raw_continuity_loss", "Continuity", True),
        ("raw_momentum_x_loss", "Momentum-x", True),
        ("raw_momentum_y_loss", "Momentum-y", True),
    ]
    paths = []
    for key, title, rms in specs:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for label, color in (("supervised", "#2471A3"), ("physics_informed", "#C25125")):
            hist = histories[label]
            values = [math.sqrt(e["validation"][key]) if rms else e["validation"][key] for e in hist]
            ax.plot([e["epoch"] for e in hist], values, color=color, label=label.replace("_", " "), linewidth=1.8)
            best = best_epochs[label]
            ax.scatter([best], [values[best-1]], color=color, marker="*", s=130, zorder=5)
        ax.set(xlabel="Training epoch", ylabel="sqrt(mean batch masked MSE) [nondimensional]" if rms else "Mean batch normalized masked MSE",
               title=title + (" validation residual" if rms else ""), xlim=(1,30))
        ax.legend(); ax.grid(alpha=.22)
        path = output / (key + ".png")
        fig.savefig(path, dpi=240); plt.close(fig); paths.append(path.name)
    return paths
