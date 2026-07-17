# Training script for the SimpleUNet surrogate model.
# Loads pre-processed airfoil flow-field data (.npz files), splits it into
# training and validation sets, trains a U-Net-based neural network to predict
# CFD flow fields from geometry/condition inputs, and saves the resulting model
# weights to disk.

from __future__ import annotations
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow imports from the project root (src/ package)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_ml_models_config, load_paths
from src.ml_data_split import build_train_val_split
from src.ml_clcd_dataset import AirfoilClCdDataset
from src.ml_dataset import AirfoilFlowDataset
from src.ml_models import build_model
from src.ml_training import (
    compute_grid_spacing_from_xy,
    evaluate,
    train_one_epoch,
    train_one_epoch_physics,
)
from src.ml_validation import plot_loss_curves


# --- Configuration -----------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
ML_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ml_models_config.yaml"
PATHS = load_paths(CONFIG_PATH)  # Load project paths from YAML config
ML_CFG = load_ml_models_config(ML_CONFIG_PATH)

DATA_DIR = PATHS.flow_fields_output           # Directory with .npz flow-field files
MODEL_NAME = ML_CFG.model.name
MODEL_OUTPUT_FILENAME = ML_CFG.output.filename_template.format(model_name=MODEL_NAME)
OUTPUT_PATH = PATHS.project_root / ML_CFG.output.models_subdir / MODEL_OUTPUT_FILENAME

CLCD_FEATURE_ORDER = [
    "camber_percent",
    "camber_position_tenths",
    "thickness_percent",
    "aoa_deg",
    "inlet_velocity",
]
CLCD_TARGET_ORDER = ["Cl", "Cd"]


def _load_reference_grid_spacing(data_dir: Path) -> tuple[float, float]:
    files = sorted(data_dir.glob("*_flow.npz"))
    if not files:
        raise RuntimeError(f"No *_flow.npz files found in {data_dir}")

    with np.load(files[0], allow_pickle=False) as data:
        xy = torch.from_numpy(data["xy"].astype(np.float32))
    return compute_grid_spacing_from_xy(xy)


def _train_clcd_mlp(device: str) -> None:
    clcd_cfg = ML_CFG.model.params.clcd_mlp
    cases_root = PATHS.openfoam_case_sim / clcd_cfg.cases_subdir
    dataset = AirfoilClCdDataset(
        cases_root=cases_root,
        tail_window=clcd_cfg.tail_window,
        force_coeffs_relpath=clcd_cfg.force_coeffs_relpath,
    )
    print(f"[INFO] Cl/Cd cases root: {cases_root}")
    print(f"[INFO] Cl/Cd samples: {len(dataset)}")
    print(f"[INFO] Valid samples: {len(dataset.samples)}")
    print(f"[INFO] Skipped samples: {len(dataset.skipped_samples)}")

    raw_targets = np.stack([s[1] for s in dataset.samples], axis=0)
    cl_values = raw_targets[:, 0]
    cd_values = raw_targets[:, 1]
    print(
        "[INFO] Target summary before normalization | "
        f"Cl min/max/mean/std: {float(cl_values.min()):.6e}/"
        f"{float(cl_values.max()):.6e}/"
        f"{float(cl_values.mean()):.6e}/"
        f"{float(cl_values.std()):.6e} | "
        f"Cd min/max/mean/std: {float(cd_values.min()):.6e}/"
        f"{float(cd_values.max()):.6e}/"
        f"{float(cd_values.mean()):.6e}/"
        f"{float(cd_values.std()):.6e}"
    )

    print("[INFO] First five samples:")
    for features, targets, case_name in dataset.samples[:5]:
        print(f"  Case: {case_name}")
        print(f"  features: {features.tolist()}")
        print(f"  targets: {targets.tolist()}")

    split = build_train_val_split(
        dataset=dataset,
        validation_split=ML_CFG.training.validation_split,
        split_seed=ML_CFG.training.split_seed,
    )
    train_set = split.train_set
    val_set = split.val_set
    train_indices = split.train_indices
    val_indices = split.val_indices

    dataset.fit_normalization(train_indices)
    print(
        "[INFO] Feature normalization | "
        f"{dataset.feature_mean.tolist()} / {dataset.feature_std.tolist()}"
    )
    print(
        "[INFO] Target normalization (Cl, Cd) | "
        f"{dataset.target_mean.tolist()} / {dataset.target_std.tolist()}"
    )

    train_loader = DataLoader(train_set, batch_size=ML_CFG.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=ML_CFG.training.batch_size, shuffle=False)

    model_kwargs = {
        "input_dim": clcd_cfg.input_dim,
        "hidden_dims": clcd_cfg.hidden_dims,
        "output_dim": clcd_cfg.output_dim,
        "dropout": clcd_cfg.dropout,
    }
    model = build_model(MODEL_NAME, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ML_CFG.training.learning_rate)
    criterion = torch.nn.MSELoss()

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, ML_CFG.training.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            loss = criterion(preds, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss detected.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                preds = model(features)
                loss = criterion(preds, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss detected.")
                val_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6e} | "
            f"val loss: {val_loss:.6e}"
        )

    if not np.isfinite(np.asarray(train_history, dtype=np.float64)).all():
        raise RuntimeError("Non-finite values detected in train_history. Refusing to save model.")
    if not np.isfinite(np.asarray(val_history, dtype=np.float64)).all():
        raise RuntimeError("Non-finite values detected in val_history. Refusing to save model.")

    checkpoint = {
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
        "target_mean": dataset.target_mean.tolist(),
        "target_std": dataset.target_std.tolist(),
        "feature_order": CLCD_FEATURE_ORDER,
        "target_order": CLCD_TARGET_ORDER,
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
    }
    torch.save(checkpoint, OUTPUT_PATH)
    print("[INFO] Final verification passed:")
    print("[INFO] - No NaN entered normalization")
    print("[INFO] - No NaN entered training")
    print("[INFO] - No corrupted model was saved")
    print(f"[INFO] Total valid samples: {len(dataset.samples)}")
    print(f"[INFO] Total skipped samples: {len(dataset.skipped_samples)}")
    print(f"[INFO] Train samples: {len(train_set)}")
    print(f"[INFO] Validation samples: {len(val_set)}")
    print(
        "[INFO] Feature statistics (mean/std): "
        f"{dataset.feature_mean.tolist()} / {dataset.feature_std.tolist()}"
    )
    print(
        "[INFO] Target statistics (mean/std): "
        f"{dataset.target_mean.tolist()} / {dataset.target_std.tolist()}"
    )
    print(f"[INFO] Checkpoint location: {OUTPUT_PATH}")

    plot_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_loss_curves.png")
    plot_loss_curves(train_history, val_history, plot_path)


def main() -> None:
    # Ensure the output directory exists before saving the model
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {DATA_DIR}")

    if MODEL_NAME == "clcd_mlp":
        _train_clcd_mlp(device)
        return

    # Load the full dataset of flow-field samples
    dataset = AirfoilFlowDataset(DATA_DIR)
    dx, dy = _load_reference_grid_spacing(DATA_DIR)
    print(f"[INFO] Grid spacing | dx: {dx:.6e} | dy: {dy:.6e}")

    split = build_train_val_split(
        dataset=dataset,
        validation_split=ML_CFG.training.validation_split,
        split_seed=ML_CFG.training.split_seed,
    )
    train_set = split.train_set
    val_set = split.val_set
    train_indices = split.train_indices

    # Fit condition normalisation from training samples only
    dataset.fit_condition_normalization(train_indices)
    dataset.fit_target_normalization(train_indices)
    print(
        "[INFO] Condition norm stats | "
        f"aoa mean/std: {dataset.aoa_mean:.4f}/{dataset.aoa_std:.4f} | "
        f"inlet_u mean/std: {dataset.inlet_u_mean:.4f}/{dataset.inlet_u_std:.4f}"
    )
    print(
        "[INFO] Target norm stats | "
        f"p mean/std: {dataset.p_mean:.4f}/{dataset.p_std:.4f} | "
        f"ux mean/std: {dataset.ux_mean:.4f}/{dataset.ux_std:.4f} | "
        f"uy mean/std: {dataset.uy_mean:.4f}/{dataset.uy_std:.4f}"
    )

    # Create data loaders for batched iteration
    train_loader = DataLoader(train_set, batch_size=ML_CFG.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=ML_CFG.training.batch_size, shuffle=False)

    # Initialise selected model and Adam optimiser
    if MODEL_NAME == "simple_unet":
        model_kwargs = asdict(ML_CFG.model.params.simple_unet)
    elif MODEL_NAME == "rans_pinn":
        model_kwargs = asdict(ML_CFG.model.params.rans_pinn)
    else:
        raise ValueError(f"Unsupported model in config: {MODEL_NAME}")

    model = build_model(MODEL_NAME, **model_kwargs).to(device)
    print(f"[INFO] Model: {MODEL_NAME}")
    optimizer = torch.optim.Adam(model.parameters(), lr=ML_CFG.training.learning_rate)
    physics_available = hasattr(model, "rans_residual_loss")
    if physics_available:
        print("[INFO] Physics residual loss: enabled (model supports rans_residual_loss)")
    else:
        print("[INFO] Physics residual loss: disabled (model has no rans_residual_loss)")

    # Accumulators for loss history (used for the final plot)
    train_history: list[float] = []
    val_history: list[float] = []

    # --- Training loop -------------------------------------------------------
    for epoch in range(1, ML_CFG.training.epochs + 1):
        if not physics_available:
            physics_weight = 0.0
        elif epoch < ML_CFG.physics_loss.warmup_epochs:
            physics_weight = 0.0
        else:
            physics_weight = ML_CFG.physics_loss.weight

        if physics_available:
            train_metrics = train_one_epoch_physics(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                dx=dx,
                dy=dy,
                nu=ML_CFG.physics_loss.nu,
                p_mean=dataset.p_mean,
                p_std=dataset.p_std,
                ux_mean=dataset.ux_mean,
                ux_std=dataset.ux_std,
                uy_mean=dataset.uy_mean,
                uy_std=dataset.uy_std,
                physics_weight=physics_weight,
                pressure_is_kinematic=ML_CFG.physics_loss.pressure_is_kinematic,
                mask_erode_pixels=ML_CFG.physics_loss.mask_erode_pixels,
            )
        else:
            data_loss = train_one_epoch(model, train_loader, optimizer, device)
            train_metrics = {
                "loss_total": data_loss,
                "loss_data": data_loss,
                "loss_physics": 0.0,
                "loss_continuity": 0.0,
                "loss_momentum_x": 0.0,
                "loss_momentum_y": 0.0,
            }
        val_loss = evaluate(model, val_loader, device)                        # Evaluate on validation set

        train_history.append(train_metrics["loss_total"])
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"physics_weight: {physics_weight:.6e} | "
            f"total loss: {train_metrics['loss_total']:.6e} | "
            f"data loss: {train_metrics['loss_data']:.6e} | "
            f"physics loss: {train_metrics['loss_physics']:.6e} | "
            f"continuity loss: {train_metrics['loss_continuity']:.6e} | "
            f"momentum_x_loss: {train_metrics['loss_momentum_x']:.6e} | "
            f"momentum_y_loss: {train_metrics['loss_momentum_y']:.6e} | "
            f"val loss: {val_loss:.6e}"
        )

    # Save model weights after training is complete
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"[INFO] Saved model to: {OUTPUT_PATH}")

    # Plot and save the train/val loss curves
    plot_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_loss_curves.png")
    plot_loss_curves(train_history, val_history, plot_path)


if __name__ == "__main__":
    main()