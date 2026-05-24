# Training script for the SimpleUNet surrogate model.
# Loads pre-processed airfoil flow-field data (.npz files), splits it into
# training and validation sets, trains a U-Net-based neural network to predict
# CFD flow fields from geometry/condition inputs, and saves the resulting model
# weights to disk.

from __future__ import annotations
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow imports from the project root (src/ package)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_ml_models_config, load_paths
from src.ml_dataset import AirfoilFlowDataset
from src.ml_models import build_model
from src.ml_training import evaluate, masked_mse
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


class PhysicsLossModel(Protocol):
    def rans_residual_loss(
        self,
        pred: torch.Tensor,
        fluid_mask: torch.Tensor,
        dx: float,
        dy: float,
        nu: float,
        nu_t: torch.Tensor | None = None,
        u_scale: float = 50.0,
        p_scale: float = 1000.0,
        pressure_is_kinematic: bool = True,
        mask_erode_pixels: int = 1,
    ) -> dict[str, torch.Tensor]:
        ...


def _compute_grid_spacing(xy: torch.Tensor) -> tuple[float, float]:
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy grid tensor must have shape [H, W, 2]")

    x = xy[:, :, 0]
    y = xy[:, :, 1]

    dx_candidates = torch.abs(x[:, 1:] - x[:, :-1]).reshape(-1)
    dy_candidates = torch.abs(y[1:, :] - y[:-1, :]).reshape(-1)

    dx_valid = dx_candidates[dx_candidates > 0]
    dy_valid = dy_candidates[dy_candidates > 0]

    if dx_valid.numel() == 0 or dy_valid.numel() == 0:
        raise RuntimeError("Unable to infer positive dx/dy from xy grid")

    dx = float(dx_valid.median().item())
    dy = float(dy_valid.median().item())
    return dx, dy


def _load_reference_grid_spacing(data_dir: Path) -> tuple[float, float]:
    files = sorted(data_dir.glob("*_flow.npz"))
    if not files:
        raise RuntimeError(f"No *_flow.npz files found in {data_dir}")

    with np.load(files[0]) as data:
        xy = torch.from_numpy(data["xy"].astype(np.float32))
    return _compute_grid_spacing(xy)


def train_one_epoch_physics(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    dx: float,
    dy: float,
    nu: float,
    physics_weight: float,
    u_scale: float,
    p_scale: float,
    pressure_is_kinematic: bool,
    mask_erode_pixels: int,
) -> dict[str, float]:
    if not hasattr(model, "rans_residual_loss"):
        raise TypeError("Selected model does not implement rans_residual_loss")

    physics_model = cast(PhysicsLossModel, model)
    model.train()

    totals = {
        "loss_total": 0.0,
        "loss_data": 0.0,
        "loss_physics": 0.0,
        "loss_continuity": 0.0,
        "loss_momentum_x": 0.0,
        "loss_momentum_y": 0.0,
    }

    for inp, target, mask in loader:
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device)

        pred = model(inp)

        loss_data = masked_mse(pred, target, mask)
        phys = physics_model.rans_residual_loss(
            pred=pred,
            fluid_mask=mask,
            dx=dx,
            dy=dy,
            nu=nu,
            u_scale=u_scale,
            p_scale=p_scale,
            pressure_is_kinematic=pressure_is_kinematic,
            mask_erode_pixels=mask_erode_pixels,
        )
        loss_total = loss_data + physics_weight * phys["physics_loss"]

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        totals["loss_total"] += float(loss_total.item())
        totals["loss_data"] += float(loss_data.item())
        totals["loss_physics"] += float(phys["physics_loss"].item())
        totals["loss_continuity"] += float(phys["continuity_loss"].item())
        totals["loss_momentum_x"] += float(phys["momentum_x_loss"].item())
        totals["loss_momentum_y"] += float(phys["momentum_y_loss"].item())

    n_batches = len(loader)
    if n_batches == 0:
        raise RuntimeError("Training loader has zero batches")

    return {k: v / n_batches for k, v in totals.items()}


def main() -> None:
    # Ensure the output directory exists before saving the model
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {DATA_DIR}")

    # Load the full dataset of flow-field samples
    dataset = AirfoilFlowDataset(DATA_DIR)
    dx, dy = _load_reference_grid_spacing(DATA_DIR)
    print(f"[INFO] Grid spacing | dx: {dx:.6e} | dy: {dy:.6e}")

    # Split dataset 80/20 into training and validation subsets
    train_size = max(1, int((1.0 - ML_CFG.training.validation_split) * len(dataset)))
    val_size = len(dataset) - train_size

    if val_size == 0:
        # If dataset is too small to split, reuse it for both sets
        train_set = dataset
        val_set = dataset
    else:
        generator = torch.Generator().manual_seed(ML_CFG.training.split_seed)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    # Fit condition normalisation from training samples only
    if hasattr(train_set, "indices"):
        train_indices = list(train_set.indices)
    else:
        train_indices = list(range(len(dataset)))
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

    # Accumulators for loss history (used for the final plot)
    train_history: list[float] = []
    val_history: list[float] = []

    # --- Training loop -------------------------------------------------------
    for epoch in range(1, ML_CFG.training.epochs + 1):
        if epoch < ML_CFG.physics_loss.warmup_epochs:
            physics_weight = 0.0
        else:
            physics_weight = ML_CFG.physics_loss.weight

        train_metrics = train_one_epoch_physics(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            dx=dx,
            dy=dy,
            nu=ML_CFG.physics_loss.nu,
            physics_weight=physics_weight,
            u_scale=ML_CFG.physics_loss.u_scale,
            p_scale=ML_CFG.physics_loss.p_scale,
            pressure_is_kinematic=ML_CFG.physics_loss.pressure_is_kinematic,
            mask_erode_pixels=ML_CFG.physics_loss.mask_erode_pixels,
        )
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