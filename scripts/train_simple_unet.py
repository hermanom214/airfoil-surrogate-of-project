# Training script for the SimpleUNet surrogate model.
# Loads pre-processed airfoil flow-field data (.npz files), splits it into
# training and validation sets, trains a U-Net-based neural network to predict
# CFD flow fields from geometry/condition inputs, and saves the resulting model
# weights to disk.

from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Allow imports from the project root (src/ package)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_paths
from src.ml_dataset import AirfoilFlowDataset
from src.ml_models import SimpleUNet
from src.ml_training import train_one_epoch, evaluate
from src.ml_validation import plot_loss_curves


# --- Configuration -----------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
PATHS = load_paths(CONFIG_PATH)  # Load project paths from YAML config

DATA_DIR = PATHS.flow_fields_output           # Directory with .npz flow-field files
OUTPUT_PATH = PATHS.project_root / "data" / "models" / "simple_unet_airfoil.pt"  # Where to save trained weights

# Hyper-parameters
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-3


def main() -> None:
    # Ensure the output directory exists before saving the model
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {DATA_DIR}")

    # Load the full dataset of flow-field samples
    dataset = AirfoilFlowDataset(DATA_DIR)

    # Split dataset 80/20 into training and validation subsets
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size

    if val_size == 0:
        # If dataset is too small to split, reuse it for both sets
        train_set = dataset
        val_set = dataset
    else:
        generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialise the U-Net model and Adam optimiser
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Accumulators for loss history (used for the final plot)
    train_history: list[float] = []
    val_history: list[float] = []

    # --- Training loop -------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)  # One pass over training data
        val_loss = evaluate(model, val_loader, device)                        # Evaluate on validation set

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6e} | "
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