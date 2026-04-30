from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_paths
from src.ml_dataset import AirfoilFlowDataset
from src.ml_models import SimpleUNet
from src.ml_training import train_one_epoch, evaluate


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
PATHS = load_paths(CONFIG_PATH)

DATA_DIR = PATHS.flow_fields_output
OUTPUT_PATH = PATHS.project_root / "data" / "models" / "simple_unet_airfoil.pt"

BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-3


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {DATA_DIR}")

    dataset = AirfoilFlowDataset(DATA_DIR)

    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size

    if val_size == 0:
        train_set = dataset
        val_set = dataset
    else:
        train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6e} | "
            f"val loss: {val_loss:.6e}"
        )

    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"[INFO] Saved model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()