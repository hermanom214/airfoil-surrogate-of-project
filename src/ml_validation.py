# Validation utilities for visualising training progress.
# Provides a function to plot train and validation loss curves over epochs
# and save the resulting figure to disk next to the saved model weights.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Plot train and validation loss vs. epoch and save the figure.

    Args:
        train_losses: Loss value recorded after each training epoch.
        val_losses:   Loss value recorded after each validation epoch.
        output_path:  Path where the PNG figure will be saved.
                      The parent directory must already exist.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot both curves
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_losses, label="Val loss", linewidth=1.8, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (masked MSE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Loss curve saved to: {output_path}")
