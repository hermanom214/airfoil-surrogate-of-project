from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FLOW_FIELDS_DIR = Path(__file__).parent.parent / "data" / "flow_fields"
PICTURES_DIR = Path(__file__).parent.parent / "data" / "pictures"


def print_stats(name: str, arr: np.ndarray) -> None:
    print(f"\n{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  min:   {np.nanmin(arr):.6f}")
    print(f"  max:   {np.nanmax(arr):.6f}")
    print(f"  mean:  {np.nanmean(arr):.6f}")
    print(f"  std:   {np.nanstd(arr):.6f}")
    print(f"  NaN count: {np.isnan(arr).sum()}")
    print(f"  Inf count: {np.isinf(arr).sum()}")


def load_npz(file_path: Path):
    data = np.load(file_path, allow_pickle=True)

    required = ["xy", "p", "U", "fluid_mask"]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Missing key '{key}' in {file_path}")

    return data["xy"], data["p"], data["U"], data["fluid_mask"]


def validate_shapes(xy, p, U, fluid_mask) -> None:
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise RuntimeError(f"Invalid xy shape: {xy.shape}")

    if p.ndim != 2:
        raise RuntimeError(f"Invalid p shape: {p.shape}")

    if U.ndim != 3 or U.shape[2] != 2:
        raise RuntimeError(f"Invalid U shape: {U.shape}")

    if fluid_mask.ndim != 2:
        raise RuntimeError(f"Invalid fluid_mask shape: {fluid_mask.shape}")

    ny, nx = p.shape

    if xy.shape[:2] != (ny, nx):
        raise RuntimeError("xy and p shape mismatch")

    if U.shape[:2] != (ny, nx):
        raise RuntimeError("U and p shape mismatch")

    if fluid_mask.shape != (ny, nx):
        raise RuntimeError("fluid_mask and p shape mismatch")


def robust_limits(arr: np.ndarray, low: float = 1.0, high: float = 99.0):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    return np.percentile(valid, low), np.percentile(valid, high)


def plot_fields(xy, p, U, fluid_mask, title: str = "", save_path: Path | None = None) -> None:
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    u_mag = np.sqrt(U[:, :, 0] ** 2 + U[:, :, 1] ** 2)

    p_masked = np.where(fluid_mask > 0.5, p, np.nan)
    u_mag_masked = np.where(fluid_mask > 0.5, u_mag, np.nan)

    # ParaView-like fixed / robust color limits
    p_vmin, p_vmax = robust_limits(p_masked, 1.0, 99.0)
    u_vmin, u_vmax = 0.0, robust_limits(u_mag_masked, 1.0, 99.5)[1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    im0 = axes[0].imshow(
        p_masked,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap="turbo",
        vmin=p_vmin,
        vmax=p_vmax,
    )
    axes[0].set_title(f"Pressure p\nrange: {p_vmin:.2f} to {p_vmax:.2f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_mag_masked,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap="turbo",
        vmin=u_vmin,
        vmax=u_vmax,
    )
    axes[1].set_title(f"|U|\nrange: {u_vmin:.2f} to {u_vmax:.2f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        fluid_mask,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("fluid_mask")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    npz_files = sorted(FLOW_FIELDS_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {FLOW_FIELDS_DIR}")

    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(npz_files)} file(s). Saving images to: {PICTURES_DIR}")

    for file_path in npz_files:
        print(f"[INFO] Processing: {file_path.name}")
        try:
            xy, p, U, fluid_mask = load_npz(file_path)
            validate_shapes(xy, p, U, fluid_mask)
            save_path = PICTURES_DIR / (file_path.stem + ".png")
            plot_fields(xy, p, U, fluid_mask, title=file_path.name, save_path=save_path)
            print(f"  -> saved: {save_path.name}")
        except Exception as exc:
            print(f"  [ERROR] {file_path.name}: {exc}")


if __name__ == "__main__":
    main()