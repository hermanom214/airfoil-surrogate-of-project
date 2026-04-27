from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FILES = [
    r"D:\Martina\REPOS\airfoil_OF_ML_project\airfoil-surrogate-of-project\data\flow_fields\case_0001_naca0012_aoa0p0_u20p0_flow.npz",
    r"D:\Martina\REPOS\airfoil_OF_ML_project\airfoil-surrogate-of-project\data\flow_fields\case_0002_naca2412_aoa0p0_u20p0_flow.npz",
    r"D:\Martina\REPOS\airfoil_OF_ML_project\airfoil-surrogate-of-project\data\flow_fields\case_0003_naca4415_aoa0p0_u20p0_flow.npz",
    r"D:\Martina\REPOS\airfoil_OF_ML_project\airfoil-surrogate-of-project\data\flow_fields\case_0004_naca2208_aoa0p0_u20p0_flow.npz"
]

FILE_PATH = Path(FILES[0])


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


def plot_fields(xy, p, U, fluid_mask, title: str = "") -> None:
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    u_mag = np.sqrt(U[:, :, 0] ** 2 + U[:, :, 1] ** 2)

    p_masked = np.where(fluid_mask > 0.5, p, np.nan)
    u_mag_masked = np.where(fluid_mask > 0.5, u_mag, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(
        p_masked,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
    )
    axes[0].set_title("Pressure p masked")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_mag_masked,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
    )
    axes[1].set_title("|U| masked")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        fluid_mask,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
    )
    axes[2].set_title("fluid_mask")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    if not FILE_PATH.exists():
        raise FileNotFoundError(FILE_PATH)

    print(f"[INFO] Loading: {FILE_PATH}")

    xy, p, U, fluid_mask = load_npz(FILE_PATH)

    validate_shapes(xy, p, U, fluid_mask)

    print_stats("Pressure p", p)
    print_stats("Velocity Ux", U[:, :, 0])
    print_stats("Velocity Uy", U[:, :, 1])
    print_stats("fluid_mask", fluid_mask)

    solid_fraction = 1.0 - float(np.mean(fluid_mask))
    print(f"\nSolid fraction: {solid_fraction:.6f}")

    plot_fields(xy, p, U, fluid_mask, title=FILE_PATH.name)


if __name__ == "__main__":
    main()