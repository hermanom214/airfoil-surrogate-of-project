#!/usr/bin/env python3
"""
Skript pro vykreslení residuí z OpenFOAM residuals.dat
Lze spustit přímo ze složky s residuals.dat nebo zadat cestu jako argument.

Použití:
    python plot_residuals.py
    python plot_residuals.py /cesta/ke/residuals.dat
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_residuals(file_path: Path) -> dict:
    """
    Načte OpenFOAM residuals.dat.
    Vrací dict s klíči: 'time', 'p', 'Ux', 'Uy', 'k', 'omega'
    """
    data = {
        "time": [],
        "p": [],
        "Ux": [],
        "Uy": [],
        "k": [],
        "omega": [],
    }

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                time = float(parts[0])
                p = float(parts[1]) if parts[1] != "N/A" else np.nan
                ux = float(parts[2]) if parts[2] != "N/A" else np.nan
                uy = float(parts[3]) if parts[3] != "N/A" else np.nan
                k = float(parts[4]) if parts[4] != "N/A" else np.nan
                omega = float(parts[5]) if parts[5] != "N/A" else np.nan

                data["time"].append(time)
                data["p"].append(p)
                data["Ux"].append(ux)
                data["Uy"].append(uy)
                data["k"].append(k)
                data["omega"].append(omega)

            except (ValueError, IndexError):
                continue

    # Konverze na numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def plot_residuals(data: dict, file_path: Path) -> None:
    """
    Vykreslí residua na log skále.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"OpenFOAM Residuals: {file_path.name}", fontsize=14, fontweight="bold")

    time = data["time"]

    # Pressure
    ax = axes[0, 0]
    ax.semilogy(time, data["p"], "b-", linewidth=1.5, label="p")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("Pressure")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Velocity components
    ax = axes[0, 1]
    ax.semilogy(time, data["Ux"], "r-", linewidth=1.5, label="Ux")
    ax.semilogy(time, data["Uy"], "g-", linewidth=1.5, label="Uy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("Velocity Components")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Turbulence k
    ax = axes[1, 0]
    ax.semilogy(time, data["k"], "m-", linewidth=1.5, label="k")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("Turbulent Kinetic Energy (k)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Turbulence omega
    ax = axes[1, 1]
    ax.semilogy(time, data["omega"], "c-", linewidth=1.5, label="omega")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("Omega (ω)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Statistika posledních 100 iterací
    if len(time) > 100:
        print("\n=== STATISTIKA (poslední 100 iterací) ===")
        print(f"{'Proměnná':<15} {'Min':<15} {'Max':<15} {'Průměr':<15}")
        print("-" * 60)

        for key in ["p", "Ux", "Uy", "k", "omega"]:
            vals = data[key][-100:]
            vals_valid = vals[~np.isnan(vals)]
            if len(vals_valid) > 0:
                print(
                    f"{key:<15} {np.min(vals_valid):<15.6e} {np.max(vals_valid):<15.6e} {np.mean(vals_valid):<15.6e}"
                )


def main():
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Hledej residuals.dat v aktuální složce
        file_path = Path.cwd() / "residuals.dat"

    if not file_path.exists():
        print(f"Soubor nenalezen: {file_path}")
        print("\nUžití:")
        print("  python plot_residuals.py                    # Hledá residuals.dat v aktuální složce")
        print("  python plot_residuals.py /cesta/k/residuals.dat  # Zadaná cesta")
        sys.exit(1)

    print(f"Načítám: {file_path}")
    data = parse_residuals(file_path)

    if len(data["time"]) == 0:
        print("Žádná validní data v souboru!")
        sys.exit(1)

    print(f"Nalezeno {len(data['time'])} iterací")
    plot_residuals(data, file_path)


if __name__ == "__main__":
    main()
