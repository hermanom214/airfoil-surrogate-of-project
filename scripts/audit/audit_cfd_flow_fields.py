"""Read-only local aerodynamic audit of selected archived CFD NPZ fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "flow_fields"
OUT = ROOT / "outputs" / "cfd_field_audit"
FIG = OUT / "figures"
NU = 1.5e-5
WAKE_X = (1.05, 1.10, 1.25, 1.50)

SELECTION = {
    "case_0063_naca0012_aoa0p0_u20p0": "symmetric NACA 0012; zero-AoA baseline",
    "case_0073_naca0012_aoa4p0_u20p0": "NACA 0012; highest positive AoA",
    "case_0053_naca0012_aoam4p0_u20p0": "NACA 0012; corresponding negative AoA",
    "case_0563_naca2412_aoa0p0_u20p0": "cambered NACA 2412; zero AoA",
    "case_0573_naca2412_aoa4p0_u20p0": "cambered NACA 2412; positive AoA",
    "case_0011_naca0008_aoa0p0_u15p0": "thin symmetric profile and lowest inlet speed",
    "case_0115_naca0016_aoa0p0_u25p0": "thick symmetric profile and highest inlet speed",
    "case_0575_naca2412_aoa4p0_u25p0": "cambered positive-AoA high-speed comparison",
}


def body_coordinates(x, y, aoa, chord):
    a = np.deg2rad(aoa)
    xr, yr = x - 0.25 * chord, y
    return np.cos(a) * xr - np.sin(a) * yr + 0.25 * chord, np.sin(a) * xr + np.cos(a) * yr


def naca_camber_y(code, x, chord=1.0):
    m, pos = int(code[0]) / 100.0, int(code[1]) / 10.0
    xc = np.asarray(x) / chord
    yc = np.zeros_like(xc, dtype=float)
    if m and pos:
        left = xc < pos
        yc[left] = m / pos**2 * (2 * pos * xc[left] - xc[left] ** 2) * chord
        yc[~left] = m / (1 - pos) ** 2 * ((1 - 2 * pos) + 2 * pos * xc[~left] - xc[~left] ** 2) * chord
    return yc


def nearest_profile(xb, yb, field, valid, station, half_height=0.25):
    dx = float(np.median(np.diff(np.unique(xb[0])))) if np.allclose(xb, xb[0:1]) else 0.004
    band = valid & (np.abs(xb - station) <= max(abs(dx), 0.0045)) & (np.abs(yb) <= half_height)
    yy, vv = yb[band], field[band]
    order = np.argsort(yy)
    return yy[order], vv[order]


def wake_metrics(y, ux_ratio):
    deficit = 1.0 - ux_ratio
    if len(y) == 0 or not np.isfinite(deficit).any():
        return np.nan, np.nan, np.nan, np.nan
    k = int(np.nanargmax(deficit))
    dmax = float(deficit[k])
    threshold = 0.5 * dmax
    good = deficit >= threshold
    lo = hi = k
    while lo > 0 and good[lo - 1]: lo -= 1
    while hi + 1 < len(good) and good[hi + 1]: hi += 1
    width = float(y[hi] - y[lo]) if hi > lo else 0.0
    return float(np.nanmin(ux_ratio)), dmax, float(y[k]), width


def draw_panels(path, title, x, y, fields, mask, extent, labels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    for ax, field, label in zip(axes.flat, fields, labels):
        shown = np.ma.masked_where(~mask, field)
        vals = shown.compressed()
        if label == "Cp":
            lim = max(abs(np.nanpercentile(vals, 1)), abs(np.nanpercentile(vals, 99)))
            im = ax.pcolormesh(x, y, shown, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        elif label == "Uy/U∞":
            lim = max(abs(np.nanpercentile(vals, 1)), abs(np.nanpercentile(vals, 99)), 1e-9)
            im = ax.pcolormesh(x, y, shown, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        else:
            lo, hi = np.nanpercentile(vals, [1, 99])
            im = ax.pcolormesh(x, y, shown, shading="auto", cmap="viridis", vmin=lo, vmax=hi)
        ax.contour(x, y, mask.astype(float), levels=[0.5], colors="k", linewidths=0.7)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x/c (body)", ylabel="y/c (body)", title=label)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def load_case(case_id):
    path = DATA / case_id / f"{case_id}.npz"
    with np.load(path) as z:
        params = json.loads(str(z["params"]))
        return {"xy": z["xy"].copy(), "p": z["p"].copy(), "U": z["U"].copy(),
                "mask": z["fluid_mask"].astype(bool), "params": params,
                "latest_time": str(z["latest_time"]), "path": path}


def audit_case(case_id, reason):
    d = load_case(case_id); q = d["params"]
    c, aoa, uinf, code = float(q["chord"]), float(q["aoa_deg"]), float(q["inlet_velocity"]), str(q["naca_code"])
    x, y = d["xy"][..., 0], d["xy"][..., 1]
    xb, yb = body_coordinates(x, y, aoa, c); xb /= c; yb /= c
    p, ux, uy, valid = d["p"], d["U"][..., 0], d["U"][..., 1], d["mask"]
    cp = p / (0.5 * uinf**2); speed = np.hypot(ux, uy)
    fields = [cp, speed/uinf, ux/uinf, uy/uinf]
    labels = ["Cp", "|U|/U∞", "Ux/U∞", "Uy/U∞"]
    draw_panels(FIG/f"{case_id}_full.png", f"{case_id}: full interpolated field", xb, yb, fields, valid, (-0.75,1.75,-0.75,0.75), labels)
    draw_panels(FIG/f"{case_id}_leading_edge.png", f"{case_id}: leading edge", xb, yb, fields, valid, (-0.05,0.20,-0.15,0.15), labels)
    draw_panels(FIG/f"{case_id}_trailing_edge.png", f"{case_id}: trailing edge / immediate wake", xb, yb, fields, valid, (0.85,1.15,-0.12,0.12), labels)
    draw_panels(FIG/f"{case_id}_wake.png", f"{case_id}: wake", xb, yb, fields, valid, (1.0,1.75,-0.25,0.25), labels)

    le = valid & (xb >= -0.05) & (xb <= 0.20) & (np.abs(yb) <= 0.15)
    imax = np.nanargmax(np.where(le, cp, np.nan)); imin = np.nanargmin(np.where(le, cp, np.nan)); ivmax = np.nanargmax(np.where(le, speed, np.nan))
    nose = valid & (xb >= -0.025) & (xb <= 0.055) & (np.abs(yb) <= 0.09)
    istag = np.nanargmin(np.where(nose, speed, np.nan))
    unravel = lambda idx: np.unravel_index(idx, cp.shape)
    rmax, rmin, rvmax, rstag = map(unravel, (imax, imin, ivmax, istag))

    xte = 0.98; yc = float(naca_camber_y(code, np.array([xte*c]), c)[0]/c)
    band = valid & (np.abs(xb-xte) <= 0.006)
    upper_candidates = np.argwhere(band & (yb > yc)); lower_candidates = np.argwhere(band & (yb < yc))
    up = tuple(upper_candidates[np.argmin(yb[tuple(upper_candidates.T)]-yc)])
    lo = tuple(lower_candidates[np.argmin(yc-yb[tuple(lower_candidates.T)])])
    yd, ud = nearest_profile(xb, yb, ux/uinf, valid, 1.02, 0.12)

    probes = [{
        "case_id":case_id,"probe":"leading_edge_summary","x_over_c":xb[rmax],"y_over_c":yb[rmax],
        "value":cp[rmax],"secondary_value":cp[rmin],"secondary_x_over_c":xb[rmin],"secondary_y_over_c":yb[rmin],
        "notes":f"Cp_max; secondary=Cp_min. Stagnation proxy x/c={xb[rstag]:.4f}, y/c={yb[rstag]:.4f}, |U|/Uinf={speed[rstag]/uinf:.4f}; max |U|/Uinf={speed[rvmax]/uinf:.4f} at ({xb[rvmax]:.4f},{yb[rvmax]:.4f})"},
        {"case_id":case_id,"probe":"trailing_edge_summary","x_over_c":xte,"y_over_c":yc,"value":cp[up]-cp[lo],
         "secondary_value":float(1-np.nanmin(ud)) if len(ud) else np.nan,"secondary_x_over_c":1.02,"secondary_y_over_c":yd[np.nanargmin(ud)] if len(ud) else np.nan,
         "notes":f"value=Cp_upper-Cp_lower; Uupper/Uinf={speed[up]/uinf:.4f}; Ulower/Uinf={speed[lo]/uinf:.4f}; secondary=immediate wake deficit"}]
    profile_data=[]
    for station in WAKE_X:
        yp, ur = nearest_profile(xb, yb, ux/uinf, valid, station)
        _, vr = nearest_profile(xb, yb, uy/uinf, valid, station)
        umin, deficit, center, width = wake_metrics(yp, ur)
        probes.append({"case_id":case_id,"probe":f"wake_x_{station:.2f}","x_over_c":station,"y_over_c":center,"value":umin,
                       "secondary_value":deficit,"secondary_x_over_c":width,"secondary_y_over_c":np.nan,
                       "notes":"value=min Ux/Uinf; secondary=1-min; secondary_x=50%-maximum-deficit contiguous width/c"})
        profile_data.append((station,yp,ur,vr))
    fig, (a1,a2)=plt.subplots(1,2,figsize=(10,5),sharey=True,constrained_layout=True)
    for station,yp,ur,vr in profile_data:
        a1.plot(ur,yp,label=f"x/c={station:.2f}"); a2.plot(vr,yp,label=f"x/c={station:.2f}")
    a1.set(xlabel="Ux/U∞",ylabel="y/c",title="Streamwise profiles",ylim=(-0.25,0.25)); a2.set(xlabel="Uy/U∞",title="Transverse profiles")
    a1.axvline(1,color="k",lw=.6); a2.axvline(0,color="k",lw=.6); a1.legend(fontsize=8); a2.legend(fontsize=8)
    fig.suptitle(f"{case_id}: wake profiles"); fig.savefig(FIG/f"{case_id}_wake_profiles.png",dpi=170); plt.close(fig)
    selected={"case_id":case_id,"NACA":code,"AoA_deg":aoa,"U_inf_mps":uinf,"chord_m":c,"Re":uinf*c/NU,"reason_selected":reason}
    return selected, probes


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    selected=[]; probes=[]
    for case_id, reason in SELECTION.items():
        s,p=audit_case(case_id,reason); selected.append(s); probes.extend(p); print(case_id)
    with (OUT/"selected_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=selected[0].keys()); w.writeheader(); w.writerows(selected)
    with (OUT/"quantitative_probes.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=probes[0].keys()); w.writeheader(); w.writerows(probes)
    metadata={"case_count":609,"grid_shape":[320,640],"grid_extent_m":{"x":[-0.75,1.75],"y":[-0.75,0.75]},
              "dx_m":2.5/639,"dy_m":1.5/319,"solver":"simpleFoam","simulation":"steady RANS",
              "turbulence_model":"kOmegaSST","nu_m2_s":NU,"sample_time":"latest/final directory 1000",
              "extraction":"linear scipy.griddata from VTK with nearest fill outside convex hull",
              "pressure":"OpenFOAM kinematic pressure [m2/s2], outlet reference 0","velocity":"U [m/s]",
              "mask":"1 fluid, 0 solid; analytic NACA mask rotated -AoA about quarter chord"}
    (OUT/"audit_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
