"""Evaluate local wake v1 and compare it fairly with the global improved U-Net."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.highres_wake_patch_feasibility import lab_to_body, body_to_lab, wake_metric
from src.ml_models import build_model
from src.ml_supervised import near_wall_region_mask, physical_to_targets, targets_to_physical


LOCAL_DATA = ROOT / "data" / "flow_fields_highres_wake"
GLOBAL_DATA = ROOT / "data" / "flow_fields"
LOCAL_CHECKPOINT = ROOT / "data" / "models" / "simple_unet_local_wake_v1" / "simple_unet_airfoil.pt"
GLOBAL_CHECKPOINT = ROOT / "data" / "models" / "simple_unet_improved" / "simple_unet_airfoil.pt"
OUTPUT = ROOT / "data" / "models" / "simple_unet_local_wake_v1_validation"
STATIONS = (1.02, 1.05, 1.10, 1.25, 1.50)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model("simple_unet", **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def model_input(npz_path: Path, checkpoint: dict) -> tuple[np.ndarray, dict, dict]:
    with np.load(npz_path, allow_pickle=False) as z:
        xy, p, u, mask = z["xy"].astype(np.float32), z["p"].astype(np.float32), z["U"].astype(np.float32), z["fluid_mask"].astype(np.float32)
        params = json.loads(str(z["params"]))
    aoa, u_inf = float(params["aoa_deg"]), float(params["inlet_velocity"])
    x, y = xy[..., 0], xy[..., 1]
    inp = np.stack((mask, (x-x.mean())/(x.std()+1e-8), (y-y.mean())/(y.std()+1e-8),
                    np.full_like(mask, (aoa-checkpoint["aoa_mean"])/checkpoint["aoa_std"]),
                    np.full_like(mask, (u_inf-checkpoint["inlet_u_mean"])/checkpoint["inlet_u_std"])))
    return inp.astype(np.float32), {"xy": xy, "p": p, "U": u, "mask": mask}, params


@torch.no_grad()
def predict(model, inp: np.ndarray, checkpoint: dict, u_inf: float, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = model(torch.from_numpy(inp[None]).to(device)).squeeze(0).cpu().numpy()
    p, ux, uy = targets_to_physical(pred, u_inf, checkpoint["target_representation"], checkpoint["supervised_loss_config"]["p_inf"])
    return pred, p, ux, uy


def resample_global(fields: list[np.ndarray], xy_global: np.ndarray, xy_local_body: np.ndarray,
                    aoa: float, chord: float) -> list[np.ndarray]:
    xl, yl = body_to_lab(xy_local_body[..., 0], xy_local_body[..., 1], aoa, chord)
    queries = np.c_[yl.ravel(), xl.ravel()]
    xs, ys = xy_global[0, :, 0], xy_global[:, 0, 1]
    return [RegularGridInterpolator((ys, xs), value, bounds_error=False, fill_value=np.nan)(queries).reshape(xl.shape)
            for value in fields]


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[valid]-target[valid])**2))) if np.any(valid) else float("nan")


def profile(field: np.ndarray, xy: np.ndarray, station: float, y: np.ndarray) -> np.ndarray:
    return RegularGridInterpolator((xy[:, 0, 1], xy[0, :, 0]), field, bounds_error=False, fill_value=np.nan)(
        np.c_[y, np.full_like(y, station)])


def relative_summary(rows: list[dict], local_key: str, global_key: str) -> dict[str, float | int]:
    local = np.array([float(r[local_key]) for r in rows]); global_ = np.array([float(r[global_key]) for r in rows])
    improvement = (global_-local)/global_
    return {"local_better_cases": int(np.count_nonzero(local < global_)),
            "global_better_cases": int(np.count_nonzero(global_ < local)),
            "ties": int(np.count_nonzero(global_ == local)),
            "median_relative_improvement": float(np.nanmedian(improvement)),
            "mean_relative_improvement": float(np.nanmean(improvement)),
            "worst_local_regression": float(np.nanmin(improvement)),
            "best_local_improvement": float(np.nanmax(improvement))}


def te_pressure_difference(field: np.ndarray, xy: np.ndarray, fluid: np.ndarray) -> float:
    """Upper-minus-lower pressure at the grid column nearest x/c=0.98."""
    column = int(np.argmin(np.abs(xy[0, :, 0] - 0.98)))
    y = xy[:, column, 1]
    upper = np.flatnonzero(fluid[:, column] & (y >= 0.0))
    lower = np.flatnonzero(fluid[:, column] & (y < 0.0))
    if not len(upper) or not len(lower): return float("nan")
    iu = upper[np.argmin(y[upper])]
    il = lower[np.argmax(y[lower])]
    return float(field[iu, column] - field[il, column])


def choose_visual_cases(case_rows: list[dict]) -> dict[str, str]:
    def naca(case_id: str) -> str: return re.search(r"naca(\d{4})", case_id).group(1)
    unused = {str(r["case_id"]) for r in case_rows}
    result: dict[str, str] = {}
    criteria = [
        ("easy_wake", lambda r: float(r["target_wake_deficit_mean"]) < .03),
        ("difficult_wake", lambda r: float(r["target_wake_deficit_mean"]) > .12),
        ("symmetric_aoa0", lambda r: naca(r["case_id"]).startswith("00") and float(r["AoA"]) == 0),
        ("positive_aoa", lambda r: float(r["AoA"]) > 0),
        ("negative_aoa", lambda r: float(r["AoA"]) < 0),
        ("cambered", lambda r: not naca(r["case_id"]).startswith("00")),
        ("thin", lambda r: int(naca(r["case_id"])[2:]) == 8),
        ("thick", lambda r: int(naca(r["case_id"])[2:]) == 16),
    ]
    for label, criterion in criteria:
        candidates = [r for r in case_rows if r["case_id"] in unused and criterion(r)]
        if not candidates: candidates = [r for r in case_rows if criterion(r)]
        if candidates:
            chosen = min(candidates, key=lambda r: float(r["local_wake_velocity_rmse"]))
            result[label] = str(chosen["case_id"]); unused.discard(str(chosen["case_id"]))
    return result


def plot_case(case_id: str, label: str, target: dict, global_pred: dict, local_pred: dict,
              xy: np.ndarray, u_inf: float) -> None:
    for field, unit in (("Cp", ""), ("Ux", "m/s"), ("Uy", "m/s"), ("speed", "m/s")):
        truth, gp, lp = target[field], global_pred[field], local_pred[field]
        values = truth[np.isfinite(truth)]
        lo, hi = np.percentile(values, [1, 99])
        errmax = np.percentile(np.abs(np.r_[gp.ravel()-truth.ravel(), lp.ravel()-truth.ravel()]), 99)
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharex=True, sharey=True)
        panels = ((truth, "high-res CFD", lo, hi, "viridis"), (gp, "global resampled", lo, hi, "viridis"),
                  (lp, "local prediction", lo, hi, "viridis"), (gp-truth, "global error", -errmax, errmax, "coolwarm"),
                  (lp-truth, "local error", -errmax, errmax, "coolwarm"))
        for ax, (arr, title, vmin, vmax, cmap) in zip(axes, panels):
            im = ax.pcolormesh(xy[...,0], xy[...,1], arr, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set(title=title, xlabel="x_body/c", xlim=(.7,1.75), ylim=(-.15,.15))
        axes[0].set_ylabel("y_body/c")
        fig.suptitle(f"{label}: {case_id} {field} {unit}")
        fig.tight_layout()
        fig.savefig(OUTPUT/"figures"/f"{label}_{case_id}_{field}.png", dpi=140); plt.close(fig)


def plot_wake_profiles(case_id: str, label: str, target_u: np.ndarray,
                       global_u: np.ndarray, local_u: np.ndarray,
                       xy: np.ndarray, u_inf: float) -> None:
    ys = np.linspace(-0.15, 0.15, 2401)
    fig, axes = plt.subplots(2, len(STATIONS), figsize=(18, 7), sharey=True)
    for column, station in enumerate(STATIONS):
        for row, component in enumerate((0, 1)):
            ax = axes[row, column]
            ax.plot(profile(target_u[..., component]/u_inf, xy, station, ys), ys, "k-", lw=1.4, label="CFD")
            ax.plot(profile(global_u[..., component]/u_inf, xy, station, ys), ys, color="tab:orange", lw=1.1, label="global")
            ax.plot(profile(local_u[..., component]/u_inf, xy, station, ys), ys, color="tab:blue", lw=1.1, label="local")
            ax.set_title(f"x/c={station:.2f}")
            ax.set_xlabel(("Ux" if component == 0 else "Uy") + "/U_inf")
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("y_body/c"); axes[1, 0].set_ylabel("y_body/c")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{label}: {case_id} wake profiles")
    fig.tight_layout(); fig.savefig(OUTPUT/"figures"/f"{label}_{case_id}_wake_profiles.png", dpi=140); plt.close(fig)


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"Refusing evaluation collision: {OUTPUT}")
    OUTPUT.mkdir(parents=True); (OUTPUT/"figures").mkdir()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_ckpt, local_model = load_model(LOCAL_CHECKPOINT, device)
    global_ckpt, global_model = load_model(GLOBAL_CHECKPOINT, device)
    case_rows: list[dict] = []; profile_rows: list[dict] = []
    visual_cache: dict[str, tuple] = {}
    for index, case_id in enumerate(local_ckpt["test_case_ids"], 1):
        local_path = LOCAL_DATA/case_id/f"{case_id}.npz"; global_path = GLOBAL_DATA/case_id/f"{case_id}.npz"
        linp, lt, params = model_input(local_path, local_ckpt)
        ginp, gt, _ = model_input(global_path, global_ckpt)
        u_inf, aoa, chord = float(params["inlet_velocity"]), float(params["aoa_deg"]), float(params["chord"])
        local_nd, lp, lux, luy = predict(local_model, linp, local_ckpt, u_inf, device)
        _, gp0, gux0, guy0 = predict(global_model, ginp, global_ckpt, u_inf, device)
        gp, gux, guy, gmask = resample_global([gp0, gux0, guy0, gt["mask"]], gt["xy"], lt["xy"], aoa, chord)
        mask = lt["mask"] > .5
        wall = near_wall_region_mask(lt["xy"], mask, chord, .05)
        x_c, y_c = lt["xy"][...,0]/chord, lt["xy"][...,1]/chord
        wake = mask & (x_c >= .9) & (x_c <= 1.75) & (np.abs(y_c) <= .20)
        te = mask & (x_c >= .90) & (x_c <= 1.10)
        target_speed = np.hypot(lt["U"][...,0], lt["U"][...,1]); local_speed=np.hypot(lux,luy); global_speed=np.hypot(gux,guy)
        target_nd = physical_to_targets(lt["p"], lt["U"][...,0], lt["U"][...,1], u_inf, "nondimensional_perturbation", 0.0)
        local_mse = float(np.sum((local_nd-target_nd)**2 * mask)/(np.count_nonzero(mask)*3))
        row = {"case_id":case_id,"NACA":params["naca_code"],"AoA":aoa,"U_inf":u_inf,"local_normalized_masked_mse":local_mse}
        for prefix, region in (("patch",mask),("wake",wake),("te",te),("wall",wall)):
            row[f"local_{prefix}_p_rmse"]=rmse(lp,lt["p"],region); row[f"global_{prefix}_p_rmse"]=rmse(gp,lt["p"],region)
            row[f"local_{prefix}_Ux_rmse"]=rmse(lux,lt["U"][...,0],region); row[f"global_{prefix}_Ux_rmse"]=rmse(gux,lt["U"][...,0],region)
            row[f"local_{prefix}_Uy_rmse"]=rmse(luy,lt["U"][...,1],region); row[f"global_{prefix}_Uy_rmse"]=rmse(guy,lt["U"][...,1],region)
            row[f"local_{prefix}_velocity_rmse"]=rmse(local_speed,target_speed,region); row[f"global_{prefix}_velocity_rmse"]=rmse(global_speed,target_speed,region)
            row[f"local_{prefix}_vector_rmse"]=float(math.hypot(row[f"local_{prefix}_Ux_rmse"],row[f"local_{prefix}_Uy_rmse"]))
            row[f"global_{prefix}_vector_rmse"]=float(math.hypot(row[f"global_{prefix}_Ux_rmse"],row[f"global_{prefix}_Uy_rmse"]))
        target_dp = te_pressure_difference(lt["p"], lt["xy"]/chord, mask)
        row["target_te_pressure_difference"] = target_dp
        row["local_te_pressure_difference_abs_error"] = abs(te_pressure_difference(lp, lt["xy"]/chord, mask)-target_dp)
        row["global_te_pressure_difference_abs_error"] = abs(te_pressure_difference(gp, lt["xy"]/chord, mask)-target_dp)
        deficits=[]
        ys=np.linspace(-.15,.15,2401)
        for station in STATIONS:
            tux=profile(lt["U"][...,0]/u_inf,lt["xy"]/chord,station,ys); tuy=profile(lt["U"][...,1]/u_inf,lt["xy"]/chord,station,ys)
            guxp=profile(gux/u_inf,lt["xy"]/chord,station,ys); guyp=profile(guy/u_inf,lt["xy"]/chord,station,ys)
            luxp=profile(lux/u_inf,lt["xy"]/chord,station,ys); luyp=profile(luy/u_inf,lt["xy"]/chord,station,ys)
            tm,gm,lm=wake_metric(ys,tux),wake_metric(ys,guxp),wake_metric(ys,luxp); deficits.append(tm.deficit)
            valid=np.isfinite(tux)&np.isfinite(guxp)&np.isfinite(luxp)
            profile_rows.append({"case_id":case_id,"station_xc":station,
                "global_Ux_profile_rmse":float(np.sqrt(np.mean((guxp[valid]-tux[valid])**2))),
                "local_Ux_profile_rmse":float(np.sqrt(np.mean((luxp[valid]-tux[valid])**2))),
                "global_Uy_profile_rmse":float(np.sqrt(np.mean((guyp[valid]-tuy[valid])**2))),
                "local_Uy_profile_rmse":float(np.sqrt(np.mean((luyp[valid]-tuy[valid])**2))),
                "target_Ux_min":tm.minimum,"global_Ux_min":gm.minimum,"local_Ux_min":lm.minimum,
                "global_Ux_min_abs_error":abs(gm.minimum-tm.minimum),"local_Ux_min_abs_error":abs(lm.minimum-tm.minimum),
                "target_wake_centre_yc":tm.centre,"global_wake_centre_abs_error":abs(gm.centre-tm.centre),"local_wake_centre_abs_error":abs(lm.centre-tm.centre),
                "target_half_deficit_width_c":tm.width,"global_width_abs_error":abs(gm.width-tm.width),"local_width_abs_error":abs(lm.width-tm.width)})
        row["target_wake_deficit_mean"]=float(np.mean(deficits)); case_rows.append(row)
        visual_cache[case_id]=(params,lt,gp,gux,guy,lp,lux,luy)
        print(f"[{index:03d}/116] {case_id}",flush=True)

    for row in case_rows:
        row["local_wake_velocity_rmse"] = row["local_wake_vector_rmse"]
        row["global_wake_velocity_rmse"] = row["global_wake_vector_rmse"]
    with (OUTPUT/"case_metrics.csv").open("x",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(case_rows[0]));w.writeheader();w.writerows(case_rows)
    with (OUTPUT/"wake_profile_metrics.csv").open("x",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(profile_rows[0]));w.writeheader();w.writerows(profile_rows)
    comparisons={
        "wake_velocity":relative_summary(case_rows,"local_wake_vector_rmse","global_wake_vector_rmse"),
        "te_velocity":relative_summary(case_rows,"local_te_vector_rmse","global_te_vector_rmse"),
        "near_wall_velocity":relative_summary(case_rows,"local_wall_vector_rmse","global_wall_vector_rmse"),
        "patch_pressure":relative_summary(case_rows,"local_patch_p_rmse","global_patch_p_rmse"),
        "te_pressure":relative_summary(case_rows,"local_te_p_rmse","global_te_p_rmse"),
        "te_pressure_difference":relative_summary(case_rows,"local_te_pressure_difference_abs_error","global_te_pressure_difference_abs_error"),
    }
    profile_comparisons={name:relative_summary(profile_rows,f"local_{name}",f"global_{name}") for name in
        ("Ux_profile_rmse","Uy_profile_rmse","Ux_min_abs_error","wake_centre_abs_error","width_abs_error")}
    visuals=choose_visual_cases(case_rows)
    for label,case_id in visuals.items():
        params,lt,gp,gux,guy,lp,lux,luy=visual_cache[case_id];u_inf=float(params["inlet_velocity"])
        target={"Cp":lt["p"]/(.5*u_inf**2),"Ux":lt["U"][...,0],"Uy":lt["U"][...,1],"speed":np.hypot(lt["U"][...,0],lt["U"][...,1])}
        glob={"Cp":gp/(.5*u_inf**2),"Ux":gux,"Uy":guy,"speed":np.hypot(gux,guy)}
        loc={"Cp":lp/(.5*u_inf**2),"Ux":lux,"Uy":luy,"speed":np.hypot(lux,luy)}
        plot_case(case_id,label,target,glob,loc,lt["xy"],u_inf)
        plot_wake_profiles(case_id, label, lt["U"], np.stack((gux, guy), axis=-1),
                           np.stack((lux, luy), axis=-1), lt["xy"], u_inf)
    summary={"experiment":"simple_unet_local_wake_v1","test_cases":len(case_rows),"common_reference":"high-resolution native-derived CFD patch",
        "global_checkpoint_sha256":hashlib.sha256(GLOBAL_CHECKPOINT.read_bytes()).hexdigest(),
        "local_checkpoint_sha256":hashlib.sha256(LOCAL_CHECKPOINT.read_bytes()).hexdigest(),
        "aggregate_case_metrics":{key:float(np.mean([float(r[key]) for r in case_rows])) for key in case_rows[0] if key.startswith(("local_","global_"))},
        "case_by_case":comparisons,"wake_profiles":profile_comparisons,"representative_cases":visuals,
        "runtime_seconds":time.perf_counter()-started}
    success=(comparisons["wake_velocity"]["local_better_cases"]>=88 and comparisons["wake_velocity"]["median_relative_improvement"]>0.1
             and comparisons["patch_pressure"]["median_relative_improvement"]>-0.1)
    mixed=(comparisons["wake_velocity"]["local_better_cases"]>58)
    summary["classification"]="LOCAL_MODEL_SUCCESS" if success else "MIXED_RESULT" if mixed else "LOCAL_MODEL_NOT_JUSTIFIED"
    (OUTPUT/"evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUTPUT/"classification.txt").write_text(summary["classification"]+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
