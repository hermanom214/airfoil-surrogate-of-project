"""Post-hoc diagnostics for completed improved supervised/PINN checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ml_models import build_model
from src.ml_supervised import body_coordinates, supervised_config_from_checkpoint
from src.unet_pinn import PinnFlowDataset, evaluation_mask, rans_residuals
from src.unet_pinn_ab import dataset_artifact_sha256
from src.unet_pinn_diagnostics import (aggregate_error_rows, body_to_global,
    error_statistics, gradient_magnitude, residual_statistics, select_diagnostic_cases)
from src.unet_pinn_field_evaluation import (PRESSURE_UNIT, assert_matching_normalization,
    decode_checkpoint_prediction, file_sha256, region_masks, shared_error_limit, shared_value_limits)
from src.unet_pinn_reporting import validate_pair, validate_run

OUTPUT = ROOT / "outputs/diagnostics/unet_improved_pinn_v1"
ROOTS = {"supervised": ROOT/"data/models/unet_improved_supervised",
         "pinn": ROOT/"data/models/unet_improved_pinn"}
PROFILES = {"supervised":"unet_improved_supervised", "pinn":"unet_improved_pinn"}
FIELDS = ("p", "Ux", "Uy")
UNITS = {"p":PRESSURE_UNIT, "Ux":"m/s", "Uy":"m/s"}
REGIONS = ("full_domain", "near_airfoil", "wake", "outer_flow")


def write_csv(path, rows):
    if not rows: raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def physical_tensor(fields, device):
    return torch.from_numpy(np.stack((fields["p"],fields["Ux"],fields["Uy"])).astype(np.float32))[None].to(device)


def residual_arrays(fields, sample, device):
    values=rans_residuals(physical_tensor(fields,device),sample["xy"][None].to(device),
        sample["nut"][None].to(device),sample["nu"][None].to(device))
    l=float(sample["l_ref"]); u=float(sample["u_ref"])
    return {"continuity":(values[0]*l/u)[0,0].cpu().numpy(),
            "momentum_x":(values[1]*l/u**2)[0,0].cpu().numpy(),
            "momentum_y":(values[2]*l/u**2)[0,0].cpu().numpy()}


def plot_fields(path,xy,mask,roi,reference,a,b,field,title):
    arrays=[reference,a,b,a-reference,b-reference]; extent=[xy[...,0].min(),xy[...,0].max(),xy[...,1].min(),xy[...,1].max()]
    x=xy[...,0][roi]; y=xy[...,1][roi]; limits=(x.min(),x.max(),y.min(),y.max())
    vmin,vmax=shared_value_limits(reference,roi); emax=shared_error_limit(arrays[3],arrays[4],roi)
    fig,axes=plt.subplots(1,5,figsize=(21,4.2),constrained_layout=True)
    labels=("CFD reference","Improved supervised","Improved PINN","Supervised - CFD","PINN - CFD")
    for i,(ax,array,label) in enumerate(zip(axes,arrays,labels)):
        shown=np.where(mask,array,np.nan); image=ax.imshow(shown,origin="lower",extent=extent,aspect="equal",
            cmap="turbo" if i<3 else "RdBu_r",vmin=vmin if i<3 else -emax,vmax=vmax if i<3 else emax)
        ax.set(xlim=limits[:2],ylim=limits[2:],title=label,xlabel="x [m]")
        if i==0: ax.set_ylabel("y [m]")
        fig.colorbar(image,ax=ax,shrink=.75,label=f"{field} [{UNITS[field]}]")
    fig.suptitle(title); fig.savefig(path,dpi=190); plt.close(fig)
    return {"field_limits":[float(vmin),float(vmax)],"error_limits":[-float(emax),float(emax)]}


def interpolator(xy,values):
    return RegularGridInterpolator((xy[:,0,1],xy[0,:,0]),values,bounds_error=False,fill_value=np.nan)


def plot_wake_profiles(path,xy,fields,meta):
    stations=(1.1,1.3,1.6); chord=float(meta["chord"]); angle=float(meta["aoa_deg"])
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True); rows=[]
    for ax,station in zip(axes,stations):
        yb=np.linspace(-.20,.20,241)*chord; xb=np.full_like(yb,station*chord)
        x,y=body_to_global(xb,yb,chord,angle); points=np.column_stack((y,x))
        profiles={name:interpolator(xy,value["Ux"])(points) for name,value in fields.items()}
        valid=np.logical_and.reduce([np.isfinite(v) for v in profiles.values()])
        for name,color in (("reference","black"),("supervised","#2471A3"),("pinn","#C25125")):
            ax.plot(profiles[name][valid],yb[valid]/chord,label=name,color=color)
        ref=profiles["reference"][valid]
        for name in ("supervised","pinn"):
            err=profiles[name][valid]-ref
            rows.append({"case_id":meta["case_id"],"station_x_over_c":station,"component":"global Ux",
                "model":name,"valid_samples":int(valid.sum()),"mae_m_per_s":float(np.mean(abs(err))),
                "rmse_m_per_s":float(np.sqrt(np.mean(err**2))),"minimum_ux_m_per_s":float(np.min(profiles[name][valid])),
                "reference_minimum_ux_m_per_s":float(np.min(ref)),
                "minimum_location_y_over_c":float(yb[valid][np.argmin(profiles[name][valid])]/chord)})
        ax.set(title=f"x/c={station:g}",xlabel="global Ux [m/s]",ylabel="body y/c"); ax.grid(alpha=.25)
    axes[0].legend(); fig.suptitle(f"{meta['case_id']} — global-Ux cross-wake profiles")
    fig.savefig(path,dpi=190); plt.close(fig); return rows


def plot_pressure_profiles(path,xy,sdf,valid,fields,meta):
    chord=float(meta["chord"]); xb,yb=body_coordinates(xy,chord,float(meta["aoa_deg"])); xc=xb/chord
    dx=float(xy[0,1,0]-xy[0,0,0]); dy=float(xy[1,0,1]-xy[0,0,1]); offset=2*math.hypot(dx,dy)
    band=valid&(sdf>0)&(sdf<=offset)&(xc>=0)&(xc<=1)
    bins=np.linspace(0,1,61); centers=(bins[:-1]+bins[1:])/2
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True); rows=[]
    for ax,(side,side_mask) in zip(axes,(('suction',yb>=0),('pressure',yb<0))):
        for name,color in (("reference","black"),("supervised","#2471A3"),("pinn","#C25125")):
            cp=(fields[name]["p"])/(0.5*float(meta["U_inf"])**2)
            med=[]
            for lo,hi in zip(bins[:-1],bins[1:]):
                m=band&side_mask&(xc>=lo)&(xc<hi); med.append(float(np.median(cp[m])) if m.any() else np.nan)
            ax.plot(centers,med,label=name,color=color)
            rows.extend({"case_id":meta["case_id"],"side":side,"source":name,"x_over_c":float(x),"Cp":v,
                         "sampling_offset_max_m":offset} for x,v in zip(centers,med) if np.isfinite(v))
        ax.invert_yaxis(); ax.set(title=side+"-side offset",xlabel="body x/c",ylabel="Cp"); ax.grid(alpha=.25)
    axes[0].legend(); fig.suptitle(f"{meta['case_id']} — Cartesian near-surface offset Cp (not wall Cp)")
    fig.savefig(path,dpi=190); plt.close(fig); return rows,offset


def plot_residuals(path,xy,mask,residuals,meta):
    extent=[xy[...,0].min(),xy[...,0].max(),xy[...,1].min(),xy[...,1].max()]
    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for col,component in enumerate(("continuity","momentum_x","momentum_y")):
        limit=max(np.percentile(abs(residuals[source][component][mask]),99) for source in residuals)
        for row,source in enumerate(("reference","supervised","pinn")):
            image=axes[row,col].imshow(np.where(mask,residuals[source][component],np.nan),origin="lower",extent=extent,
                aspect="equal",cmap="RdBu_r",vmin=-limit,vmax=limit)
            axes[row,col].set(title=f"{source} {component}",xlim=(-.2,1.75),ylim=(-.35,.35))
            fig.colorbar(image,ax=axes[row,col],shrink=.7)
    fig.suptitle(f"{meta['case_id']} — nondimensional residuals, shared per-component scale")
    fig.savefig(path,dpi=180); plt.close(fig)


def correlation_rows(relationships):
    output=[]
    for region in REGIONS:
        subset=[r for r in relationships if r["region"]==region]
        for x,y in (("reference_momentum_rms","p_mae_delta"),("pinn_momentum_reduction_vs_reference","p_mae_delta"),
                    ("pinn_continuity_reduction_vs_reference","ux_mae_delta")):
            a=np.asarray([r[x] for r in subset]); b=np.asarray([r[y] for r in subset])
            output.append({"region":region,"x":x,"y":y,"case_count":len(subset),
                           "pearson_r":float(np.corrcoef(a,b)[0,1]),"interpretation":"exploratory; correlation is not causation"})
    return output


def main():
    if OUTPUT.exists(): raise FileExistsError(f"Refusing to overwrite diagnostics: {OUTPUT}")
    runs={name:validate_run(ROOTS[name],PROFILES[name]) for name in ROOTS}; validate_pair(runs["supervised"][0],runs["pinn"][0])
    checkpoints={name:runs[name][3] for name in runs}; normalization=assert_matching_normalization(*checkpoints.values())
    protected=[ROOTS[n]/"unet_pinn_airfoil.pt" for n in ROOTS]+[ROOTS[n]/"split_membership.json" for n in ROOTS]
    protected += [ROOT/"data/models/unet_pinn/fixed_test_split.json"]
    hashes_before={str(p.relative_to(ROOT)):file_sha256(p) for p in protected}
    config=supervised_config_from_checkpoint(checkpoints["supervised"])
    dataset=PinnFlowDataset(ROOT/"data/pinn_dataset",supervised_config=config)
    for key,value in normalization.items(): setattr(dataset,key,value)
    fingerprint=dataset_artifact_sha256(dataset.files,dataset.root)
    if any(runs[n][0]["dataset_fingerprint"]!=fingerprint for n in runs): raise ValueError("Live dataset fingerprint changed")
    ids=runs["supervised"][0]["fixed_test_case_ids"]
    if len(ids)!=112 or ids!=runs["pinn"][0]["fixed_test_case_ids"]: raise ValueError("Fixed-test membership mismatch")
    lookup={p.parent.name:i for i,p in enumerate(dataset.files)}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models={}
    for name,checkpoint in checkpoints.items():
        model=build_model("unet_pinn",**checkpoint["model_config"]).to(device); model.load_state_dict(checkpoint["model_state_dict"],strict=True); models[name]=model.eval()
    field_rows=[]; residual_rows=[]; gradient_rows=[]; extrema_rows=[]; relationships=[]; metadata={}
    with torch.no_grad():
        for number,cid in enumerate(ids,1):
            sample=dataset[lookup[cid]]; path=dataset.files[lookup[cid]]
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z["metadata"].item())); xy=z["xy"].astype(float); fluid=(z["fluid_mask"]>.5)&(z["interpolation_valid_mask"]>.5)
                reference={"p":z["p"].astype(float),"Ux":z["U"][...,0].astype(float),"Uy":z["U"][...,1].astype(float)}
            metadata[cid]={k:meta[k] for k in ("case_id","naca_code","aoa_deg","U_inf","chord")}
            inp=sample["input"][None].to(device)
            predictions={name:decode_checkpoint_prediction(model(inp)[0].cpu().numpy(),checkpoints[name],float(meta["U_inf"])) for name,model in models.items()}
            fields={"reference":reference,**predictions}; regions=region_masks(xy,fluid,float(meta["chord"]),float(meta["aoa_deg"]))
            for region in REGIONS:
                mask=regions[region]
                for field in FIELDS:
                    stats={name:error_statistics(fields[name][field],reference[field],mask) for name in ("supervised","pinn")}
                    valid=mask&np.isfinite(reference[field])
                    row={"case_id":cid,"naca_code":meta["naca_code"],"aoa_deg":meta["aoa_deg"],"U_inf":meta["U_inf"],"region":region,"field":field,"unit":UNITS[field],"valid_cells":int(valid.sum())}
                    for name in ("supervised","pinn"):
                        err=(fields[name][field]-reference[field])[valid]; row.update({f"{name}_{k}":v for k,v in stats[name].items() if k!="count"})
                        row[f"{name}_abs_sum"]=float(abs(err).sum()); row[f"{name}_sq_sum"]=float((err**2).sum())
                    field_rows.append(row)
                    for source in fields:
                        grad,gmask=gradient_magnitude(fields[source][field],xy,mask); values=grad[gmask]
                        gradient_rows.append({"case_id":cid,"region":region,"field":field,"source":source,"valid_cells":int(values.size),
                            "mean_gradient":float(values.mean()),"p95_gradient":float(np.percentile(values,95)),"p99_gradient":float(np.percentile(values,99))})
                        vals=fields[source][field][mask]
                        extrema_rows.append({"case_id":cid,"region":region,"field":field,"source":source,
                            "minimum":float(vals.min()),"maximum":float(vals.max()),"range":float(vals.max()-vals.min())})
            physics_mask=evaluation_mask(sample["physics_mask"][None].to(device))[0,0].cpu().numpy()>0
            residuals={source:residual_arrays(value,sample,device) for source,value in fields.items()}
            case_res={}
            for region in REGIONS:
                pmask=regions[region]&physics_mask
                for source in residuals:
                    for component,array in residuals[source].items():
                        stats=residual_statistics(array,pmask); residual_rows.append({"case_id":cid,"region":region,"source":source,"component":component,"unit":"nondimensional",**stats})
                        case_res[region,source,component]=stats["rms"]
                by={(r["region"],r["field"]):r for r in field_rows if r["case_id"]==cid}
                relationships.append({"case_id":cid,"region":region,
                    "p_mae_delta":by[region,"p"]["pinn_mae"]-by[region,"p"]["supervised_mae"],
                    "ux_mae_delta":by[region,"Ux"]["pinn_mae"]-by[region,"Ux"]["supervised_mae"],
                    "reference_momentum_rms":math.hypot(case_res[region,"reference","momentum_x"],case_res[region,"reference","momentum_y"]),
                    "pinn_momentum_reduction_vs_reference":math.hypot(case_res[region,"reference","momentum_x"],case_res[region,"reference","momentum_y"])-math.hypot(case_res[region,"pinn","momentum_x"],case_res[region,"pinn","momentum_y"]),
                    "pinn_continuity_reduction_vs_reference":case_res[region,"reference","continuity"]-case_res[region,"pinn","continuity"]})
            print(f"diagnosed {number}/112 {cid}",flush=True)
    selected=select_diagnostic_cases(field_rows,metadata)
    for sub in ("tables","figures/selected_cases","figures/wake_profiles","figures/pressure_profiles","figures/residual_maps"): (OUTPUT/sub).mkdir(parents=True,exist_ok=True)
    summary=aggregate_error_rows(field_rows); write_csv(OUTPUT/"tables/per_case_field_errors.csv",field_rows); write_csv(OUTPUT/"tables/regional_error_summary.csv",summary)
    write_csv(OUTPUT/"tables/physics_residuals_per_case.csv",residual_rows)
    residual_summary=[]
    for region in REGIONS:
        for source in ("reference","supervised","pinn"):
            for component in ("continuity","momentum_x","momentum_y"):
                subset=[r for r in residual_rows if (r["region"],r["source"],r["component"])==(region,source,component)]
                values=np.asarray([r["rms"] for r in subset]); biases=np.asarray([r["bias"] for r in subset])
                smallest=sum(r["rms"]==min(x["rms"] for x in residual_rows if x["case_id"]==r["case_id"] and x["region"]==region and x["component"]==component) for r in subset)
                residual_summary.append({"region":region,"source":source,"component":component,"case_count":len(subset),"case_mean_rms":float(values.mean()),"case_median_rms":float(np.median(values)),"case_mean_bias":float(biases.mean()),"smallest_rms_cases":smallest,"smallest_rms_fraction":smallest/len(subset)})
    write_csv(OUTPUT/"tables/physics_residual_summary.csv",residual_summary); write_csv(OUTPUT/"tables/gradient_summary_per_case.csv",gradient_rows); write_csv(OUTPUT/"tables/extrema_per_case.csv",extrema_rows)
    corr=correlation_rows(relationships); write_csv(OUTPUT/"tables/error_residual_relationships.csv",relationships); write_csv(OUTPUT/"tables/error_residual_correlations.csv",corr)
    wake_rows=[]; pressure_rows=[]; plot_limits={}
    with torch.no_grad():
        for chosen in selected:
            cid=chosen["case_id"]; sample=dataset[lookup[cid]]; path=dataset.files[lookup[cid]]
            with np.load(path,allow_pickle=False) as z:
                xy=z["xy"].astype(float); fluid=(z["fluid_mask"]>.5)&(z["interpolation_valid_mask"]>.5); sdf=z["airfoil_sdf"].astype(float)
                ref={"p":z["p"].astype(float),"Ux":z["U"][...,0].astype(float),"Uy":z["U"][...,1].astype(float)}
            pred={name:decode_checkpoint_prediction(model(sample["input"][None].to(device))[0].cpu().numpy(),checkpoints[name],float(chosen["U_inf"])) for name,model in models.items()}
            fields={"reference":ref,**pred}; regions=region_masks(xy,fluid,float(chosen["chord"]),float(chosen["aoa_deg"]))
            plot_limits[cid]={}
            for field in FIELDS:
                plot_limits[cid][field]=plot_fields(OUTPUT/f"figures/selected_cases/{cid}_{field}.png",xy,fluid,regions["near_airfoil"]|regions["wake"],ref[field],pred["supervised"][field],pred["pinn"][field],field,f"{cid} — near-airfoil and wake diagnostic")
            wake_rows += plot_wake_profiles(OUTPUT/f"figures/wake_profiles/{cid}.png",xy,fields,chosen)
            p_rows,offset=plot_pressure_profiles(OUTPUT/f"figures/pressure_profiles/{cid}.png",xy,sdf,fluid,fields,chosen); pressure_rows+=p_rows; chosen["pressure_sampling_offset_max_m"]=offset
            residuals={source:residual_arrays(value,sample,device) for source,value in fields.items()}; pmask=evaluation_mask(sample["physics_mask"][None].to(device))[0,0].cpu().numpy()>0
            plot_residuals(OUTPUT/f"figures/residual_maps/{cid}.png",xy,pmask,residuals,chosen)
    write_csv(OUTPUT/"tables/wake_profiles.csv",wake_rows); write_csv(OUTPUT/"tables/near_surface_pressure_profiles.csv",pressure_rows)
    (OUTPUT/"selected_cases.json").write_text(json.dumps({"selection_is_outcome_driven":True,"cases":selected},indent=2)+"\n",encoding="utf-8")
    hashes_after={str(p.relative_to(ROOT)):file_sha256(p) for p in protected}
    if hashes_before!=hashes_after: raise RuntimeError("Protected checkpoint or split changed")
    metadata_out={"device":str(device),"population":{"total":len(dataset),"train":356,"validation":90,"fixed_test":len(ids)},"dataset_fingerprint":fingerprint,
        "fixed_test_ids":ids,"pressure_unit":PRESSURE_UNIT,"density_conversion_applied":False,"p_inf":config.p_inf,"checkpoint_hashes":hashes_before,
        "checkpoint_hashes_after":hashes_after,"normalization":normalization,"physics_operator":"src.unet_pinn.rans_residuals for reference and both predictions",
        "field_masks":"fluid_mask AND interpolation_valid_mask","physics_mask":"evaluation_mask(physics_valid_mask)","plot_limits":plot_limits}
    (OUTPUT/"metadata.json").write_text(json.dumps(metadata_out,indent=2)+"\n",encoding="utf-8")
    lines=["# Improved U-Net vs Improved U-Net + PINN diagnostic","",f"Evaluated the immutable 112-case fixed-test population on {device}. No optimization or training was performed.","",
        "Pressure is OpenFOAM kinematic pressure in m²/s². No density conversion occurred; prior Pa labeling was incorrect.","",
        "## Artifact index","","- `tables/regional_error_summary.csv`: pooled-cell and equal-case field summaries.","- `tables/physics_residual_summary.csv`: CFD/supervised/PINN residual comparison.","- `tables/error_residual_correlations.csv`: exploratory relationships.","- `selected_cases.json`: outcome-driven case selection with numerical justification.","- `figures/selected_cases/`, `wake_profiles/`, `pressure_profiles/`, `residual_maps/`: detailed diagnostics.","",
        "Near-surface Cp is sampled from a Cartesian fluid-cell offset band no farther than two grid diagonals from the analytic surface; it is not exact OpenFOAM wall Cp. Wake profiles report global Ux at body-coordinate stations.","",
        "Aerodynamic force metrics were omitted: the raster fields do not provide validated wall shear and pressure-only drag would not be total drag."]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(OUTPUT),"selected_cases":[x["case_id"] for x in selected],"protected_unchanged":True},indent=2))


if __name__=="__main__": main()
