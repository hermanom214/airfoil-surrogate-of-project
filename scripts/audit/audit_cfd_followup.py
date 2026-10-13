"""Focused read-only follow-up checks for the CFD field plausibility audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "flow_fields"
OUT = ROOT / "outputs" / "cfd_field_audit"
FIG = OUT / "figures"
UINF = 20.0

PM4 = [
    "case_0053_naca0012_aoam4p0_u20p0",
    "case_0073_naca0012_aoa4p0_u20p0",
]
TE_CASES = [
    "case_0063_naca0012_aoa0p0_u20p0",
    "case_0053_naca0012_aoam4p0_u20p0",
    "case_0073_naca0012_aoa4p0_u20p0",
    "case_0563_naca2412_aoa0p0_u20p0",
    "case_0573_naca2412_aoa4p0_u20p0",
]


def tokens(path: Path):
    with path.open("r", encoding="ascii", errors="strict") as f:
        for line in f:
            yield from line.split()


def read_legacy_vtk(path: Path, need_fields=True):
    """Read points plus FIELD-format point arrays p/U from an ASCII legacy VTK."""
    it = tokens(path)
    points = None
    fields = {}
    for tok in it:
        if tok == "POINTS":
            n = int(next(it)); next(it)
            points = np.fromiter((float(next(it)) for _ in range(3*n)), dtype=float, count=3*n).reshape(n,3)
            if not need_fields:
                return points, fields
        elif tok == "POINT_DATA":
            n = int(next(it))
            if next(it) != "FIELD": raise RuntimeError(f"Unexpected POINT_DATA layout in {path}")
            next(it); narr = int(next(it))
            for _ in range(narr):
                name, ncomp, ntuple = next(it), int(next(it)), int(next(it)); next(it)
                a = np.fromiter((float(next(it)) for _ in range(ncomp*ntuple)), dtype=float, count=ncomp*ntuple)
                fields[name] = a.reshape(ntuple,ncomp).squeeze()
            if n != len(points): raise RuntimeError(f"Point count mismatch in {path}")
            return points, fields
    raise RuntimeError(f"No POINT_DATA found in {path}")


def body_xy(points, aoa):
    a=np.deg2rad(aoa); xr=points[:,0]-0.25; y=points[:,1]
    return np.cos(a)*xr-np.sin(a)*y+0.25, np.sin(a)*xr+np.cos(a)*y


def params(case):
    return json.loads((DATA/case/"params.json").read_text(encoding="utf-8"))


def force_history(case):
    path=DATA/case/"postProcessing"/"forceCoeffs1"/"0"/"forceCoeffs.dat"
    return np.loadtxt(path, comments="#")


def residual_history(case):
    path=DATA/case/"postProcessing"/"residuals"/"0"/"residuals.dat"
    rows=[]
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#") or "N/A" in line: continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows)


def convergence_outputs():
    rows=[]
    fig,(a1,a2,a3)=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
    colors={PM4[0]:"tab:blue",PM4[1]:"tab:red"}
    labels={PM4[0]:"-4 deg",PM4[1]:"+4 deg"}
    for case in PM4:
        h=force_history(case); t,cm,cd,cl=h[:,0],h[:,1],h[:,2],h[:,3]
        for ax,v,name in ((a1,cl,"Cl"),(a2,cd,"Cd"),(a3,cm,"Cm")):
            ax.plot(t,v,color=colors[case],lw=1,label=labels[case]); ax.set_ylabel(name); ax.grid(alpha=.25)
        for n in (50,100,200):
            for name,col in (("Cm",cm),("Cd",cd),("Cl",cl)):
                z=col[-n:]
                rows.append({"case_id":case,"coefficient":name,"window":n,"final":col[-1],"mean":z.mean(),
                             "std":z.std(ddof=1),"range":z.max()-z.min(),"linear_drift_per_iteration":np.polyfit(t[-n:],z,1)[0]})
    a1.legend(); a3.set_xlabel("simpleFoam iteration")
    fig.suptitle("NACA 0012 ±4° corrected force coefficients")
    fig.savefig(FIG/"naca0012_pm4_force_convergence.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for case in PM4:
        r=residual_history(case)
        for j,name in enumerate(("p","Ux","Uy","k","omega"),start=1): axes[0].semilogy(r[:,0],r[:,j],lw=.8,label=f"{labels[case]} {name}")
        axes[1].semilogy(r[:,0],r[:,1],color=colors[case],label=f"{labels[case]} p")
    axes[0].set(ylabel="initial residual",title="All reported initial residuals"); axes[0].legend(ncol=2,fontsize=8)
    axes[1].set(xlabel="simpleFoam iteration",ylabel="p initial residual",xlim=(800,1000),title="Final 200 iterations: pressure residual")
    axes[1].legend(); [ax.grid(alpha=.25) for ax in axes]
    fig.savefig(FIG/"naca0012_pm4_residuals.png",dpi=180); plt.close(fig)
    with (OUT/"pm4_convergence_statistics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows


def native_extremum_output():
    case=PM4[1]; base=DATA/case
    z=np.load(base/f"{case}.npz"); speed=np.hypot(z["U"][...,0],z["U"][...,1]); valid=z["fluid_mask"].astype(bool)
    idx=np.unravel_index(np.nanargmax(np.where(valid,speed,np.nan)),speed.shape)
    rp=np.array([z["xy"][idx][0],z["xy"][idx][1]])
    ru=z["U"][idx]; rs=float(speed[idx])
    vtk=base/"VTK"/f"{case}_1000.vtk"; pts,f=read_legacy_vtk(vtk,True); U=f["U"][:,:2]; ns=np.linalg.norm(U,axis=1)
    xy=pts[:,:2]; dist=np.linalg.norm(xy-rp,axis=1); order=np.argsort(dist); nearest=order[0]
    local=np.where(dist<=0.015)[0]; q=np.percentile(ns[local],[0,5,25,50,75,95,100])
    # Limit plotted points to one z plane and a compact region.
    plane=np.isclose(pts[:,2],pts[:,2].max()); box=plane&(np.abs(xy[:,0]-rp[0])<.035)&(np.abs(xy[:,1]-rp[1])<.035)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    sc=a1.scatter(xy[box,0],xy[box,1],c=ns[box]/UINF,s=11,cmap="viridis"); a1.scatter(*rp,marker="x",s=80,c="red",label="raster max")
    a1.scatter(*xy[nearest],marker="o",facecolors="none",edgecolors="white",s=90,label="nearest native")
    a1.set(xlabel="x [m]",ylabel="y [m]",title="Native VTK points",aspect="equal"); a1.legend(fontsize=8); fig.colorbar(sc,ax=a1,label="|U|/Uinf")
    x=z["xy"][...,0]; y=z["xy"][...,1]; crop=(np.abs(x-rp[0])<.035)&(np.abs(y-rp[1])<.035)
    shown=np.ma.masked_where(~valid,np.where(crop,speed/UINF,np.nan)); pc=a2.pcolormesh(x,y,shown,shading="auto",cmap="viridis")
    a2.contour(x,y,valid.astype(float),levels=[.5],colors="k",linewidths=.7); a2.scatter(*rp,marker="x",s=80,c="red")
    a2.set(xlim=(rp[0]-.035,rp[0]+.035),ylim=(rp[1]-.035,rp[1]+.035),xlabel="x [m]",ylabel="y [m]",title="640×320 raster",aspect="equal")
    fig.colorbar(pc,ax=a2,label="|U|/Uinf"); fig.suptitle("NACA 0012 +4° native vs raster velocity extremum")
    fig.savefig(FIG/"naca0012_native_vs_raster_velocity_extremum.png",dpi=190); plt.close(fig)
    result={"raster":{"xy":rp.tolist(),"U":ru.tolist(),"speed":rs,"speed_ratio":rs/UINF},
            "nearest_native":{"xy":xy[nearest].tolist(),"distance":float(dist[nearest]),"U":U[nearest].tolist(),"speed":float(ns[nearest]),"speed_ratio":float(ns[nearest]/UINF)},
            "native_radius_m":.015,"native_count":int(len(local)),"native_speed_percentiles_0_5_25_50_75_95_100":q.tolist(),
            "native_local_max_ratio":float(q[-1]/UINF)}
    (OUT/"native_velocity_extremum.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def surface_te(case):
    aoa=float(params(case)["aoa_deg"]); path=DATA/case/"VTK"/"airfoil"/"airfoil_1000.vtk"
    pts,f=read_legacy_vtk(path,True); xb,yb=body_xy(pts,aoa); cp=f["p"]/(.5*20**2)
    # Collapse the duplicated front/back points at identical x/y.
    key=np.round(np.c_[xb,yb],8); _,ind=np.unique(key,axis=0,return_index=True); xb,yb,cp=xb[ind],yb[ind],cp[ind]
    aft=(xb>=.80)&(xb<=1.005); xb,yb,cp=xb[aft],yb[aft],cp[aft]
    upper=yb>=0; lower=~upper
    out={}
    for name,m in (("upper",upper),("lower",lower)):
        order=np.argsort(xb[m]); xx,cc=xb[m][order],cp[m][order]
        out[name]=(xx,cc)
    return out


def te_outputs():
    diagnostics=[]; surfaces={case:surface_te(case) for case in TE_CASES}
    groups=[("trailing_edge_cp_0012_0deg.png",[TE_CASES[0]]),
            ("trailing_edge_cp_0012_pm4.png",TE_CASES[1:3]),
            ("trailing_edge_cp_2412.png",TE_CASES[3:])]
    for filename,cases in groups:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for case in cases:
            label=case.split("_naca",1)[1].replace("_"," ")
            for side,style in (("upper","-"),("lower","--")):
                xx,cc=surfaces[case][side]; ax.plot(xx,cc,style,lw=1.5,label=f"{label} {side}")
        ax.set(xlim=(.80,1.002),xlabel="x/c (body)",ylabel="Cp",title="Native airfoil-patch trailing-edge pressure")
        ax.grid(alpha=.25); ax.legend(fontsize=8); fig.savefig(FIG/filename,dpi=190); plt.close(fig)
    for case in TE_CASES:
        for side in ("upper","lower"):
            xx,cc=surfaces[case][side]
            pick=lambda x: int(np.argmin(abs(xx-x)))
            i90,i95,i97,i98,i99=map(pick,(.90,.95,.97,.98,.99))
            last=np.argsort(xx)[-4:]; diffs=np.diff(cc[last]); prior=np.diff(cc[(xx>=.90)&(xx<xx[last[0]])])
            diagnostics.append({"case_id":case,"side":side,"x90":xx[i90],"Cp90":cc[i90],"x95":xx[i95],"Cp95":cc[i95],
                "x97":xx[i97],"Cp97":cc[i97],"x98":xx[i98],"Cp98":cc[i98],"x99":xx[i99],"Cp99":cc[i99],
                "delta_Cp_90_99":cc[i99]-cc[i90],"last_sample_x":xx[-1],"last_sample_Cp":cc[-1],
                "max_abs_final_three_dCp":float(np.max(abs(diffs))),"median_abs_prior_dCp":float(np.median(abs(prior))) if len(prior) else np.nan})
    with (OUT/"trailing_edge_cp_diagnostics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=diagnostics[0].keys()); w.writeheader(); w.writerows(diagnostics)
    return diagnostics


def wake_resolution_output():
    rows=[]
    raster_widths={}
    with (OUT/"quantitative_probes.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["probe"].startswith("wake_x_"):
                raster_widths[(r["case_id"],round(float(r["x_over_c"]),2))]=float(r["secondary_x_over_c"])
    for case in TE_CASES:
        aoa=float(params(case)["aoa_deg"]); path=DATA/case/"VTK"/f"{case}_1000.vtk"
        pts,_=read_legacy_vtk(path,False); xy=np.unique(np.round(pts[:,:2],10),axis=0); xb,yb=body_xy(np.c_[xy,np.zeros(len(xy))],aoa)
        region=(xb>=.95)&(xb<=1.75)&(np.abs(yb)<=.20); q=np.c_[xb[region],yb[region]]
        for station in (1.05,1.25,1.50):
            # Directional nearest-neighbour spacings in a compact body-coordinate box.
            # This remains meaningful when the global mesh lines are tilted in body coordinates.
            band=q[(np.abs(q[:,0]-station)<=.04)&(np.abs(q[:,1])<=.04)]
            dd=band[:,None,:]-band[None,:,:]; ad=np.abs(dd); d2=np.sum(dd**2,axis=2); np.fill_diagonal(d2,np.inf)
            stream=np.where(ad[:,:,1] <= .20*ad[:,:,0],d2,np.inf)
            transverse=np.where(ad[:,:,0] <= .20*ad[:,:,1],d2,np.inf)
            dxs=np.sqrt(np.min(stream,axis=1)); dxs=dxs[np.isfinite(dxs)]
            dys=np.sqrt(np.min(transverse,axis=1)); dys=dys[np.isfinite(dys)]
            width=raster_widths.get((case,round(station,2)),np.nan)
            rows.append({"case_id":case,"station_x_over_c":station,"native_points_region":len(q),
                         "median_dx_native":np.median(dxs),"p10_dx_native":np.percentile(dxs,10),
                         "median_dy_native":np.median(dys),"p10_dy_native":np.percentile(dys,10),
                         "raster_half_deficit_width":width,"estimated_native_samples_across_width":width/np.median(dys)})
    with (OUT/"native_wake_resolution.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    x=np.arange(len(TE_CASES)); dy=[]; dx=[]
    for case in TE_CASES:
        rr=[r for r in rows if r["case_id"]==case]; dx.append(np.median([r["median_dx_native"] for r in rr])); dy.append(np.median([r["median_dy_native"] for r in rr]))
    ax.bar(x-.18,dx,.36,label="native median dx"); ax.bar(x+.18,dy,.36,label="native median dy")
    ax.axhline(2.5/639,color="k",ls="--",label="raster dx=0.00391c"); ax.axhline(1.5/319,color="gray",ls=":",label="raster dy=0.00470c")
    ax.set_xticks(x,[c.split("_")[1]+" "+c.split("_aoa")[1].split("_")[0] for c in TE_CASES],rotation=20)
    ax.set(ylabel="spacing / chord",title="Native wake-region point spacing versus raster"); ax.set_yscale("log"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.savefig(FIG/"native_vs_raster_wake_resolution.png",dpi=190); plt.close(fig)
    return rows


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    convergence_outputs(); native_extremum_output(); te_outputs(); wake_resolution_output()


if __name__ == "__main__": main()
