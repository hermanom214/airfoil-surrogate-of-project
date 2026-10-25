"""Small read-only reference-field residual audit for Phase A."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from src.continuity_audit import region_masks
from src.unet_pinn import stencil_mask, rans_residuals, evaluation_mask

def audit(path: Path) -> dict[str, object]:
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files if k!="metadata"}; meta=json.loads(str(z["metadata"].item()))
    xy=torch.from_numpy(a["xy"])[None]; p=torch.from_numpy(a["p"])[None,None]; u=torch.from_numpy(a["U"][...,0])[None,None]; v=torch.from_numpy(a["U"][...,1])[None,None]
    k=torch.from_numpy(a["k"])[None,None]; nut=torch.from_numpy(a["nut"])[None,None]; base=torch.from_numpy(a["physics_valid_mask"].astype(np.float32))[None,None]
    rc,rx,ry=rans_residuals(torch.cat((p,u,v),1),xy,nut,torch.tensor([float(meta["nu"])]))
    scale_c=float(meta["chord"])/float(meta["U_inf"]); scale_m=float(meta["chord"])/float(meta["U_inf"])**2
    values={"continuity":(rc*scale_c)[0,0].numpy(),"momentum_x":(rx*scale_m)[0,0].numpy(),"momentum_y":(ry*scale_m)[0,0].numpy()}
    valid=evaluation_mask(base)[0,0].bool().numpy(); regions=region_masks(a["xy"],a["airfoil_sdf"],valid,float(meta["chord"]),float(meta["aoa_deg"]))
    old_valid=stencil_mask(base,2)[0,0].bool().numpy()
    old_regions=region_masks(a["xy"],a["airfoil_sdf"],old_valid,float(meta["chord"]),float(meta["aoa_deg"]))
    fluid_regions=region_masks(a["xy"],a["airfoil_sdf"],a["fluid_mask"].astype(bool),float(meta["chord"]),float(meta["aoa_deg"]))
    out={"case_id":meta["case_id"],"fluid_count":int(a["fluid_mask"].sum()),"physics_count":int(base.sum()),
         "evaluated_count":int(valid.sum()),"region_counts":{r:int(m.sum()) for r,m in regions.items()},
         "old_region_counts":{r:int(m.sum()) for r,m in old_regions.items()},
         "fluid_region_counts":{r:int(m.sum()) for r,m in fluid_regions.items()}}
    from src.pinn_dataset import build_physics_valid_mask
    required=build_physics_valid_mask(a["fluid_mask"],a["interpolation_valid_mask"],2).astype(bool)
    out["unsafe_stored_stencil_count"]=int((valid&~required).sum())
    for name,value in values.items():
        out[name+"_rms"]=float(np.sqrt(np.mean(value[valid]**2)))
        out[name+"_region_rms"]={r:float(np.sqrt(np.mean(value[m]**2))) for r,m in regions.items() if m.any()}
    return out

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("paths",nargs="+",type=Path); args=parser.parse_args()
    print(json.dumps([audit(path) for path in args.paths],indent=2))
if __name__ == "__main__": main()
