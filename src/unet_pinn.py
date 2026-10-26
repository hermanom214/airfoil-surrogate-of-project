"""Physics-informed U-Net data and differentiable RANS loss utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.ml_supervised import (
    NONDIMENSIONAL_TARGETS,
    SupervisedLossConfig,
    near_wall_region_mask,
    physical_to_targets,
    wake_region_mask,
)
from src.ml_training import supervised_loss_components


class PinnFlowDataset(Dataset):
    """Full-field U-Net interface for the immutable physics-ready dataset."""
    def __init__(self, root: Path, supervised_config: SupervisedLossConfig | None = None,
                 return_regions: bool = False):
        self.root = Path(root); self.files = sorted((self.root / "cases").glob("*/*.npz"))
        if not self.files: raise RuntimeError(f"No PINN artifacts under {self.root / 'cases'}")
        self.supervised_config = supervised_config or SupervisedLossConfig()
        self.return_regions = return_regions
        self.sample_params = []
        for path in self.files:
            with np.load(path, allow_pickle=False) as z: meta=json.loads(str(z["metadata"].item()))
            self.sample_params.append((float(meta["aoa_deg"]), float(meta["U_inf"])))
        self.aoa_mean=self.inlet_u_mean=self.p_mean=self.ux_mean=self.uy_mean=0.0
        self.aoa_std=self.inlet_u_std=self.p_std=self.ux_std=self.uy_std=1.0
        self.fit_condition_normalization(); self.fit_target_normalization()

    def __len__(self): return len(self.files)
    def fit_condition_normalization(self, indices: Sequence[int] | None=None):
        ids=list(range(len(self))) if indices is None else list(indices)
        values=np.asarray([self.sample_params[i] for i in ids], dtype=np.float64)
        if not values.size: raise RuntimeError("Cannot normalize an empty set")
        self.aoa_mean,self.inlet_u_mean=values.mean(0).tolist(); self.aoa_std,self.inlet_u_std=(values.std(0)+1e-8).tolist()
    def fit_target_normalization(self, indices: Sequence[int] | None=None):
        if self.supervised_config.target_representation == NONDIMENSIONAL_TARGETS:
            self.p_mean=self.ux_mean=self.uy_mean=0.0
            self.p_std=self.ux_std=self.uy_std=1.0
            return
        for name, value in self.compute_raw_target_normalization(indices).items():
            setattr(self, name, value)

    def compute_raw_target_normalization(self, indices: Sequence[int] | None=None):
        """Raw-field statistics for common evaluation metrics, independent of targets."""
        ids=list(range(len(self))) if indices is None else list(indices)
        if not ids: raise RuntimeError("Cannot normalize an empty set")
        sums=np.zeros(3); sums2=np.zeros(3); count=0
        for i in ids:
            with np.load(self.files[i],allow_pickle=False) as z:
                m=z["fluid_mask"].astype(bool); vals=np.stack((z["p"],z["U"][...,0],z["U"][...,1]))[:,m]
            sums+=vals.sum(1); sums2+=np.square(vals).sum(1); count+=vals.shape[1]
        means=sums/count; stds=np.sqrt(np.maximum(sums2/count-means**2,0))+1e-8
        return dict(zip(("p_mean", "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std"),
                        (means[0], stds[0], means[1], stds[1], means[2], stds[2])))
    def __getitem__(self, idx):
        with np.load(self.files[idx],allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files if k!="metadata"}; meta=json.loads(str(z["metadata"].item()))
        xy=a["xy"].astype(np.float32); m=a["fluid_mask"].astype(np.float32); x,y=xy[...,0],xy[...,1]
        inp=np.stack((m,(x-x.mean())/(x.std()+1e-8),(y-y.mean())/(y.std()+1e-8),
          np.full_like(m,(meta["aoa_deg"]-self.aoa_mean)/self.aoa_std),np.full_like(m,(meta["U_inf"]-self.inlet_u_mean)/self.inlet_u_std)))
        physical=np.stack((a["p"],a["U"][...,0],a["U"][...,1])).astype(np.float32)
        mean=np.asarray((self.p_mean,self.ux_mean,self.uy_mean),np.float32)[:,None,None]; std=np.asarray((self.p_std,self.ux_std,self.uy_std),np.float32)[:,None,None]
        if self.supervised_config.target_representation == NONDIMENSIONAL_TARGETS:
            target=physical_to_targets(*physical, float(meta["U_inf"]),
                self.supervised_config.target_representation, self.supervised_config.p_inf)
        else:
            target=(physical-mean)/std
        result={"input":torch.from_numpy(inp),"target":torch.from_numpy(target),"fluid_mask":torch.from_numpy(m[None]),
          "physics_mask":torch.from_numpy(a["physics_valid_mask"].astype(np.float32)[None]),"xy":torch.from_numpy(xy),
          "k":torch.from_numpy(a["k"].astype(np.float32)[None]),"omega":torch.from_numpy(a["omega"].astype(np.float32)[None]),
          "nut":torch.from_numpy(a["nut"].astype(np.float32)[None]),"u_ref":torch.tensor(float(meta["U_inf"])),
          "l_ref":torch.tensor(float(meta["chord"])),"nu":torch.tensor(float(meta["nu"])),"case_id":str(meta["case_id"])}
        if self.return_regions:
            wall=near_wall_region_mask(xy,m,float(meta["chord"]),self.supervised_config.wall_distance_fraction)
            wake=wake_region_mask(xy,m,float(meta["chord"]),float(meta["aoa_deg"]),
                self.supervised_config.wake_x_start,self.supervised_config.wake_x_end,
                self.supervised_config.wake_half_height)
            outer=(m>.5)&~wall&~wake
            result["regions"]=torch.from_numpy(np.stack((wall,wake,outer)).astype(np.float32))
        return result


def rans_residuals(physical, xy, nut, nu):
    """Installed simpleFoam/divDevReff continuum operator, acceleration units.

    OpenFOAM p already includes the isotropic turbulent contribution. The
    3D deviatoric correction is retained for U=(u,v,0). k is not in this operator.
    Convection is conservative: (U.grad)U + U div(U). This continuum
    operator does not reproduce OpenFOAM's discrete finite-volume flux scheme.
    Nested centered differences require radius two, including mixed diagonals.
    """
    p,u,v=physical[:,0:1],physical[:,1:2],physical[:,2:3]
    dx=(xy[:,0,1,0]-xy[:,0,0,0]).view(-1,1,1,1)
    dy=(xy[:,1,0,1]-xy[:,0,0,1]).view(-1,1,1,1)
    if torch.any(dx<=0) or torch.any(dy<=0): raise ValueError("xy axis ordering is invalid")
    ux,uy,vx,vy=_ddx(u,dx),_ddy(u,dy),_ddx(v,dx),_ddy(v,dy)
    rc=ux+vy; nue=nu.reshape(-1,1,1,1)+nut
    txx=2*nue*(ux-rc/3); tyy=2*nue*(vy-rc/3); txy=nue*(uy+vx)
    rx=u*ux+v*uy+u*rc+_ddx(p,dx)-_ddx(txx,dx)-_ddy(txy,dy)
    ry=u*vx+v*vy+v*rc+_ddy(p,dy)-_ddx(txy,dx)-_ddy(tyy,dy)
    return rc,rx,ry


def evaluation_mask(physics_mask):
    """Stored mask already erodes fluid AND interpolation support by radius 2."""
    mask=(physics_mask>.5).to(physics_mask.dtype).clone()
    mask[...,:2,:]=0; mask[...,-2:,:]=0
    mask[...,:,:2]=0; mask[...,:,-2:]=0
    return mask


def _ddx(f,dx):
    out=torch.zeros_like(f); out[...,1:-1]=(f[...,2:]-f[...,:-2])/(2*dx); return out
def _ddy(f,dy):
    out=torch.zeros_like(f); out[...,1:-1,:]=(f[...,2:,:]-f[...,:-2,:])/(2*dy); return out
def stencil_mask(base,radius=2):
    invalid=1-(base>.5).float()
    invalid=F.pad(invalid,(radius,radius,radius,radius),value=1.0)
    return 1-F.max_pool2d(invalid,2*radius+1,stride=1)
def _ms(value,mask,name):
    count=mask.sum()
    if int(count.detach())==0: raise RuntimeError(f"{name} has no valid stencil points")
    return (value.square()*mask).sum()/count


def unet_pinn_losses(pred,target,fluid_mask,physics_mask,xy,k,nut,nu,u_ref,l_ref,normalization,
                     supervised_weight=1.0,continuity_weight=.1,momentum_weight=.1,
                     compute_physics=True, supervised_config: SupervisedLossConfig | None=None,
                     regions: torch.Tensor | None=None):
    """Loss using the installed OpenFOAM modified-pressure formulation."""
    config=supervised_config or SupervisedLossConfig()
    if config.target_representation == NONDIMENSIONAL_TARGETS:
        us=u_ref.view(-1,1,1,1)
        physical=torch.cat((pred[:,0:1]*(.5*us.square())+config.p_inf,
                            us*(1+pred[:,1:2]),us*pred[:,2:3]),dim=1)
    else:
        means=pred.new_tensor(normalization[::2]).view(1,3,1,1); stds=pred.new_tensor(normalization[1::2]).view(1,3,1,1)
        physical=pred*stds+means
    sq=(pred-target).square(); den=fluid_mask.sum()
    if int(den.detach())==0: raise RuntimeError("supervised loss has no valid fluid cells")
    lp,lux,luy=[(sq[:,i:i+1]*fluid_mask).sum()/den for i in range(3)]
    if regions is None:
        lsup=(lp+lux+luy)/3
        supervised_parts={"data_loss":lsup,"gradient_loss":pred.new_zeros(()),
                          "global_data_loss":lsup,"near_wall_data_loss":pred.new_zeros(()),
                          "wake_data_loss":pred.new_zeros(()),"outer_data_loss":pred.new_zeros(())}
    else:
        dx=float((xy[0,0,1,0]-xy[0,0,0,0]).detach())
        dy=float((xy[0,1,0,1]-xy[0,0,0,1]).detach())
        supervised_parts=supervised_loss_components(pred,target,fluid_mask,regions,config,dx,dy)
        lsup=supervised_parts["total_loss"]
    if compute_physics:
        mask=evaluation_mask(physics_mask)
        rc,rx,ry=rans_residuals(physical,xy,nut,nu)
        ls=l_ref.view(-1,1,1,1); us=u_ref.view(-1,1,1,1)
        if torch.any(us<=0) or torch.any(ls<=0): raise ValueError("reference scales must be positive")
        lc=_ms(rc*ls/us,mask,"continuity"); lmx=_ms(rx*ls/us.square(),mask,"momentum-x"); lmy=_ms(ry*ls/us.square(),mask,"momentum-y"); lm=lmx+lmy
    else:
        lc=lmx=lmy=lm=pred.new_zeros(())
    total=supervised_weight*lsup+continuity_weight*lc+momentum_weight*lm
    return {"loss_total":total,"loss_p":lp,"loss_ux":lux,"loss_uy":luy,"loss_supervised":lsup,
      "supervised_loss":lsup,"improved_weighted_supervised_loss":lsup,
      "supervised_weighted_data_loss":supervised_parts["data_loss"],
      "supervised_gradient_loss":supervised_parts["gradient_loss"],
      "supervised_global_mse":supervised_parts["global_data_loss"],
      "supervised_near_wall_mse":supervised_parts["near_wall_data_loss"],
      "supervised_wake_mse":supervised_parts["wake_data_loss"],
      "supervised_outer_mse":supervised_parts["outer_data_loss"],
      "raw_continuity_loss":lc,"raw_momentum_x_loss":lmx,
      "raw_momentum_y_loss":lmy,"raw_momentum_loss":lm,"loss_continuity":lc,
      "loss_momentum_x":lmx,"loss_momentum_y":lmy,"loss_momentum":lm,"weighted_supervised":supervised_weight*lsup,
      "weighted_continuity":continuity_weight*lc,"weighted_momentum":momentum_weight*lm}
