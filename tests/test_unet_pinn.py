from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ml_models import SimpleUNet, build_model
from src.unet_pinn import stencil_mask, unet_pinn_losses, rans_residuals, evaluation_mask


def fields(nx=17, ny=13):
    x,y=np.meshgrid(np.linspace(-1,1,nx),np.linspace(-.5,.5,ny))
    xy=torch.tensor(np.stack((x,y),-1),dtype=torch.float32)[None]
    mask=torch.ones((1,1,ny,nx)); k=torch.ones_like(mask)*.2; nut=torch.ones_like(mask)*1e-3
    return xy,mask,k,nut


def loss_for(physical, target=None, **kwargs):
    xy,mask,k,nut=fields(physical.shape[-1],physical.shape[-2])
    return unet_pinn_losses(physical, physical.detach() if target is None else target, mask, mask, xy,k,nut,
        torch.tensor([1.5e-5]),torch.tensor([10.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.),**kwargs)


def test_model_reuses_exact_unet_architecture():
    model=build_model("unet_pinn",in_channels=5,out_channels=3,encoder_channels=[4,8,16],bottleneck_channels=32)
    assert type(model) is SimpleUNet
    assert model(torch.zeros(2,5,32,64)).shape == (2,3,32,64)
    with pytest.raises(NotImplementedError): build_model("mlp_pinn")


def test_supervised_components_mask_solids():
    pred=torch.zeros(1,3,13,17); target=pred.clone(); target[:,:,0,0]=100
    xy,mask,k,nut=fields(); mask[:,:,0,0]=0
    losses=unet_pinn_losses(pred,target,mask,torch.ones_like(mask),xy,k,nut,torch.tensor([1.5e-5]),torch.tensor([10.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.))
    assert losses["loss_supervised"] == 0


def test_continuity_constant_divergence_free_and_divergent():
    xy,_,_,_=fields(); x=xy[...,0]; y=xy[...,1]
    constant=torch.cat((torch.zeros_like(x)[:,None],torch.ones_like(x)[:,None]*3,torch.ones_like(x)[:,None]*-2),1)
    assert loss_for(constant)["loss_continuity"] < 1e-12
    free=torch.cat((torch.zeros_like(x)[:,None],x[:,None],-y[:,None]),1)
    assert loss_for(free)["loss_continuity"] < 1e-10
    divergent=torch.cat((torch.zeros_like(x)[:,None],x[:,None],y[:,None]),1)
    assert loss_for(divergent)["loss_continuity"] == pytest.approx(.04,rel=1e-4)


def test_uniform_rans_zero_and_backward_gradient():
    pred=torch.zeros(1,3,13,17,requires_grad=True); pred.data[:,1]=2
    losses=loss_for(pred)
    assert losses["loss_momentum"] < 1e-12
    perturbed=pred + .01*torch.randn_like(pred)
    losses=loss_for(perturbed); losses["loss_total"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


def test_physics_loss_reaches_unet_parameters():
    model=build_model("unet_pinn",in_channels=5,out_channels=3,encoder_channels=[4,8,16],bottleneck_channels=32)
    pred=model(torch.randn(1,5,16,24)); xy,mask,k,nut=fields(24,16)
    losses=unet_pinn_losses(pred,torch.zeros_like(pred),mask,mask,xy,k,nut,torch.tensor([1.5e-5]),torch.tensor([10.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.))
    (losses["loss_continuity"]+losses["loss_momentum"]).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())


def test_stencils_exclude_invalid_neighbors_and_empty_fails():
    mask=torch.ones(1,1,9,9); mask[:,:,4,4]=0
    strict=stencil_mask(mask,2)
    assert not strict[:,:,2:7,2:7].any()
    pred=torch.zeros(1,3,9,9)
    xy,_,k,nut=fields(9,9)
    with pytest.raises(RuntimeError,match="no valid stencil"):
        unet_pinn_losses(pred,pred,torch.ones(1,1,9,9),torch.zeros(1,1,9,9),xy,k,nut,torch.tensor([1.5e-5]),torch.tensor([10.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.))


def test_variable_nut_gradient_and_kinematic_pressure_cancel():
    xy,mask,k,_=fields(); x=xy[...,0]; y=xy[...,1]
    # u=y gives tau_xy=nu+nut. With nut=y, div(tau)_x=1; kinematic p=x gives dp/dx=1.
    physical=torch.cat((x[:,None],y[:,None],torch.zeros_like(y)[:,None]),1)
    nut=(y-y.min()+.01)[:,None]
    losses=unet_pinn_losses(physical,physical,mask,mask,xy,torch.zeros_like(k),nut,torch.tensor([1.5e-5]),
        torch.tensor([1.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.))
    assert losses["loss_momentum_x"] < 1e-10
    assert losses["loss_momentum_y"] < 1e-10


def test_k_gradient_does_not_enter_modified_pressure_residual():
    xy,mask,k,nut=fields(); pred=torch.ones(1,3,13,17)
    losses=unet_pinn_losses(pred,pred,mask,mask,xy,xy[...,0][:,None]+2,nut,
        torch.tensor([.01]),torch.tensor([1.]),torch.tensor([1.]),(0.,1.,0.,1.,0.,1.))
    assert losses["loss_momentum"] == 0


def test_mixed_derivatives_and_three_dimensional_trace():
    xy,mask,k,nut=fields(); x,y=xy[...,0][:,None],xy[...,1][:,None]
    # u=x*y, v=x*y: div(U)=x+y. Both mixed derivatives equal one.
    physical=torch.cat((0*x,x*y,x*y),1)
    rc,rx,ry=rans_residuals(physical,xy,torch.zeros_like(nut),torch.tensor([.3]))
    valid=evaluation_mask(mask).bool()
    conv=x*y*(x+y)
    assert torch.allclose(rc[valid],(x+y)[valid],atol=1e-6)
    # div(grad U + grad U^T - 2/3 I div U) = (1/3,1/3).
    assert torch.allclose(rx[valid],(conv-.1)[valid],atol=2e-6)
    assert torch.allclose(ry[valid],(conv-.1)[valid],atol=2e-6)


def test_scaling_and_pressure_sign():
    xy,mask,k,nut=fields(); x,y=xy[...,0][:,None],xy[...,1][:,None]
    physical=torch.cat((-x*x/2,x,0*y),1)
    rc,rx,ry=rans_residuals(physical,xy,0*nut,torch.tensor([0.]))
    valid=evaluation_mask(mask).bool()
    assert torch.max(rx[valid].abs())<1e-6
    assert torch.max(ry[valid].abs())==0
    losses=unet_pinn_losses(physical,physical,mask,mask,xy,k,0*nut,torch.tensor([0.]),
        torch.tensor([2.]),torch.tensor([3.]),(0.,1.,0.,1.,0.,1.))
    assert losses["loss_continuity"] == pytest.approx(2.25)
    physical=torch.cat((2*x+3*y,0*x,0*y),1)
    losses=unet_pinn_losses(physical,physical,mask,mask,xy,k,0*nut,torch.tensor([0.]),
        torch.tensor([2.]),torch.tensor([3.]),(0.,1.,0.,1.,0.,1.))
    assert losses["loss_momentum_x"] == pytest.approx(2.25)
    assert losses["loss_momentum_y"] == pytest.approx(5.0625,rel=1e-5)


def test_stored_mask_is_not_eroded_twice():
    from src.pinn_dataset import build_physics_valid_mask
    base=np.ones((17,21),np.uint8); base[8,10]=0
    stored=build_physics_valid_mask(base,base,2)
    actual=evaluation_mask(torch.from_numpy(stored)[None,None].float())[0,0].numpy()
    assert np.array_equal(actual,stored)
    assert not actual[6:11,8:13].any()
    assert actual[8,13] == 1


@pytest.mark.parametrize("dev,test", [(["a","a"],["c"]),(["a","b"],["b","c"]),(["a"],["c"]),(["a","b"],["z"])])
def test_loaded_split_rejects_bad_partitions(tmp_path,dev,test):
    import json
    from src.ml_data_split import load_or_create_fixed_test_split
    path=tmp_path/"split.json"
    load_or_create_fixed_test_split(["a","b","c"],path)
    raw=json.loads(path.read_text()); raw["development_case_ids"]=dev; raw["test_case_ids"]=test
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError,match="partition"):
        load_or_create_fixed_test_split(["a","b","c"],path)


@pytest.mark.parametrize("name,subdir",[("unet_pinn","unet_pinn_phase_a"),("simple_unet","simple_unet")])
def test_evaluator_synthetic_checkpoint_resolution(tmp_path,name,subdir,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from src.config import load_ml_models_config
    from scripts.evaluate_ml_model import resolve_evaluation_checkpoint
    from src.evaluate_io import evaluate_get_model_kwargs
    cfg=load_ml_models_config(Path("configs/ml_models_config.yaml"))
    kwargs=dict(in_channels=5,out_channels=3,encoder_channels=[4,8,16],bottleneck_channels=32)
    model=build_model(name,**kwargs)
    path=tmp_path/cfg.output.models_subdir/subdir/(name+"_airfoil.pt")
    path.parent.mkdir(parents=True)
    torch.save({"model_name":name,"model_config":kwargs,"model_state_dict":model.state_dict()},path)
    resolved,_=resolve_evaluation_checkpoint(SimpleNamespace(project_root=tmp_path),cfg,name)
    assert resolved==path
    checkpoint=torch.load(resolved,weights_only=True)
    restored=build_model(checkpoint["model_name"],**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert restored(torch.zeros(1,5,16,24)).shape==(1,3,16,24)
    assert evaluate_get_model_kwargs(name,cfg)["in_channels"]==5
    with pytest.raises(FileNotFoundError):
        resolve_evaluation_checkpoint(SimpleNamespace(project_root=tmp_path),cfg,name,"missing")
    if name == "unet_pinn":
        import scripts.evaluate_ml_model as evaluator
        monkeypatch.setattr(evaluator,"load_paths",lambda _:SimpleNamespace(project_root=tmp_path,flow_fields_output=tmp_path/"flow_fields"))
        def inspect_dataset(root):
            assert root==tmp_path/"data"/"pinn_dataset"
            raise RuntimeError("PINN dataset selected")
        monkeypatch.setattr(evaluator,"PinnFlowDataset",inspect_dataset)
        with pytest.raises(RuntimeError,match="PINN dataset selected"):
            evaluator.run_validation("cpu",None,"unet_pinn")
