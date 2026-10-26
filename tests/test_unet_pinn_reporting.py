from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.train_unet_pinn_ab import parse_args
from src.unet_pinn_ab import membership_sha256
from src.ml_models import build_model
from src.unet_pinn_reporting import pooled_mse, validate_run, validate_pair, plot_histories


@pytest.fixture
def run(tmp_path):
    meta = dict(experiment_name="unet_pinn_supervised",output_name="unet_pinn_supervised",pilot=False,
        loss_weights=dict(supervised_weight=1.,continuity_weight=0.,momentum_weight=0.),
        training_config=dict(batch_size=2,learning_rate=.001,weight_decay=0.,optimizer="adam",scheduler="none",seed=42,dataloader_seed=42,epochs=30),
        best_validation_epoch=1, selection_metric="supervised_loss",model_name="unet_pinn",dataset="pinn_dataset",
        model_config=dict(in_channels=5,out_channels=3,encoder_channels=[32,64,128],bottleneck_channels=256),
        initial_state_hash="initial",dataset_fingerprint="artifact",finite_gradient_validation=True,
        gradient_batches_checked=5340,training_duration_seconds=100.,peak_cuda_memory_allocated_mib=100.,peak_cuda_memory_reserved_mib=200.)
    meta["environment"]=dict(python_version="test",pytorch_version="test",cuda_version="test",gpu_model="test",gpu_total_memory_bytes=8192,deterministic_algorithms=True,cudnn_benchmark=False)
    ids=[f"case-{i}" for i in range(558)]
    for key,values,fp in (("train_case_ids",ids[:356],"train_fingerprint"),("validation_case_ids",ids[356:446],"validation_fingerprint"),("fixed_test_case_ids",ids[446:],"fixed_test_fingerprint")):
        meta[key]=values; meta[fp]=membership_sha256(values)
    meta["dataset_membership_fingerprint"]=membership_sha256(ids)
    meta["development_case_ids"]=ids[:446]
    meta["development_fingerprint"]=membership_sha256(ids[:446])
    metrics={k:1. for k in ("supervised_loss","loss_p","loss_ux","loss_uy","raw_continuity_loss","raw_momentum_x_loss","raw_momentum_y_loss")}
    history=[dict(epoch=i,learning_rate=.001,elapsed_training_seconds=float(i),train=dict(finite_gradients=True,gradient_batches_checked=178),validation=metrics.copy()) for i in range(1,31)]
    checkpoint={k:meta[k] for k in ("model_name","model_config","loss_weights","initial_state_hash","dataset_fingerprint","dataset_membership_fingerprint","best_validation_epoch","selection_metric","train_case_ids")}
    checkpoint.update(val_case_ids=meta["validation_case_ids"],test_case_ids=meta["fixed_test_case_ids"],optimizer_config=dict(name="adam",scheduler="none"),model_state_dict=build_model("unet_pinn",**meta["model_config"]).state_dict())
    for name,data in (("experiment_metadata.json",meta),("resolved_config.json",{k:meta[k] for k in ("experiment_name","loss_weights","training_config")}),
            ("training_history.json",{"epochs":history}),("validation_metrics.json",metrics),("test_metrics.json",dict(used_for_selection=False,**metrics))):
        (tmp_path/name).write_text(json.dumps(data))
    torch.save(checkpoint,tmp_path/"unet_pinn_airfoil.pt")
    return tmp_path


def mutate(root,name,fn):
    p=root/name; data=json.loads(p.read_text()); fn(data); p.write_text(json.dumps(data))


def test_valid_run(run):
    assert len(validate_run(run,"unet_pinn_supervised")[1])==30


def test_improved_pilot_checkpoint_tuple_matches_json_list(run):
    from src.ml_supervised import SupervisedLossConfig
    cfg = SupervisedLossConfig(target_representation="nondimensional_perturbation",
        wall_weight=3.,wake_weight=3.,gradient_loss_weight=.1).metadata()
    meta = json.loads((run/"experiment_metadata.json").read_text())
    meta.update(experiment_name="unet_improved_supervised",output_name="unet_improved_supervised",
        pilot=True,selection_metric="improved_supervised_loss",supervised_loss_config=cfg,
        gradient_batches_checked=178)
    meta["training_config"]["epochs"] = 1
    (run/"experiment_metadata.json").write_text(json.dumps(meta))
    (run/"resolved_config.json").write_text(json.dumps({k:meta[k] for k in (
        "experiment_name","loss_weights","training_config")}))
    history=json.loads((run/"training_history.json").read_text())
    history["epochs"]=history["epochs"][:1]
    (run/"training_history.json").write_text(json.dumps(history))
    path=run/"unet_pinn_airfoil.pt"
    checkpoint=torch.load(path,weights_only=True)
    checkpoint.update(selection_metric=meta["selection_metric"],supervised_loss_config=cfg)
    torch.save(checkpoint,path)
    assert isinstance(checkpoint["supervised_loss_config"]["gradient_channel_weights"],tuple)
    assert len(validate_run(run,"unet_improved_supervised",True)[1])==1
    checkpoint["supervised_loss_config"]["wake_weight"]=2.
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match="supervised_loss_config"):
        validate_run(run,"unet_improved_supervised",True)


@pytest.mark.parametrize("change", ["partial","duplicate","best","profile","weights","test"])
def test_reject_invalid_artifacts(run,change):
    if change=="partial": mutate(run,"training_history.json",lambda d:d["epochs"].pop())
    if change=="duplicate": mutate(run,"training_history.json",lambda d:d["epochs"][1].update(epoch=1))
    if change=="best": mutate(run,"experiment_metadata.json",lambda d:d.update(best_validation_epoch=31))
    if change=="profile": mutate(run,"experiment_metadata.json",lambda d:d.update(experiment_name="other"))
    if change=="weights": mutate(run,"experiment_metadata.json",lambda d:d["loss_weights"].update(continuity_weight=.2))
    if change=="test": mutate(run,"test_metrics.json",lambda d:d.update(used_for_selection=True))
    with pytest.raises(ValueError): validate_run(run,"unet_pinn_supervised")


def test_missing_checkpoint_rejected(run):
    (run/"unet_pinn_airfoil.pt").rename(run/"preserved.pt")
    with pytest.raises(FileNotFoundError): validate_run(run,"unet_pinn_supervised")


def test_wrong_checkpoint_identity_rejected(run):
    p=run/"unet_pinn_airfoil.pt"; c=torch.load(p,weights_only=True); c["initial_state_hash"]="wrong"; torch.save(c,p)
    with pytest.raises(ValueError,match="checkpoint identity"): validate_run(run,"unet_pinn_supervised")


def test_pooling_weights_cells_not_batches():
    assert pooled_mse([1.,9.],[9,1]) == pytest.approx(1.8)
    assert pooled_mse([1.,9.],[9,1]) != 5.
    with pytest.raises(ValueError): pooled_mse([],[])


def test_pilot_override_requires_pilot():
    with pytest.raises(SystemExit): parse_args(["--experiment","unet_pinn_supervised","--pilot-epochs","3"])
    assert parse_args(["--experiment","unet_pinn_supervised","--pilot","--pilot-epochs","3"]).pilot_epochs==3


def test_environment_compatibility(run):
    meta=validate_run(run,"unet_pinn_supervised")[0]
    meta.update(environment={"pytorch_version":"a"},development_fingerprint="dev",split_seed=42,test_fraction=.2,validation_split=.2)
    right=copy.deepcopy(meta); right["environment"]["pytorch_version"]="b"
    with pytest.raises(ValueError,match="environment"): validate_pair(meta,right)


def test_four_plots_use_actual_history_and_explicit_aggregation(tmp_path,monkeypatch):
    import matplotlib.axes
    captured=[]
    original=matplotlib.axes.Axes.plot
    def capture(self,x,y,*args,**kwargs):
        captured.append((list(x),list(y))); return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes,"plot",capture)
    hist=[dict(epoch=i,validation=dict(supervised_loss=i,raw_continuity_loss=i*i,raw_momentum_x_loss=4*i*i,raw_momentum_y_loss=9*i*i)) for i in range(1,31)]
    paths=plot_histories(dict(supervised=hist,physics_informed=hist),dict(supervised=2,physics_informed=3),tmp_path)
    assert len(paths)==4 and all((tmp_path/p).stat().st_size>10000 for p in paths)
    assert captured[2][0]==list(range(1,31)) and captured[2][1]==list(range(1,31))
    assert captured[4][1]==[2*i for i in range(1,31)]
