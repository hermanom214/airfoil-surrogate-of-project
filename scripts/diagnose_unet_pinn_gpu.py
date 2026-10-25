"""Bounded full-grid GPU diagnostic. Never invokes the training entrypoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_ml_models_config
from src.continuity_audit import region_masks
from src.ml_data_split import build_holdout_fold, load_or_create_fixed_test_split
from src.ml_models import build_model
import src.unet_pinn as physics

ROOT=Path(__file__).resolve().parents[1]
CASES=['case_0001_naca0008_aoam4p0_u15p0','case_0624_naca2416_aoa4p0_u22p5','case_0063_naca0012_aoa0p0_u20p0']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def memory():
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                free=free,total=total)


def clean():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def loss(model,batch,norm,weights):
    pred=model(batch['input'])
    values=physics.unet_pinn_losses(pred,batch['target'],batch['fluid_mask'],batch['physics_mask'],
        batch['xy'],batch['k'],batch['nut'],batch['nu'],batch['u_ref'],batch['l_ref'],norm,**weights)
    return pred,values


def new_model(kwargs):
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    return build_model('unet_pinn',**kwargs).to('cuda:0').train()


def cuda_batch(cpu):
    return {k:(v.to('cuda:0',non_blocking=True) if torch.is_tensor(v) and k!='omega' else v) for k,v in cpu.items()}


def memory_attempt(dataset,indices,size,kwargs,norm,weights,cfg,output):
    """One fresh model, one full combined backward, at most one Adam step."""
    clean(); torch.cuda.reset_peak_memory_stats()
    row={'batch_size':size,'stages':{},'success':False}
    model=optimizer=batch=pred=values=None
    original_rans=physics.rans_residuals; original_ms=physics._ms
    def mark(name):
        row['stages'][name]=memory()
        print('MEMORY',size,name,json.dumps(row['stages'][name]),flush=True)
    # Observe the real functions; do not substitute a second physical equation.
    def observed_rans(*args):
        mark('supervised_loss_before_residuals')
        result=original_rans(*args)
        mark('continuity_and_momentum_residuals')
        return result
    def observed_ms(value,mask,name):
        result=original_ms(value,mask,name)
        mark(name+'_loss_reduction')
        return result
    try:
        row['baseline']=memory()
        cpu=next(iter(DataLoader(Subset(dataset,indices[:size]),batch_size=size,pin_memory=True)))
        mark('CPU_dataset_batch_loading')
        model=new_model(kwargs)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.training.learning_rate,weight_decay=0.)
        mark('model_and_optimizer_created')
        batch=cuda_batch(cpu); mark('GPU_batch_transfer')
        row['tensors']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'device':str(v.device)} for k,v in batch.items() if torch.is_tensor(v)}
        assert batch['input'].shape==(size,5,320,640)
        assert all(v.device==torch.device('cuda:0') for k,v in batch.items() if torch.is_tensor(v) and k!='omega')
        assert next(model.parameters()).device==torch.device('cuda:0')
        pred=model(batch['input']); mark('forward')
        physics.rans_residuals=observed_rans; physics._ms=observed_ms
        values=physics.unet_pinn_losses(pred,batch['target'],batch['fluid_mask'],batch['physics_mask'],batch['xy'],
            batch['k'],batch['nut'],batch['nu'],batch['u_ref'],batch['l_ref'],norm,**weights)
        mark('full_loss')
        row['losses']={k:float(v.detach()) for k,v in values.items()}
        assert all(np.isfinite(v) for v in row['losses'].values())
        mask=physics.evaluation_mask(batch['physics_mask'])
        row['mask_counts']={'fluid_per_case':batch['fluid_mask'].sum((1,2,3)).tolist(),'physics_per_case':mask.sum((1,2,3)).tolist()}
        assert all(v>0 for v in row['mask_counts']['physics_per_case'])
        assert torch.allclose(values['loss_total'],sum(values['loss_'+key]*weights[key+'_weight'] for key in ('supervised','continuity','momentum')))
        optimizer.zero_grad(set_to_none=True); values['loss_total'].backward(); mark('backward')
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before=[p.detach().cpu().clone() for p in model.parameters()]
        optimizer.step(); mark('one_Adam_step')
        row['parameters_changed']=any(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,model.parameters()))
        row['updated_parameters_finite']=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        row['gradients_finite']=True
        assert row['parameters_changed'] and row['updated_parameters_finite']
        row['success']=True
    except torch.cuda.OutOfMemoryError as exc:
        row['failure']='CUDA OOM: '+str(exc)
        row['failure_memory']=memory()
        print(row['failure'],flush=True)
    finally:
        physics.rans_residuals=original_rans; physics._ms=original_ms
        model=optimizer=batch=pred=values=None
        clean()
        (output/f'memory_batch_{size}.json').write_text(json.dumps(row,indent=2))
    return row


def component_gradients(dataset,indices,size,kwargs,norm,weights):
    clean(); model=new_model(kwargs)
    batch=cuda_batch(next(iter(DataLoader(Subset(dataset,indices[:size]),batch_size=size,pin_memory=True))))
    rows={}; vectors={}
    for component in ('supervised','continuity','momentum'):
        model.zero_grad(set_to_none=True)
        pred,values=loss(model,batch,norm,weights); pred.retain_grad()
        values['loss_'+component].backward()
        vector=torch.cat([p.grad.detach().reshape(-1).cpu() for p in model.parameters()])
        assert torch.isfinite(vector).all()
        raw=float(torch.linalg.vector_norm(vector.double()))
        rows[component]={'raw_gradient_norm':raw,'weight':weights[component+'_weight'],
                         'weighted_gradient_norm':raw*weights[component+'_weight'],
                         'output_channel_gradient_norms':pred.grad.detach().square().sum((0,2,3)).sqrt().cpu().tolist(),
                         'raw_loss':float(values['loss_'+component].detach())}
        vectors[component]=vector
        del pred,values
        print('GRADIENT',component,json.dumps(rows[component]),flush=True)
    denominator=rows['supervised']['weighted_gradient_norm']
    for row in rows.values(): row['weighted_ratio_to_supervised']=row['weighted_gradient_norm']/denominator
    cosines={}
    for a,b in [('supervised','continuity'),('supervised','momentum'),('continuity','momentum')]:
        cosines[a+'__'+b]=float(torch.nn.functional.cosine_similarity(vectors[a].double(),vectors[b].double(),dim=0))
    del model,batch,vectors; clean()
    return {'components':rows,'cosine_similarities':cosines,'same_initial_parameters':True,'optimizer_steps':0}


def per_case_losses(dataset,indices,kwargs,norm,weights):
    clean(); model=new_model(kwargs); rows=[]
    with torch.no_grad():
        for index in indices:
            batch=cuda_batch(next(iter(DataLoader(Subset(dataset,[index]),batch_size=1,pin_memory=True))))
            pred,values=loss(model,batch,norm,weights)
            rows.append({'case_id':dataset.files[index].parent.name,'losses':{k:float(v) for k,v in values.items()}})
            assert all(np.isfinite(v) for v in rows[-1]['losses'].values())
            del batch,pred,values
    del model; clean(); return rows


def sensitivity(path):
    # Reuse the audit's exact triangulation/geometry logic without changing its
    # production API: capture its existing affected mask at its return event.
    from src.audit_pinn_interpolation import audit_interpolation
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files if k!='metadata'}; meta=json.loads(str(z['metadata'].item()))
    captured={}
    def capture(frame,event,arg):
        if frame.f_code is audit_interpolation.__code__ and event=='return':
            captured['affected']=frame.f_locals['affected'].copy()
    previous=sys.getprofile(); sys.setprofile(capture)
    try: provenance=audit_interpolation(path,meta)
    finally: sys.setprofile(previous)
    physical=torch.from_numpy(np.stack((a['p'],a['U'][...,0],a['U'][...,1])))[None]
    with torch.no_grad():
        residuals=physics.rans_residuals(physical,torch.from_numpy(a['xy'])[None],torch.from_numpy(a['nut'])[None,None],torch.tensor([meta['nu']]))
    scales=[meta['chord']/meta['U_inf'],meta['chord']/meta['U_inf']**2,meta['chord']/meta['U_inf']**2]
    residuals=[(r*s)[0,0].numpy().astype(np.float64) for r,s in zip(residuals,scales)]
    current=physics.evaluation_mask(torch.from_numpy(a['physics_valid_mask'])[None,None].float())[0,0].bool().numpy()
    excluded=current&captured['affected']; filtered=current&~excluded
    regions=region_masks(a['xy'],a['airfoil_sdf'],current,meta['chord'],meta['aoa_deg'])
    result={'case_id':meta['case_id'],'provenance':provenance,'regions':{}}
    for region,mask in {'global':current,**regions}.items():
        keep=mask&filtered; removed=mask&excluded
        result['regions'][region]={'valid_count':int(mask.sum()),'excluded_count':int(removed.sum()),'retained_count':int(keep.sum()),
            'RMS_current':{name:float(np.sqrt(np.mean(r[mask]**2))) for name,r in zip(['continuity','momentum_x','momentum_y'],residuals)},
            'RMS_excluded':{name:float(np.sqrt(np.mean(r[keep]**2))) for name,r in zip(['continuity','momentum_x','momentum_y'],residuals)},
            'fraction_residual_sum_squares_removed':{name:float(np.sum(r[removed]**2)/np.sum(r[mask]**2)) for name,r in zip(['continuity','momentum_x','momentum_y'],residuals)}}
    return result


def checkpoint_safety(kwargs,cfg,output):
    from scripts.evaluate_ml_model import resolve_evaluation_checkpoint
    model=build_model('unet_pinn',**kwargs)
    with tempfile.TemporaryDirectory(prefix='checkpoint_',dir=output) as temporary:
        root=Path(temporary); experiment=cfg.experiment_for('unet_pinn').output_name
        path=root/cfg.output.models_subdir/experiment/cfg.output.filename_template.format(model_name='unet_pinn')
        path.parent.mkdir(parents=True)
        torch.save({'model_name':'unet_pinn','model_config':kwargs,'model_state_dict':model.state_dict()},path)
        resolved,_=resolve_evaluation_checkpoint(SimpleNamespace(project_root=root),cfg,'unet_pinn',experiment)
        checkpoint=torch.load(resolved,map_location='cpu',weights_only=True)
        assert checkpoint['model_name']=='unet_pinn'
        restored=build_model(checkpoint['model_name'],**checkpoint['model_config'])
        restored.load_state_dict(checkpoint['model_state_dict'])
        assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
        return {'save_load_equal':True,'resolved_relative_path':str(resolved.relative_to(root)),
                'temporary_directory_removed_on_exit':True,'identity_guard_in_evaluator':"disagrees with checkpoint model" in inspect.getsource(__import__('scripts.evaluate_ml_model',fromlist=['run_validation']).run_validation)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    if ROOT/'outputs' not in output.parents: raise ValueError('Output must be a dedicated child of outputs/')
    output.mkdir(parents=True,exist_ok=False)
    cfg=load_ml_models_config(ROOT/'configs/ml_models_config.yaml')
    split_path=ROOT/'data/models/unet_pinn/fixed_test_split.json'
    assert split_path.is_file()
    raw=json.loads(split_path.read_text())
    split=load_or_create_fixed_test_split(raw['dataset_case_ids'],split_path)
    assert (len(split.development_case_ids),len(split.test_case_ids))==(446,112)
    assert all(i in split.development_case_ids and i not in split.test_case_ids for i in CASES)
    protected=[split_path,ROOT/'configs/ml_models_config.yaml',*sorted((ROOT/'data/models').rglob('*.pt'))]
    protected_hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    dataset=physics.PinnFlowDataset.__new__(physics.PinnFlowDataset)
    dataset.root=ROOT/'data/pinn_dataset'
    dataset.files=[dataset.root/'cases'/i/(i+'.npz') for i in split.development_case_ids]
    dataset.sample_params=[]
    for path in dataset.files:
        with np.load(path,allow_pickle=False) as z: meta=json.loads(str(z['metadata'].item()))
        dataset.sample_params.append((float(meta['aoa_deg']),float(meta['U_inf'])))
    fold=build_holdout_fold(list(range(len(dataset.files))),split.development_case_ids,cfg.training.validation_split,cfg.experiment_for('unet_pinn').data_split.seed)
    dataset.fit_condition_normalization(fold.train_indices)
    print('Fitting normalization on',len(fold.train_indices),'fold-training development cases; no fixed-test fields',flush=True)
    dataset.fit_target_normalization(fold.train_indices)
    norm=tuple(getattr(dataset,k) for k in ('p_mean','p_std','ux_mean','ux_std','uy_mean','uy_std'))
    indices=[split.development_case_ids.index(i) for i in CASES]
    weights=cfg.experiment_for('unet_pinn').pinn_loss_weights
    kwargs=asdict(cfg.model.params.unet_pinn)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    result={'environment':{'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,'initial_memory':memory()},
        'cases':CASES,'normalization_population':'356 fold-training development cases only','normalization':norm,
        'architecture':kwargs,'weights':weights,'memory_attempts':[]}
    selected_hashes={str(dataset.files[i].relative_to(ROOT)):sha(dataset.files[i]) for i in indices}
    for size in dict.fromkeys([cfg.training.batch_size,1]):
        row=memory_attempt(dataset,indices,size,kwargs,norm,weights,cfg,output)
        result['memory_attempts'].append(row)
        if row['success']:
            successful_size=size; break
    else: raise RuntimeError('No tested batch completes the GPU smoke test')
    result['largest_successful_tested_batch_size']=successful_size
    result['gradients']=component_gradients(dataset,indices,successful_size,kwargs,norm,weights)
    result['per_case_initial_losses']=per_case_losses(dataset,indices,kwargs,norm,weights)
    result['checkpoint_safety']=checkpoint_safety(kwargs,cfg,output)
    (output/'gpu_results.json').write_text(json.dumps(result,indent=2))
    result['sensitivity']=[]
    for i in indices:
        print('Interpolation sensitivity',dataset.files[i].parent.name,flush=True)
        row=sensitivity(dataset.files[i]); result['sensitivity'].append(row)
        (output/(dataset.files[i].parent.name+'_sensitivity.json')).write_text(json.dumps(row,indent=2))
        print('SENSITIVITY',json.dumps(row),flush=True)
    result['protected_files_unchanged']=all(sha(ROOT/p)==h for p,h in {**protected_hashes,**selected_hashes}.items())
    assert result['protected_files_unchanged']
    result['protected_file_hashes']=protected_hashes
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',output,flush=True)


if __name__=='__main__': main()
