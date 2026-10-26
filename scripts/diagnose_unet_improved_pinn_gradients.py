"""Measure local loss-gradient interactions without changing trained models."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ml_models import build_model
from src.ml_supervised import supervised_config_from_checkpoint
from src.unet_pinn import PinnFlowDataset
from src.unet_pinn_ab import (configure_determinism, cuda_peak_memory,
    dataset_artifact_sha256, membership_sha256, state_dict_sha256)
from src.unet_pinn_field_evaluation import (assert_matching_normalization, case_metadata,
    file_sha256, region_masks)
from src.unet_pinn_gradient_diagnostics import (COMPONENTS, PAIRS, complete_vectors,
    flatten_gradients, head_slice, loss_components, select_batches, summarize, vector_metrics)
from src.unet_pinn_reporting import validate_pair, validate_run

OUTPUT=ROOT/'outputs/diagnostics/unet_improved_pinn_gradients'
PROFILES={'A':'unet_improved_supervised','B':'unet_improved_pinn'}
CHANNELS=('p','Ux','Uy')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def measure(model,batch,config,normalization,weights,regions):
    names,parameters=zip(*model.named_parameters())
    prediction=model(batch['input'])
    losses=loss_components(prediction,batch,config,normalization,weights)
    parameter_base={}; heads={c:{} for c in CHANNELS}; output_base={}
    for index,(component,loss) in enumerate(losses.items()):
        gradients=torch.autograd.grad(loss,(*parameters,prediction),
            retain_graph=index<len(losses)-1,allow_unused=True)
        parameter_base[component]=flatten_gradients(gradients[:-1],parameters)
        for channel,c in enumerate(CHANNELS):
            heads[c][component]=head_slice(gradients[:-1],names,parameters,channel)
        output_base[component]=gradients[-1].detach().cpu().double()
    global_metrics=vector_metrics(parameter_base)
    head_metrics={c:vector_metrics(base) for c,base in heads.items()}
    regional=[]; output_stats=[]
    for region,mask in regions.items():
        if mask.shape != (prediction.shape[0],prediction.shape[2],prediction.shape[3]):
            raise ValueError('Regional output mask shape mismatch')
        mask=torch.as_tensor(mask,dtype=torch.bool)
        for channel,c in enumerate(CHANNELS):
            base={k:v[:,channel][mask] for k,v in output_base.items()}
            regional.append(dict(region=region,channel=c,valid_cells=int(mask.sum()),**vector_metrics(base)))
            for component,values in complete_vectors(base).items():
                output_stats.append(dict(region=region,channel=c,component=component,
                    valid_cells=int(values.numel()),norm=float(values.norm()),
                    mean_signed=float(values.mean()),p95_absolute=float(torch.quantile(values.abs(),.95))))
    return global_metrics,head_metrics,regional,output_stats,{k:float(v.detach()) for k,v in losses.items()}


def plots(rows,heads,regional,path):
    colors={'A':'#2471A3','B':'#C25125'}
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for model in ('A','B'):
        data=[r for r in rows if r['model']==model]
        means=[np.mean([r['norm_'+c] for r in data]) for c in COMPONENTS]
        ax.plot(COMPONENTS,means,'o-',color=colors[model],label=model)
    ax.set(yscale='log',ylabel='Mean parameter-gradient L2 norm',title='Weighted component gradients (10 validation batches)')
    ax.legend(); ax.grid(alpha=.2); fig.savefig(path/'gradient_norms.png',dpi=190); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for model,ax in zip(('A','B'),axes):
        matrix=np.full((len(COMPONENTS)-1,len(COMPONENTS)-1),np.nan)
        labels=COMPONENTS[:-1]
        for a,b in PAIRS:
            value=np.mean([r[f'cos_{a}_{b}'] for r in rows if r['model']==model and r[f'cos_{a}_{b}'] is not None])
            i,j=labels.index(a),labels.index(b); matrix[i,j]=matrix[j,i]=value
        np.fill_diagonal(matrix,1)
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r'); ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,title=model)
        ax.tick_params(axis='x',rotation=45)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i,j]): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes,label='Mean batch cosine'); fig.savefig(path/'cosine_similarity_heatmap.png',dpi=190); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    keys=('cos_supervised_cont','cos_supervised_mom','cos_supervised_physics')
    for ax,c in zip(axes,CHANNELS):
        for model,offset in (('A',-.18),('B',.18)):
            values=[]
            for key in keys:
                sample=[r[key] for r in heads if r['model']==model and r['channel']==c and r[key] is not None]
                values.append(np.mean(sample) if sample else np.nan)
            ax.bar(np.arange(3)+offset,values,width=.35,color=colors[model],label=model)
        ax.set(title=c,ylim=(-1,1),xticks=range(3),xticklabels=('continuity','momentum','physics'),ylabel='Head cosine with supervised'); ax.axhline(0,color='black',lw=.7)
        if c=='p':
            ax.text(0,.025,'undefined\nzero gradient',ha='center',va='bottom',fontsize=8)
    axes[0].legend(); fig.savefig(path/'channel_gradient_cosines.png',dpi=190); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,c in zip(axes,CHANNELS):
        for model,offset in (('A',-.18),('B',.18)):
            means=[]
            for region in ('full_domain','near_airfoil','wake'):
                sample=[r['cos_supervised_physics'] for r in regional if r['model']==model and r['channel']==c and r['region']==region and r['cos_supervised_physics'] is not None]
                means.append(np.mean(sample) if sample else np.nan)
            ax.bar(np.arange(3)+offset,means,width=.35,color=colors[model],label=model)
        ax.set(title=c,ylim=(-1,1),xticks=range(3),xticklabels=('full','near airfoil','wake'),ylabel='Output cosine: supervised vs physics'); ax.axhline(0,color='black',lw=.7)
    axes[0].legend(); fig.savefig(path/'regional_gradient_conflicts.png',dpi=190); plt.close(fig)


def make_report(metadata,global_summary,head_summary,regional_summary):
    def find(summary,model,metric,**keys):
        return next(r for r in summary if r['model']==model and r['metric']==metric and all(r[k]==v for k,v in keys.items()))
    lines=['# Phase A v1 gradient-interaction diagnostic','',
        'Measured 10 deterministic validation batches of two cases at each selected epoch-28 checkpoint. All components use the same production forward pass and the same diagnostic weights at A and B: data + 0.1 gradient matching + 0.1 continuity + 0.1 momentum.','',
        '## Global parameter gradients','',
        '| Model | Mean supervised norm | Mean physics norm | Mean physics/supervised ratio | Mean cosine | Median cosine | Negative batches |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in ('A','B'):
        values=[find(global_summary,model,k) for k in ('norm_supervised','norm_physics','ratio_physics_to_supervised','cos_supervised_physics')]
        a,b,c,d=values
        lines.append(f"| {model} | {a['mean']:.6g} | {b['mean']:.6g} | {c['mean']:.4g} | {d['mean']:.4f} | {d['median']:.4f} | {d['negative_fraction']:.0%} |")
    lines+=['','All pairwise means, medians, standard deviations, minima, maxima, undefined counts and negative fractions are in the CSV tables. Metrics are calculated per batch before summarization.','',
        '## Direct output-head gradients','',
        'The final Conv2d is `out`: weights [3,32,1,1], bias [3], channels [p*, ux*, uy*]. Each head slice includes its weights and bias.','',
        '| Model | Channel | Supervised vs continuity | Supervised vs momentum | Supervised vs physics | Negative physics fraction |',
        '|---|---|---:|---:|---:|---:|']
    for model in ('A','B'):
        for c in CHANNELS:
            a,b,d=[find(head_summary,model,k,channel=c) for k in ('cos_supervised_cont','cos_supervised_mom','cos_supervised_physics')]
            fmt=lambda v:'undefined (zero gradient)' if v is None else f'{v:.4f}'
            lines.append(f"| {model} | {c} | {fmt(a['mean'])} | {fmt(b['mean'])} | {fmt(d['mean'])} | {d['negative_fraction']:.0%} |")
    lines+=['','Continuity depends only on velocity, so its direct pressure-output and pressure-head gradient is zero; that cosine is undefined. Earlier shared layers can still transmit continuity-driven changes into pressure predictions.','',
        '## Output-tensor regional gradients','',
        'These derivatives are with respect to nondimensional network output values, not spatial gradients such as dp/dx. The loss remains the original globally reduced production loss; masks restrict which output coordinates are compared, without redefining the loss. Near-airfoil and wake use the existing evaluation ROIs intersected with fluid and interpolation validity.','',
        '| Model | Channel | Full cosine | Near-airfoil cosine | Wake cosine |',
        '|---|---|---:|---:|---:|']
    for model in ('A','B'):
        for c in CHANNELS:
            values=[find(regional_summary,model,'cos_supervised_physics',channel=c,region=region)['mean'] for region in ('full_domain','near_airfoil','wake')]
            lines.append(f"| {model} | {c} | "+' | '.join(f'{v:.4f}' if v is not None else 'undefined' for v in values)+' |')
    lines+=['','## First-order directional interpretation','',
        'For a hypothetical infinitesimal Euclidean gradient step along physics, the supervised-loss change per unit step is `-dot(g_supervised,g_physics)`. Positive values predict a local increase in the supervised objective. This direction is computed but never applied; it is not an Adam update or a training trajectory.','']
    for model in ('A','B'):
        s=find(global_summary,model,'negative_dot_supervised_physics')
        lines.append(f"- {model}: mean derivative {s['mean']:.6g}, median {s['median']:.6g}, range [{s['min']:.6g}, {s['max']:.6g}].")
    def stat(summary,model,metric,key='mean',**filters):
        return find(summary,model,metric,**filters)[key]
    observed=any(stat(global_summary,m,'cos_supervised_physics','negative_fraction')>0 for m in ('A','B'))
    verdict=('Some full-objective local gradient opposition was observed.' if observed else
        'No full-objective gradient opposition was observed in this validation sample. The hypothesis that physics generally opposes the complete improved supervised objective at these selected checkpoints is not supported.')
    lines+=['','## Interpretation of the measured checkpoint states','',verdict+' Component-level and head-level opposition must be assessed separately from the complete objective.', '']
    for model in ('A','B'):
        lines.append(f"- {model}: global data/continuity mean cosine {stat(global_summary,model,'cos_data_cont'):.4f}, negative fraction {stat(global_summary,model,'cos_data_cont','negative_fraction'):.0%}; data/momentum mean {stat(global_summary,model,'cos_data_mom'):.4f}, negative fraction {stat(global_summary,model,'cos_data_mom','negative_fraction'):.0%}. Weighted gradient-matching norm averages {stat(global_summary,model,'norm_gradient'):.5g} versus data {stat(global_summary,model,'norm_data'):.5g} (ratio of mean norms {stat(global_summary,model,'norm_gradient')/stat(global_summary,model,'norm_data'):.2f}).")
    lines+=['','The weighted gradient-matching component changes the combined supervised direction; local data-fit opposition can be hidden by the full-objective cosine. These norm comparisons describe parameter-space magnitudes, not physical channel importance.','']
    for model in ('A','B'):
        lines.append(f"- {model} pressure head: data/momentum mean cosine {stat(head_summary,model,'cos_data_mom',channel='p'):+.4f}, negative fraction {stat(head_summary,model,'cos_data_mom','negative_fraction',channel='p'):.0%}; gradient-matching/momentum mean {stat(head_summary,model,'cos_gradient_mom',channel='p'):+.4f}, negative fraction {stat(head_summary,model,'cos_gradient_mom','negative_fraction',channel='p'):.0%}; full supervised/momentum mean {stat(head_summary,model,'cos_supervised_mom',channel='p'):+.4f}, negative fraction {stat(head_summary,model,'cos_supervised_mom','negative_fraction',channel='p'):.0%}.")
    lines+=['','The pressure data-fit and gradient-matching components can therefore have different relationships to momentum. Direct pressure-head competition is distinct from persistent opposition to the complete supervised gradient.','']
    for c in CHANNELS:
        lines.append(f"- B {c} head: physics/supervised norm ratio {stat(head_summary,'B','ratio_physics_to_supervised',channel=c):.3f}; negative full supervised/physics fraction {stat(head_summary,'B','cos_supervised_physics','negative_fraction',channel=c):.0%}. B wake output-coordinate mean cosine {stat(regional_summary,'B','cos_supervised_physics',channel=c,region='wake'):+.4f}, negative fraction {stat(regional_summary,'B','cos_supervised_physics','negative_fraction',channel=c,region='wake'):.0%}.")
    lines+=['','The regional tables show where direct output-coordinate opposition occurs. Parameter-head and output-coordinate gradients can differ because the output-to-parameter Jacobian projects spatial directions into a smaller parameter subspace. Head comparisons do not isolate effects mediated by shared hidden features.', '',
        'The results are consistent with a changed local landscape and selective competition within the compound objective. They do not establish that momentum caused pressure smoothing or that continuity caused Ux improvement. In particular, positive pressure gradient-matching/momentum alignment can coexist with the earlier observed suppressed extrema: matching reference gradients penalizes both excessive and insufficient gradients, and this diagnostic does not determine whether an actual update would smooth a particular structure. Earlier shared-layer competition, Adam dynamics, finite-step effects, masks, reference turbulence and approximation error remain possible contributors.', '',
        '## Artifacts','',
        '- `tables/per_batch_gradient_metrics.csv` and `gradient_similarity_summary.csv`: all global norms, ratios, cosines and first-order derivatives.',
        '- `tables/per_batch_head_gradients.csv` and `channel_gradient_summary.csv`: channel slices including weight and bias.',
        '- `tables/per_batch_regional_output_gradients.csv` and `regional_output_gradient_summary.csv`: masked output-coordinate comparisons.',
        '- `tables/output_gradient_statistics.csv`: per-batch/component/channel/region norms, signed means and P95 magnitudes.',
        '- `tables/weighted_loss_components.csv`: measured weighted production component losses.',
        '- `figures/`: gradient norms, pairwise cosine heatmaps, head cosines and regional output cosines.', '',
        '## Limitations','',
        'These are checkpoint-local gradients on 20 metadata-selected validation cases, not measurements along 30 training epochs. A and B occupy different parameter states; their comparison is not a causal intervention. Direct output-head slices do not isolate the shared network. Output coordinates have different nondimensional physical scaling, so cross-channel norm comparisons do not measure physical importance. Residuals use reference CFD nut and the production Cartesian finite differences. Negative cosines establish local first-order opposition, not worse physical accuracy or a causal explanation of pressure smoothing. Finite steps, Adam preconditioning, and stochastic training can behave differently.','',
        '## Reproducibility and safety','',
        f"Device: {metadata['device']}. Seed: 42. Dataset and all protected run/split/prior-diagnostic artifact hashes matched before and after. Model state hashes matched before and after; all parameter .grad buffers remained None. No optimizer was constructed, no parameter update was applied, and no commit was created.",
        f"Elapsed diagnostic computation and provenance checks: {metadata['elapsed_seconds']:.2f} s (figure/report rendering excluded). Peak allocated/reserved CUDA memory: {metadata['memory']['peak_cuda_memory_allocated_mib']} / {metadata['memory']['peak_cuda_memory_reserved_mib']} MiB."]
    return '\n'.join(lines)+'\n'


def write_report(output,metadata,global_summary,head_summary,regional_summary):
    """Render the report from already measured batch summaries, without inference."""
    (output/'report.md').write_text(make_report(metadata,global_summary,head_summary,regional_summary),encoding='utf-8')


def main():
    if OUTPUT.exists(): raise FileExistsError(f'Refusing to overwrite {OUTPUT}')
    started=time.perf_counter(); configure_determinism(42)
    runs={name:validate_run(ROOT/'data/models'/profile,profile) for name,profile in PROFILES.items()}
    validate_pair(runs['A'][0],runs['B'][0])
    checkpoints={name:run[3] for name,run in runs.items()}
    norm=assert_matching_normalization(checkpoints['A'],checkpoints['B'])
    config=supervised_config_from_checkpoint(checkpoints['B']); weights=runs['B'][0]['loss_weights']
    protected=[]
    for profile in PROFILES.values(): protected.extend(p for p in (ROOT/'data/models'/profile).rglob('*') if p.is_file())
    protected.append(ROOT/'data/models/unet_pinn/fixed_test_split.json')
    prior=ROOT/'outputs/diagnostics/unet_improved_pinn_v1'
    if prior.exists(): protected.extend(p for p in prior.rglob('*') if p.is_file())
    before={str(p.relative_to(ROOT)):file_sha256(p) for p in protected}
    dataset=PinnFlowDataset(ROOT/'data/pinn_dataset',supervised_config=config,return_regions=True)
    if len(dataset)!=558: raise ValueError('Unexpected dataset count')
    for key,value in norm.items(): setattr(dataset,key,value)
    fingerprint=dataset_artifact_sha256(dataset.files,dataset.root)
    if any(run[0]['dataset_fingerprint']!=fingerprint for run in runs.values()): raise ValueError('Dataset fingerprint mismatch')
    split=json.loads((ROOT/'data/models/unet_pinn/fixed_test_split.json').read_text())
    fixed=split['test_case_ids']
    if set(fixed)!=set(runs['A'][0]['fixed_test_case_ids']): raise ValueError('Persisted fixed-test membership mismatch')
    val_ids=runs['A'][0]['validation_case_ids']; lookup={p.parent.name:i for i,p in enumerate(dataset.files)}
    batches=select_batches([case_metadata(dataset.files[lookup[cid]]) for cid in val_ids],val_ids)
    selected_ids=[c['case_id'] for batch in batches for c in batch]
    if set(selected_ids)&set(fixed): raise ValueError('Fixed test leaked into gradient sample')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    all_rows=[]; head_rows=[]; regional_rows=[]; output_stats=[]; loss_rows=[]; state_hashes={}
    normalization=tuple(norm[k] for k in ('p_mean','p_std','ux_mean','ux_std','uy_mean','uy_std'))
    for name,checkpoint in checkpoints.items():
        model=build_model('unet_pinn',**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
        if tuple(model.out.weight.shape)!=(3,32,1,1) or tuple(model.out.bias.shape)!=(3,): raise ValueError('Unsupported head')
        initial=state_dict_sha256(model.state_dict())
        for batch_id,cases in enumerate(batches,1):
            samples=[dataset[lookup[c['case_id']]] for c in cases]
            batch=default_collate(samples)
            masks={r:[] for r in ('full_domain','near_airfoil','wake')}
            for case,sample in zip(cases,samples):
                with np.load(dataset.files[lookup[case['case_id']]],allow_pickle=False) as z:
                    valid=(z['fluid_mask']>.5)&(z['interpolation_valid_mask']>.5)
                rs=region_masks(sample['xy'].numpy(),valid,float(case['chord']),float(case['aoa_deg']))
                for r in masks: masks[r].append(rs[r])
            masks={r:np.stack(v) for r,v in masks.items()}
            batch={k:v.to(device) if torch.is_tensor(v) and k!='omega' else v for k,v in batch.items()}
            metrics,heads,regional,stats,losses=measure(model,batch,config,normalization,weights,masks)
            identity=dict(model=name,batch=batch_id,case_ids=';'.join(c['case_id'] for c in cases))
            all_rows.append({**identity,**metrics}); head_rows.extend({**identity,'channel':c,**v} for c,v in heads.items())
            regional_rows.extend({**identity,**r} for r in regional); output_stats.extend({**identity,**r} for r in stats)
            loss_rows.append({**identity,**losses})
            if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Parameter gradient buffer changed')
            print(f'{name}: batch {batch_id}/10 completed',flush=True)
        final=state_dict_sha256(model.state_dict())
        if final!=initial: raise RuntimeError('Model state mutated')
        state_hashes[name]={'before':initial,'after':final}; del model
    memory=cuda_peak_memory(device)
    after={str(p.relative_to(ROOT)):file_sha256(p) for p in protected}
    fingerprint_after=dataset_artifact_sha256(dataset.files,dataset.root)
    if before!=after or fingerprint_after!=fingerprint: raise RuntimeError('Protected artifact changed')
    metadata=dict(device=str(device),seed=42,head={'weight_shape':[3,32,1,1],'bias_shape':[3],'channels':list(CHANNELS)},
        elapsed_seconds=time.perf_counter()-started,memory=memory,normalization=norm,physics_weights=weights,
        gradient_matching_weight=config.gradient_loss_weight,dataset_fingerprint=fingerprint,dataset_fingerprint_after=fingerprint_after,
        split_counts={'train':356,'validation':90,'fixed_test':112},validation_membership_hash=membership_sha256(val_ids),
        selected_batches=[[{k:c[k] for k in ('case_id','naca_code','aoa_deg','U_inf','chord')} for c in batch] for batch in batches],
        selection_rule='Existing metadata-only categorical plus normalized maximin coverage over camber, thickness, AoA, U_inf; sorted deterministic tie-breaks; sequential pairs; 20 validation cases',
        execution_batch_size=2,checkpoint_epochs={k:r[0]['best_validation_epoch'] for k,r in runs.items()},
        parameter_state_hashes=state_hashes,protected_hashes_before=before,protected_hashes_after=after,
        pressure_unit='m^2/s^2 kinematic pressure; no density conversion',cosine_zero_norm_tolerance=1e-12)
    global_summary=summarize(all_rows,['model']); head_summary=summarize(head_rows,['model','channel'])
    regional_summary=summarize(regional_rows,['model','region','channel'])
    (OUTPUT/'tables').mkdir(parents=True); (OUTPUT/'figures').mkdir()
    for filename,rows in [('per_batch_gradient_metrics.csv',all_rows),('gradient_similarity_summary.csv',global_summary),
        ('per_batch_head_gradients.csv',head_rows),('channel_gradient_summary.csv',head_summary),
        ('per_batch_regional_output_gradients.csv',regional_rows),('regional_output_gradient_summary.csv',regional_summary),
        ('output_gradient_statistics.csv',output_stats),('weighted_loss_components.csv',loss_rows)]: write_csv(OUTPUT/'tables'/filename,rows)
    plots(all_rows,head_rows,regional_rows,OUTPUT/'figures')
    (OUTPUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    write_report(OUTPUT,metadata,global_summary,head_summary,regional_summary)
    print(f'Completed: {OUTPUT}',flush=True)


if __name__=='__main__': main()
