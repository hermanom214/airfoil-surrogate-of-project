"""Read-only, checkpoint-local gradient measurements using production losses."""
from __future__ import annotations

import numpy as np
import torch

from src.unet_pinn import unet_pinn_losses
from src.unet_pinn_field_evaluation import select_representative_cases

COMPONENTS = ('data', 'gradient', 'cont', 'mom', 'supervised', 'physics', 'total')
PAIRS = (('data','gradient'), ('data','cont'), ('data','mom'),
         ('gradient','cont'), ('gradient','mom'), ('supervised','cont'),
         ('supervised','mom'), ('supervised','physics'), ('cont','mom'))


def loss_components(prediction, batch, config, normalization, physics_weights):
    """Decompose one production call; diagnostic physics weights come from arm B."""
    loss = unet_pinn_losses(prediction, batch['target'], batch['fluid_mask'],
        batch['physics_mask'], batch['xy'], batch['k'], batch['nut'], batch['nu'],
        batch['u_ref'], batch['l_ref'], normalization,
        supervised_weight=physics_weights['supervised_weight'],
        continuity_weight=physics_weights['continuity_weight'],
        momentum_weight=physics_weights['momentum_weight'], compute_physics=True,
        supervised_config=config, regions=batch['regions'])
    return dict(data=loss['supervised_weighted_data_loss'],
        gradient=config.gradient_loss_weight * loss['supervised_gradient_loss'],
        cont=loss['weighted_continuity'], mom=loss['weighted_momentum'])


def complete_vectors(base):
    vectors = dict(base)
    vectors['supervised'] = vectors['data'] + vectors['gradient']
    vectors['physics'] = vectors['cont'] + vectors['mom']
    vectors['total'] = vectors['supervised'] + vectors['physics']
    return vectors


def flatten_gradients(gradients, parameters):
    if len(gradients) != len(parameters):
        raise ValueError('Gradient and parameter counts differ')
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu().double()
                      for g,p in zip(gradients,parameters)])


def cosine(a, b, tolerance=1e-12):
    a,b = a.reshape(-1).double(), b.reshape(-1).double()
    na,nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Non-finite diagnostic gradient')
    if na <= tolerance or nb <= tolerance:
        return None
    return float(torch.clamp(torch.dot(a,b)/(na*nb),-1,1))


def vector_metrics(base):
    vectors=complete_vectors(base)
    if not all(torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Non-finite diagnostic gradient')
    norms={k:float(torch.linalg.vector_norm(v.double())) for k,v in vectors.items()}
    result={f'norm_{k}':v for k,v in norms.items()}
    for k in ('cont','mom','physics'):
        result[f'ratio_{k}_to_supervised'] = norms[k]/norms['supervised'] if norms['supervised']>1e-12 else None
    for a,b in PAIRS:
        result[f'cos_{a}_{b}']=cosine(vectors[a],vectors[b])
    result['negative_dot_supervised_physics']=-float(torch.dot(vectors['supervised'].reshape(-1).double(),vectors['physics'].reshape(-1).double()))
    return result


def head_slice(gradients, names, parameters, channel):
    parts=[]
    for g,name,p in zip(gradients,names,parameters):
        if name in ('out.weight','out.bias'):
            parts.append((torch.zeros_like(p) if g is None else g)[channel].detach().reshape(-1).cpu().double())
    if len(parts)!=2:
        raise ValueError('Expected direct out.weight and out.bias')
    return torch.cat(parts)


def select_batches(candidates, validation_ids, count=10, batch_size=2):
    eligible=set(validation_ids)
    if len(eligible)!=len(validation_ids):
        raise ValueError('Duplicate validation membership')
    if any(str(c['case_id']) not in eligible for c in candidates):
        raise ValueError('Selection includes cases outside validation')
    selected=select_representative_cases(candidates,count*batch_size)
    return [selected[i:i+batch_size] for i in range(0,len(selected),batch_size)]


def summarize(rows, group_keys):
    result=[]
    groups={tuple(r[k] for k in group_keys) for r in rows}
    for group in sorted(groups):
        subset=[r for r in rows if tuple(r[k] for k in group_keys)==group]
        metric_keys=[k for k in subset[0] if k not in group_keys and
                     (k.startswith(('norm_','ratio_','cos_')) or k=='negative_dot_supervised_physics')]
        for key in metric_keys:
            values=np.asarray([r[key] for r in subset if r[key] is not None],float)
            result.append({**dict(zip(group_keys,group)), 'metric':key,'batch_count':len(subset),
                'defined_batches':len(values),'mean':float(values.mean()) if len(values) else None,
                'median':float(np.median(values)) if len(values) else None,
                'std':float(values.std()) if len(values) else None,
                'min':float(values.min()) if len(values) else None,
                'max':float(values.max()) if len(values) else None,
                'negative_fraction':float((values < 0).mean()) if len(values) else None})
    return result
