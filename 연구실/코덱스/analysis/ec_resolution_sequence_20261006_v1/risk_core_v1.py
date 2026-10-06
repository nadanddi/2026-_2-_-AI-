"""Pure causal feature and bounded-correction helpers; no fitting or fixed candidate parameters."""
import numpy as np
import pandas as pd

FORBIDDEN={'sub_ec','y','y_day','high','hard_high','missed_high','hard_low','inner_j','row_id','day','hour','farm'}
MEMBERS=['smooth_et','smooth_lgb','smooth_mlp','smooth_pfn']

def design(frame,base_columns):
    if FORBIDDEN.intersection(base_columns):raise ValueError('Labels and split/identity metadata cannot be features')
    if len(base_columns)!=len(set(base_columns)):raise ValueError('Duplicate base columns')
    if not frame.index.is_unique:raise ValueError('Unique index required')
    keys=['farm','day','hour']
    if frame.duplicated(keys).any():raise ValueError('Duplicate time point')
    x=frame[base_columns].astype(float).copy()
    x['farm_F47']=frame.farm.eq('F47').astype(float)
    for name in ['A']+MEMBERS:x[name]=frame[name].to_numpy(float)
    ordered=frame.sort_values(keys)
    prefix=ordered.groupby(['farm','day'],sort=False).A.transform(lambda a:a.expanding().mean())
    x['prefix_A']=prefix.reindex(frame.index)
    for name in ['et','mlp','pfn']:x[name+'_minus_lgb']=frame['smooth_'+name]-frame.smooth_lgb
    x['member_range']=frame[MEMBERS].max(axis=1)-frame[MEMBERS].min(axis=1)
    x['A_minus_lgb']=frame.A-frame.smooth_lgb
    assert not FORBIDDEN.intersection(x.columns)
    return x

def downward(prediction,reference,risk,eligible,*,threshold,cap):
    p,r,s,ok=np.broadcast_arrays(np.asarray(prediction,float),np.asarray(reference,float),np.asarray(risk,float),np.asarray(eligible,bool))
    if not 0<=threshold<1 or cap<0:raise ValueError('Invalid correction parameters')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(s).all():raise ValueError('Nonfinite predictions')
    if (p<0).any() or ((s<0)|(s>1)).any():raise ValueError('Invalid predictions or risk scores')
    amplitude=np.minimum(np.minimum(np.maximum(p-r,0),cap),p)
    amount=ok*amplitude*np.clip((s-threshold)/(1-threshold),0,1)
    return p-amount
