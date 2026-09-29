# -*- coding: utf-8 -*-
"""EC 독립 후보 공통 피처. 원본 CSV 경로 또는 DataFrame; 정답은 읽지 않는다."""
from pathlib import Path
import numpy as np
import pandas as pd
SENSORS=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=SENSORS[3:]
FEATURE_COLUMNS=['day','hour','farm_id','hr_sin','hr_cos','midnight']
for c in SENSORS:
    FEATURE_COLUMNS += [c,c+'_h0',c+'_prefix',c+'_dev0']
    if c in ACTS:FEATURE_COLUMNS += [c+'_zero']
FEATURE_COLUMNS += ['closed_prefix','prefix_observations']
SPLINE_COLUMNS=['day','hour','in_temp','in_hum','in_co2']+ACTS+[c+'_prefix' for c in SENSORS]+[c+'_h0' for c in SENSORS[:3]]

def read_x(value):
    x=pd.read_csv(value) if isinstance(value,(str,Path)) else value.copy(deep=True)
    x=x[['row_id']+SENSORS].copy()
    x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int);x['hour']=x.row_id.str[8:10].astype(int);x['t']=24*x.day+x.hour
    return x

def build_features(train_X_csv,test_X_csv,train_y_csv=None):
    a=pd.concat([read_x(train_X_csv),read_x(test_X_csv)],ignore_index=True)
    a=a[a.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
    if not a.row_id.is_unique:raise ValueError('중복 row_id')
    z={c:a[c] for c in ['row_id','farm','day','hour','t']}
    z.update(farm_id=(a.farm=='F47').astype(float),hr_sin=np.sin(2*np.pi*a.hour/24),hr_cos=np.cos(2*np.pi*a.hour/24),midnight=(a.hour==0).astype(float))
    for c in SENSORS:
        v=a[c];g=v.groupby([a.farm,a.day])
        z[c]=v;z[c+'_h0']=v.where(a.hour==0).groupby([a.farm,a.day]).ffill()
        z[c+'_prefix']=g.transform(lambda s:s.expanding().mean())
        z[c+'_dev0']=v-z[c+'_h0']
        if c in ACTS:z[c+'_zero']=(v==0).astype(float).where(v.notna()).groupby([a.farm,a.day]).transform(lambda s:s.expanding().mean())
    z['closed_prefix']=((z['act_circfan_prefix']<10)&(z['act_vent_zero']>.85)).astype(float)
    z['prefix_observations']=a.groupby(['farm','day']).cumcount()+1
    return pd.DataFrame(z)
