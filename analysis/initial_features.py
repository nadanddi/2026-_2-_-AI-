"""Compact, input-only features for initial model v1. Hourly grid, no imputation."""
import numpy as np
import pandas as pd
from causal_features import RAW


def build(frame):
    outputs=[]
    for farm,g in frame.groupby('farm'):
        g=g.sort_values('time')
        assert g.time.is_unique and g.row_id.is_unique
        q=g.set_index('time').reindex(range(int(g.time.min()),int(g.time.max())+1))
        t=pd.Series(q.index,index=q.index); hour=t%24; day=t//24
        f={c:q[c] for c in RAW}
        f.update(day=day,hour_sin=np.sin(2*np.pi*hour/24),hour_cos=np.cos(2*np.pi*hour/24),midnight=(hour==0).astype(float))
        for c in ['in_temp','out_temp','in_hum','in_co2']:
            for lag in [1,3,6]: f[f'{c}_lag{lag}']=q[c].shift(lag)
            for w in [3,6]: f[f'{c}_mean{w}']=q[c].rolling(w,min_periods=1).mean()
            f[c+'_delta3']=q[c]-q[c].shift(3)
        for w in [3,6,12]:
            f[f'rad_energy{w}']=q.out_rad.rolling(w,min_periods=w).sum()*.0036
        for c in ['in_temp','in_hum','in_co2']:
            f[c+'_missing']=q[c].isna().astype(float)
            f[c+'_age']=t-t.where(q[c].notna()).ffill()
            f[c+'_coverage6']=q[c].rolling(6,min_periods=1).count()/6
        f['temp_difference']=q.in_temp-q.out_temp
        f['rad_open']=q.out_rad*q.act_shade/100
        f['heating_closed']=q.act_heating*(1-q.act_thermal/100)
        f['vent_temp']=q.act_vent/100*f['temp_difference']
        f['vent_wind']=q.act_vent/100*q.out_wspd
        for c in ['in_temp','out_temp','act_heating']:
            f[c+'_today_mean']=q[c].groupby(day).transform(lambda s:s.expanding().mean())
        temp_columns=list(f)
        es=lambda v:.6108*np.exp(17.27*v/(v+237.3))
        f['vpd']=es(q.in_temp)*(1-q.in_hum/100)
        f['vapour_difference']=es(q.in_temp)*q.in_hum/100-es(q.out_temp)*q.out_hum/100
        for w in [3,6]: f[f'vpd_mean{w}']=f['vpd'].rolling(w,min_periods=1).mean()
        for c in ['act_circfan','act_fog','act_co2','act_vent']:
            f[c+'_mean6']=q[c].rolling(6,min_periods=1).mean()
            f[c+'_delta1']=q[c].diff()
        f['fog_vpd']=q.act_fog/100*f['vpd']
        f['vent_vapour']=q.act_vent/100*f['vapour_difference']
        f['co2_rad']=q.in_co2*q.out_rad
        f['fan_vpd']=q.act_circfan/100*f['vpd']
        ec_columns=[c for c in f if c not in ['vent_wind','heating_closed','act_heating_today_mean'] and not c.startswith('out_temp_')]
        z=pd.DataFrame(f).loc[g.time].astype(float)
        z.index=g.row_id
        outputs.append(z)
    return pd.concat(outputs),{'raw':RAW+['day','hour_sin','hour_cos','midnight'],'sub_temp':temp_columns,'sub_ec':ec_columns}
