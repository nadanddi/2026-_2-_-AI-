"""Causal EC-control and humidity features; no target-dependent features.

Saturation vapour pressure: FAO-56, chapter 3, Eq. 11 (kPa, Celsius).
https://www.fao.org/4/x0490e/x0490e07.htm
Hourly mean T/RH give an approximate VPD proxy, not measured transpiration.
"""
import numpy as np
import pandas as pd
from causal_features import build_all

CONTROLS=['act_circfan','act_co2','act_fog','act_heating','act_vent','act_shade','act_thermal']


def add_ec_features(frame):
    base,groups=build_all(frame)
    extras=[]
    ec_names=None; humidity_names=None
    for _,g in frame.groupby('farm'):
        g=g.sort_values('time')
        q=g.set_index('time').reindex(range(int(g.time.min()),int(g.time.max())+1))
        t=pd.Series(q.index,index=q.index,dtype=float); day=t//24; hour=t%24
        features={}
        for c in CONTROLS:
            s=q[c]
            features[c+'_day_start']=s.where(hour.eq(0)).groupby(day).ffill()
            features[c+'_day_mean']=s.groupby(day).transform(lambda z:z.expanding().mean())
            features[c+'_day_max']=s.groupby(day).cummax()
            for w in [3,6,24,48,168]:
                features[f'{c}_ec_mean{w}']=s.rolling(w,min_periods=1).mean()
                features[f'{c}_ec_on_fraction{w}']=s.gt(0).where(s.notna()).rolling(w,min_periods=1).mean()
            for lag in [1,3,24,48]:features[f'{c}_ec_lag{lag}']=s.shift(lag)
            features[c+'_hours_since_on']=(t-t.where(s.gt(0)).ffill()).clip(upper=168)
            features[c+'_hours_since_change']=(t-t.where(s.diff().abs().gt(1e-8)).ffill()).clip(upper=168)
        for c in ['in_temp','out_temp','in_hum','out_hum','in_co2']:
            features[c+'_day_start']=q[c].where(hour.eq(0)).groupby(day).ffill()
        ec_names=list(features)
        h={}
        for location in ['in','out']:
            temp=q[location+'_temp']; rh=q[location+'_hum']
            es=.6108*np.exp(17.27*temp/(temp+237.3))
            ea=es*rh/100
            h[location+'_vpd_kpa']=es-ea
            h[location+'_vapour_kpa']=ea
            logratio=np.log(ea/.6108).where(ea>0)
            dew=237.3*logratio/(17.27-logratio)
            h[location+'_dewpoint_c']=dew
            h[location+'_dewpoint_depression']=temp-dew
            for w in [3,6,24,48]:
                h[f'{location}_vpd_mean{w}']=(es-ea).rolling(w,min_periods=1).mean()
        h['vapour_in_out_difference']=h['in_vapour_kpa']-h['out_vapour_kpa']
        h['vpd_x_rad']=h['in_vpd_kpa']*q.out_rad
        h['vpd_x_vent']=h['in_vpd_kpa']*q.act_vent/100
        humidity_names=list(h)
        features.update(h)
        result=pd.DataFrame(features).loc[g.time]
        result.index=g.row_id
        extras.append(result.astype(float))
    groups={**groups,'ec_controls':ec_names,'humidity':humidity_names}
    return pd.concat([base,pd.concat(extras).loc[base.index]],axis=1),groups
