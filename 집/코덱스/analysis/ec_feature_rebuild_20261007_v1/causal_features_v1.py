"""Input-only, within-record-day dynamics. No fitted statistics or labels."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd

RAW = ['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2',
       'act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
WINDOWS=(1,2,3,4,6)
def annotate(frame):
    z=frame[['row_id']+RAW].copy()
    parts=z.row_id.str.split('_',expand=True)
    z['farm'],z['day'],z['hour']=parts[0],parts[1].astype(int),parts[2].astype(int)
    assert z.row_id.is_unique
    assert z.hour.between(0,23).all()
    return z.sort_values(['farm','day','hour']).reset_index(drop=True)

def run_state(values, hours):
    on=np.zeros(len(values));off=on.copy();switch=on.copy();age=on.copy();state_changes=on.copy()
    last_event=None;last_value=None;last_on=None;last_h=None;on_count=off_count=0
    for i,(v,h) in enumerate(zip(values,hours)):
        if not np.isfinite(v):
            on[i]=off[i]=switch[i]=age[i]=state_changes[i]=np.nan
            last_value=last_on=last_h=None;on_count=off_count=0;last_event=None
            continue
        active=v>0
        consecutive=last_h is not None and h==last_h+1
        if not consecutive:on_count=off_count=0;last_event=None
        switch[i]=float(consecutive and active!=last_on)
        state_changes[i]=float(consecutive and v!=last_value)
        if state_changes[i]:last_event=h
        on_count=on_count+1 if active else 0
        off_count=off_count+1 if not active else 0
        on[i],off[i]=on_count,off_count
        age[i]=h-last_event if last_event is not None else np.nan
        last_value,last_on,last_h=v,active,h
    return on,off,switch,age,state_changes

def day_features(g):
    h=g.hour.to_numpy();values={c:g[c].to_numpy(float) for c in RAW}
    T,H=values['in_temp'],values['in_hum']
    To,Ho=values['out_temp'],values['out_hum']
    valid=(H>=0)&(H<=100)&(T>-237.3)
    es=.6108*np.exp(17.27*T/(T+237.3));eo=.6108*np.exp(17.27*To/(To+237.3))
    vpd=np.where(valid,es*(1-H/100),np.nan)
    vp_gap=es*H/100-eo*Ho/100
    derived={
        'vpd':vpd,'temp_gap':T-To,'humidity_gap':H-Ho,
        'rad_shade':values['out_rad']*values['act_shade']/100,
        'rad_thermal':values['out_rad']*values['act_thermal']/100,
        'moisture_flux':vp_gap*values['act_vent']/100,
        'fan_vpd':values['act_circfan']*vpd/100,
        'heating_vpd':values['act_heating']*vpd/100,
        'fog_vpd':values['act_fog']*vpd/100,
        'vent_co2':values['act_vent']*values['in_co2']/100,
        'vpd_rad':vpd*values['out_rad'],
    }
    columns={};meta={}
    for name,a in {**values,**derived}.items():
        s=pd.Series(a,index=h)
        columns[name]=a
        meta[name]={'operator':'current','origin':name}
        for lag in WINDOWS:
            la=s.reindex(h-lag).to_numpy()
            columns[f'{name}__lag{lag}']=la
            columns[f'{name}__rate{lag}']=(a-la)/lag
            meta[f'{name}__lag{lag}']={'operator':'lag','origin':name,'hours':lag}
            meta[f'{name}__rate{lag}']={'operator':'rate','origin':name,'hours':lag}
        columns[f'{name}__d2']=a-2*s.reindex(h-1).to_numpy()+s.reindex(h-2).to_numpy()
        meta[f'{name}__d2']={'operator':'second_difference','origin':name}
        for w in WINDOWS:
            means=[];stds=[];sums=[];counts=[]
            for hour in h:
                b=s.reindex(np.arange(max(0,hour-w+1),hour+1)).to_numpy()
                n=np.isfinite(b).sum();counts.append(n)
                means.append(np.nanmean(b) if n else np.nan)
                sums.append(np.nansum(b) if n else np.nan)
                stds.append(np.nanstd(b) if n else np.nan)
            for op,z in [('mean',means),('std',stds),('sum',sums),('count',counts)]:
                key=f'{name}__{op}{w}';columns[key]=z
                meta[key]={'operator':op,'origin':name,'hours':w,'partial_prefix_allowed':True}
    for c in [c for c in RAW if c.startswith('act_')]:
        arrays=run_state(values[c],h)
        for op,a in zip(['on_run','off_run','onoff_switch','since_value_change','value_change'],arrays):
            key=f'{c}__{op}';columns[key]=a;meta[key]={'operator':op,'origin':c}
    mask=(values['act_vent']==0)&(values['act_circfan']==0)
    mask=np.where(np.isfinite(values['act_vent'])&np.isfinite(values['act_circfan']),mask.astype(float),np.nan)
    columns['sealed_run']=run_state(mask,h)[0]
    meta['sealed_run']={'operator':'on_run','origin':'vent==0 and fan==0; daily reset'}
    deltaC=values['in_co2']-pd.Series(values['in_co2'],index=h).reindex(h-1).to_numpy()
    columns['co2_dose_response']=np.where(values['act_co2']>0,deltaC,np.nan)
    columns['co2_uptake_proxy']=np.where((values['act_co2']==0)&(values['act_vent']==0),-deltaC,np.nan)
    meta['co2_dose_response']={'operator':'masked_delta','origin':'CO2 change while dosing; not causal effect'}
    meta['co2_uptake_proxy']={'operator':'masked_delta','origin':'CO2 decrease, dosing and vent zero; uptake unverified'}
    out=pd.DataFrame(columns,index=g.row_id)
    return out,meta

def build(frame):
    z=annotate(frame);parts=[];metadata=None
    for _,g in z.groupby(['farm','day'],sort=True):
        f,m=day_features(g);parts.append(f);metadata=m
    return pd.concat(parts),metadata
