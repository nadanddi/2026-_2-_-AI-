"""Input-only features. Reindex in elapsed hours; never bfill or center windows."""
import numpy as np
import pandas as pd

RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2',
     'act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
HISTORY=['in_temp','out_temp','out_rad','in_hum','act_heating','act_vent','act_shade','act_thermal']


def build_farm(g):
    """One farm only, no target access. Return original row IDs and feature groups."""
    assert g.farm.nunique()==1
    assert not g.time.duplicated().any()
    g=g.sort_values('time')
    grid=g.set_index('time').reindex(range(int(g.time.min()),int(g.time.max())+1))
    time=pd.Series(grid.index,index=grid.index,dtype=float)
    raw=grid[RAW].copy()
    clock=pd.DataFrame(index=grid.index)
    clock['day']=time//24
    for k in [1,2,3]:
        clock[f'hour_sin{k}']=np.sin(2*np.pi*k*(time%24)/24)
        clock[f'hour_cos{k}']=np.cos(2*np.pi*k*(time%24)/24)
    lagged={}
    for c in HISTORY:
        for lag in [1,2,3,6,12,24,48]:
            lagged[f'{c}_lag{lag}']=grid[c].shift(lag)
        lagged[f'{c}_delta1']=grid[c]-grid[c].shift(1)
        lagged[f'{c}_delta24']=grid[c]-grid[c].shift(24)
    smooth={}
    for c in HISTORY:
        for w in [3,6,12,24,48,168]:
            window=grid[c].rolling(w,min_periods=1)
            smooth[f'{c}_mean{w}']=window.mean()
            smooth[f'{c}_coverage{w}']=window.count()/w
        smooth[f'{c}_std24']=grid[c].rolling(24,min_periods=2).std()
        last_time=time.where(grid[c].notna()).ffill()
        smooth[f'{c}_age']=time-last_time
    physics=pd.DataFrame(index=grid.index)
    physics['inside_outside_temp']=grid.in_temp-grid.out_temp
    physics['inside_outside_hum']=grid.in_hum-grid.out_hum
    physics['rad_unshaded']=grid.out_rad*grid.act_shade/100
    physics['vent_temp_exchange']=(grid.in_temp-grid.out_temp)*grid.act_vent/100
    physics['vent_wind']=grid.act_vent*grid.out_wspd/100
    physics['heating_curtain']=grid.act_heating*(1-grid.act_thermal/100)
    physics['heating_cold']=grid.act_heating*(20-grid.out_temp).clip(lower=0)/100
    physics['temp_hum']=grid.in_temp*grid.in_hum/100
    physics['light_hours24']=grid.out_rad.gt(10).where(grid.out_rad.notna()).rolling(24,min_periods=1).sum()
    for c in ['act_heating','act_vent','act_co2','act_fog']:
        on=grid[c].gt(0).where(grid[c].notna())
        physics[c+'_on']=on
        physics[c+'_on_hours24']=on.rolling(24,min_periods=1).sum()
    trend=pd.DataFrame(index=grid.index)
    for knot in range(30,241,30):
        trend[f'day_hinge{knot}']=(clock.day-knot).clip(lower=0)
    day_context={}
    day_key=time//24
    for c in HISTORY:
        day_context[c+'_today_mean']=grid[c].groupby(day_key).transform(lambda s:s.expanding().mean())
        day_context[c+'_today_min']=grid[c].groupby(day_key).cummin()
        day_context[c+'_today_max']=grid[c].groupby(day_key).cummax()
        day_context[c+'_today_count']=grid[c].notna().astype(int).groupby(day_key).cumsum()
        for lag in [1,2,3,6]:
            day_context[f'{c}_today_lag{lag}']=grid[c].shift(lag).where((time%24)>=lag)
    quality={}
    for c in RAW:
        quality[c+'_missing']=grid[c].isna().astype(float)
        quality[c+'_repeat24']=grid[c].eq(grid[c].shift(24)).where(grid[c].notna()&grid[c].shift(24).notna()).astype(float)
    parts=[raw,clock,pd.DataFrame(lagged),pd.DataFrame(smooth),physics,trend,pd.DataFrame(day_context),pd.DataFrame(quality)]
    names=['raw','clock','lags','smooth','physics','trend','day_context','quality']
    groups={n:list(p.columns) for n,p in zip(names,parts)}
    result=pd.concat(parts,axis=1).loc[g.time]
    result.index=g.row_id
    return result.astype(float),groups


def build_all(frame):
    frames=[]
    groups=None
    for _,g in frame.groupby('farm'):
        features,groups=build_farm(g)
        frames.append(features)
    return pd.concat(frames),groups
