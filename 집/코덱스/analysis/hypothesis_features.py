"""Causal two-stream hypothesis. Official timestamps retained; no label features."""
import numpy as np
import pandas as pd
from initial_features import build as build_base

WEATHER=['out_temp','out_hum','out_rad','out_wspd']
STATE=['in_temp','in_hum','in_co2','act_circfan','act_vent','act_heating']


def build(frame):
    base,groups=build_base(frame)
    extras=[]
    for _,g in frame.groupby('farm'):
        g=g.sort_values('time')
        q=g.set_index('time').reindex(range(int(g.time.min()),int(g.time.max())+1))
        t=pd.Series(q.index,index=q.index);hour=t%24;day=t//24
        f={}
        past=q[WEATHER].shift(24)
        valid=q[WEATHER].notna().all(axis=1)&past.notna().all(axis=1)
        equal=q[WEATHER].eq(past).all(axis=1)&valid
        count=valid.astype(int).groupby(day).cumsum()
        matches=equal.astype(int).groupby(day).cumsum()
        f['hyp_weather_prefix_coverage']=count/(hour+1)
        f['hyp_weather_prefix_match_fraction']=matches/count.replace(0,np.nan)
        f['hyp_weather_prefix_complete_match']=((count==hour+1)&(matches==count)).astype(float)
        # Evidence may change as the current day is observed. It is not a fixed house ID.
        for c in STATE:
            s=q[c]
            f[c+'_hyp_lag48']=s.shift(48)
            f[c+'_hyp_delta48']=s-s.shift(48)
            f[c+'_hyp_start']=s.where(hour.eq(0)).groupby(day).ffill()
            f[c+'_hyp_today_mean']=s.groupby(day).transform(lambda v:v.expanding().mean())
            for k in [1,3,6]:
                # Crossing midnight: previous same-slot day is official day-2.
                # e.g. day d hour 0 -> day d-2 hour 23 (25 elapsed hours).
                f[f'{c}_hyp_bridge{k}']=s.shift(k).where(hour>=k,s.shift(k+24))
        for c in ['in_temp','out_temp']:
            stitched=pd.concat([q[c].shift(k).where(hour>=k,q[c].shift(k+24)) for k in range(6)],axis=1)
            f[c+'_hyp_bridge_mean6']=stitched.mean(axis=1)
            f[c+'_hyp_bridge_coverage6']=stitched.notna().sum(axis=1)/6
        z=pd.DataFrame(f).loc[g.time].astype(float);z.index=g.row_id;extras.append(z)
    extra=pd.concat(extras).loc[base.index]
    result=pd.concat([base,extra],axis=1)
    columns={
        'sub_temp':dict(baseline=groups['sub_temp'],hypothesis=groups['sub_temp']+list(extra.columns)),
        'sub_ec':dict(baseline=groups['raw'],hypothesis=groups['raw']+list(extra.columns))}
    assert result.columns.is_unique
    return result,columns
