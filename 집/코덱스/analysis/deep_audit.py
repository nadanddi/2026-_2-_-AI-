"""Train-data repetitions, resolution, alignment and within-farm temporal evidence."""
import json
import itertools
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    cols=list(x.columns[1:20])
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left',validate='one_to_one')
    out={}
    patterns=[]
    for f,g in d.groupby('farm'):
        patterns.append(dict(farm=f,n=len(g),available=[c for c in cols if g[c].notna().any()],
            integer_temp_fraction=float(np.isclose(g.sub_temp.dropna()%1,0).mean()),
            temp_unique=g.sub_temp.nunique(),
            input_complete_rows=int(g[cols].notna().any(axis=1).sum())))
    out['sensor_patterns']=patterns
    # Compare exact matches only when at least three values are observed.
    d['hash']=pd.util.hash_pandas_object(d[cols],index=False)
    valid=d[d[cols].notna().sum(axis=1)>=3]
    pairs=[]
    for a,b in itertools.combinations(sorted(d.farm.unique()),2):
        ga=valid[valid.farm==a]; gb=valid[valid.farm==b]
        z=ga[['hash','time','sub_temp']].merge(gb[['hash','time','sub_temp']],on='hash',suffixes=('_a','_b'))
        if len(z)>=30:
            dt=z.time_b-z.time_a
            mode=int(dt.mode().iloc[0]); aligned=z[dt==mode]
            pairs.append(dict(a=a,b=b,pair_matches=len(z),modal_hour_offset=mode,
                offset_support=len(aligned),same_target_rate=float(np.isclose(aligned.sub_temp_a,aligned.sub_temp_b,equal_nan=False).mean()),
                target_mae=float((aligned.sub_temp_a-aligned.sub_temp_b).abs().mean())))
    out['cross_farm_exact_matches']=sorted(pairs,key=lambda z:z['offset_support'],reverse=True)
    print('Cross-farm matches examined',flush=True)
    # Weather alignment is a diagnostic, not permission to use another farm at inference.
    a=d[d.farm=='F47'].set_index('time'); b=d[d.farm=='F13'].set_index('time')
    alignment=[]
    for shift in range(-7*24,7*24+1,24):
        bb=b.reindex(a.index+shift); bb.index=a.index
        scores={}
        for c in ['out_temp','out_hum','out_rad','out_wspd','in_temp']+TARGETS:
            ok=a[c].notna()&bb[c].notna()
            scores[c]=dict(n=int(ok.sum()),exact=float(np.isclose(a.loc[ok,c],bb.loc[ok,c],atol=1e-8,rtol=0).mean()),
                corr=float(a.loc[ok,c].corr(bb.loc[ok,c])),mae=float((a.loc[ok,c]-bb.loc[ok,c]).abs().mean()))
        alignment.append(dict(shift_hours=shift,scores=scores))
    out['F47_F13_alignment']=alignment
    out['target_dynamics']={}
    out['anomaly_details']={}
    for f in ['F13','F47']:
        g=d[d.farm==f].sort_values('time').set_index('time',drop=False)
        out['target_dynamics'][f]={}
        for t in TARGETS:
            dy=g[t].diff().where(g.time.diff()==1)
            daily=g.groupby('day')[t].agg(['mean','min','max','std'])
            records=[]
            for lo,hi in [(0,60),(60,120),(120,180),(180,250)]:
                s=g.loc[g.day.between(lo,hi-1),t]
                records.append(dict(start=lo,end=hi,n=s.count(),mean=s.mean(),std=s.std(),min=s.min(),max=s.max()))
            changes=daily['mean'].diff().where(daily.index.to_series().diff()==1)
            out['target_dynamics'][f][t]=dict(periods=records,abs_hour_diff_quantiles=dy.abs().quantile([.5,.9,.95,.99,1]).to_dict(),
                top_daily_changes=changes.abs().nlargest(10).to_dict(),
                largest_hour_changes=dy.abs().nlargest(10).to_dict(),
                daily_mean_std=float(daily['mean'].std()),median_intraday_std=float(daily['std'].median()))
        details={}
        for c in cols:
            s=g[c]
            if s.notna().sum()==0: continue
            daily_unique=g.groupby('day')[c].nunique()
            daily_size=g.groupby('day')[c].count()
            details[c]=dict(constant_full_days=int(((daily_unique==1)&(daily_size==24)).sum()),
                mode_values={str(k):int(v) for k,v in s.value_counts().head(5).items()},
                missing_by_day=g.loc[s.isna()].groupby('day').size().to_dict())
        out['anomaly_details'][f]=details
    # Multi-sensor clipping/zeros in other farms.
    out['suspicious_counts']={}
    for c,cond in [('in_co2',d.in_co2.eq(0)),('in_co2_at_5000',d.in_co2.eq(5000)),
                   ('in_hum_zero',d.in_hum.eq(0)),('in_hum_100',d.in_hum.eq(100)),
                   ('in_temp_50',d.in_temp.eq(50)),('all_inputs_missing',d[cols].isna().all(axis=1))]:
        out['suspicious_counts'][c]=d[cond].groupby('farm').size().to_dict()
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(w) for k,w in v.items()}
        if isinstance(v,list):return [clean(w) for w in v]
        if isinstance(v,np.integer):return int(v)
        if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
        return v
    (OUT/'deep_audit.json').write_text(json.dumps(clean(out),ensure_ascii=False,indent=2),encoding='utf8')
    best=max(alignment,key=lambda v:v['scores']['out_temp']['exact'])
    print(json.dumps(clean(dict(sensor_patterns=patterns,top_pairs=out['cross_farm_exact_matches'][:12],best_alignment=best,
        target_dynamics=out['target_dynamics'],suspicious_counts=out['suspicious_counts'])),indent=2))


if __name__=='__main__':main()
