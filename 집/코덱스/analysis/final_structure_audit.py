"""Three descriptive checks; elapsed-time alignment, training data only."""
import json
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS


def main():
    x,y=load('train_X.csv'),load('train_y.csv')
    features=[c for c in x if c not in ['row_id','farm','day','hour','time']]
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left',validate='one_to_one')
    quality=[]; missing_runs=[]; constant_runs=[]; periods=[]; repeats=[]
    for farm,g in d.groupby('farm'):
        g=g.sort_values('time').set_index('time',drop=False)
        for c in features:
            s=g[c]; observed=s.notna()
            quality.append(dict(farm=farm,column=c,rows=len(s),missing=int(s.isna().sum()),all_missing=not observed.any(),unique=int(s.nunique())))
            if not observed.any():continue
            for kind,mask in [('missing',s.isna()),('constant',observed)]:
                breaks=g.time.diff().ne(1) | mask.ne(mask.shift())
                if kind=='constant': breaks=breaks | s.ne(s.shift())
                ids=breaks.cumsum()
                if kind=='constant':mask=mask & ids.map(ids[mask].value_counts()).ge(24)
                for _,z in g[mask].groupby(ids[mask]):
                    if kind=='constant' and len(z)<24:continue
                    row=dict(farm=farm,column=c,start_day=int(z.day.iloc[0]),start_hour=int(z.hour.iloc[0]),hours=len(z))
                    if kind=='constant':row['value']=float(z[c].iloc[0]);constant_runs.append(row)
                    else:missing_runs.append(row)
        for period,z in g.groupby((g.day//30)*30):
            for c in TARGETS:
                s=z[c].dropna()
                if len(s)<2:continue
                jump=(g[c]-g[c].reindex(g.index-1).set_axis(g.index)).abs().loc[z.index]
                threshold=.2 if c=='sub_ec' else 3
                row=dict(farm=farm,period=int(period),target=c,n=len(s),jump_n=int(jump.gt(threshold).sum()),midnight_jump_n=int((jump.gt(threshold)&z.hour.eq(0)).sum()),midnight_boundaries=int((jump.notna()&z.hour.eq(0)).sum()))
                for lag in [24,48]:
                    old=g[c].reindex(z.index-lag).set_axis(z.index)
                    valid=z[c].notna()&old.notna()
                    row[f'lag{lag}_n']=int(valid.sum())
                    row[f'lag{lag}_mae']=float((z.loc[valid,c]-old[valid]).abs().mean())
                old24=g[c].reindex(z.index-24).set_axis(z.index)
                old48=g[c].reindex(z.index-48).set_axis(z.index)
                common=z[c].notna()&old24.notna()&old48.notna()
                row['common_lag_n']=int(common.sum())
                row['common_lag24_mae']=float((z.loc[common,c]-old24[common]).abs().mean())
                row['common_lag48_mae']=float((z.loc[common,c]-old48[common]).abs().mean())
                periods.append(row)
        if farm not in ['F13','F47']:continue
        pairs=pd.read_csv(OUT/f'matched_weather_pairs_{farm}.csv')
        for r in pairs[pairs.gap.gt(1)].itertuples():
            a=g[g.day.eq(r.first_day)].sort_values('hour')
            b=g[g.day.eq(r.second_day)].sort_values('hour')
            assert len(a)==len(b)==24
            for c in features+TARGETS:
                av,bv=a[c].to_numpy(),b[c].to_numpy()
                valid=np.isfinite(av)&np.isfinite(bv)
                if not valid.any():continue
                repeats.append(dict(farm=farm,first_day=r.first_day,second_day=r.second_day,column=c,n=int(valid.sum()),equal_hours=int((av[valid]==bv[valid]).sum()),full_day_equal=bool(valid.all() and np.array_equal(av,bv)),mae=float(np.abs(av[valid]-bv[valid]).mean())))
    for name,rows in [('quality_columns',quality),('missing_runs',missing_runs),('constant_runs',constant_runs),('period_stability',periods),('distant_repeat_comparison',repeats)]:
        pd.DataFrame(rows).to_csv(OUT/f'final_{name}.csv',index=False)
    q,m,c,p,r=map(pd.DataFrame,[quality,missing_runs,constant_runs,periods,repeats])
    summary={}
    for farm in ['F13','F47']:
        fm=m[m.farm.eq(farm)];fc=c[c.farm.eq(farm)]
        rp=r[r.farm.eq(farm)]
        pp=p[p.farm.eq(farm)&p.target.eq('sub_ec')]
        summary[farm]=dict(structural_missing=q[q.farm.eq(farm)&q.all_missing].column.tolist(),
            missing_runs=fm.groupby('column').hours.agg(['count','sum','max']).to_dict('index'),
            longest_constants=fc.sort_values('hours',ascending=False).head(8).to_dict('records'),
            repeated_days=rp.groupby('column').agg(pairs=('n','size'),full_equal=('full_day_equal','sum'),mean_mae=('mae','mean')).to_dict('index'),
            ec_periods=pp.to_dict('records'))
    temperature=p[p.target.eq('sub_temp')].groupby('farm')[['jump_n','midnight_jump_n']].sum()
    summary['temperature_farms_with_large_jumps']=temperature[temperature.jump_n.gt(0)].to_dict('index')
    def clean(v):
        if isinstance(v,dict):return {k:clean(w) for k,w in v.items()}
        if isinstance(v,list):return [clean(w) for w in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        if isinstance(v,np.generic):return v.item()
        return v
    summary=clean(summary)
    (OUT/'final_structure_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
