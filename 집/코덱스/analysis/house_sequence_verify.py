"""Independent continuity control and aggregate test-weather check.

Retrospective investigation only; never use a current day's future input to
infer an online feature. Test values stay local; stdout contains aggregates.
"""
import json
import numpy as np
import pandas as pd
from profile_tabular import load,OUT,TARGETS
from house_sequence_audit import WEATHER


def main():
    x=load('train_X.csv'); y=load('train_y.csv'); test=load('test_X.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left')
    report={}
    for f in ['F13','F47']:
        g=d[d.farm==f].copy(); te=test[test.farm==f]
        idx=g.set_index(['day','hour'])
        table=g.pivot(index='day',columns='hour',values=WEATHER).dropna()
        sig=pd.util.hash_pandas_object(table,index=False)
        run_id=(sig.ne(sig.shift())|sig.index.to_series().diff().ne(1)).cumsum()
        runs=[list(v) for v in sig.groupby(run_id).groups.values()]
        records=[]
        for prev,cur in zip(runs,runs[1:]):
            if len(prev)!=2 or len(cur)!=2 or cur[0]!=prev[-1]+1:continue
            for slot in [0,1]:
                for target in TARGETS+['in_temp','in_hum','in_co2']:
                    now=idx.loc[(cur[slot],0),target]
                    same=idx.loc[(prev[slot],23),target]
                    swapped=idx.loc[(prev[1-slot],23),target]
                    if not np.isfinite([now,same,swapped]).all():continue
                    records.append(dict(block=int(cur[0]),slot=slot,target=target,same=abs(now-same),swapped=abs(now-swapped)))
        r=pd.DataFrame(records); summary={}
        rng=np.random.default_rng(20260919)
        for target,a in r.groupby('target'):
            delta=a.groupby('block').apply(lambda b: (b.swapped-b.same).mean(),include_groups=False).to_numpy()
            boot=np.mean(rng.choice(delta,size=(10000,len(delta)),replace=True),axis=1)
            summary[target]=dict(n_endpoints=len(a),n_blocks=len(delta),
                same_mean=float(a.same.mean()),swapped_mean=float(a.swapped.mean()),
                fraction_same_better=float(a.same.lt(a.swapped).mean()),
                mean_improvement=float(delta.mean()),
                block_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
        # Use identical column ordering. Every evaluated weather value remains local.
        tt=te.pivot(index='day',columns='hour',values=WEATHER).reindex(columns=table.columns).dropna()
        th=pd.util.hash_pandas_object(tt,index=False)
        all_table=pd.concat([table,tt]).sort_index(); ah=pd.util.hash_pandas_object(all_table,index=False)
        weather=dict(test_complete_days=len(tt),test_unique_patterns=int(th.nunique()),
            test_days_repeated_within_test=int(th.duplicated(False).sum()),
            test_days_matching_train=int(th.isin(sig).sum()),
            combined_days=len(all_table),combined_unique_patterns=int(ah.nunique()),
            combined_multiplicity_histogram=ah.value_counts().value_counts().sort_index().to_dict())
        # Select the strongest sequential four-day example, report TRAIN only.
        examples=[]
        for prev,cur in zip(runs,runs[1:]):
            if len(prev)!=2 or len(cur)!=2 or cur[0]!=prev[-1]+1:continue
            daymeans=g[g.day.isin(prev+cur)].groupby('day').sub_ec.mean()
            separation=float(abs(daymeans.loc[prev[0]]-daymeans.loc[prev[1]]))
            daily=[]
            for day in prev+cur:
                q=g[g.day==day]
                daily.append(dict(relative_day=int(day),ec_00=float(q.loc[q.hour==0,'sub_ec'].iloc[0]),
                    ec_23=float(q.loc[q.hour==23,'sub_ec'].iloc[0]),ec_mean=float(q.sub_ec.mean()),
                    in_temp_mean=float(q.in_temp.mean()),fan_mean=float(q.act_circfan.mean())))
            examples.append(dict(separation=separation,days=daily))
        report[f]=dict(continuity=summary,weather_aggregate=weather,
            training_examples=sorted(examples,key=lambda z:z['separation'],reverse=True)[:3])
    (OUT/'house_sequence_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
