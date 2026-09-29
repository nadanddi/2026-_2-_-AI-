"""Test the hypothesis that pseudo-days concatenate multiple houses per real date.

Training-only retrospective diagnostics. Full-day signatures and inferred slots
are NOT legal online features by themselves and are not added to the model.
"""
import json
import itertools
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS

WEATHER=['out_temp','out_hum','out_rad','out_wspd']


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left',validate='one_to_one')
    result={}; daily_frames={}; arrays={}
    for farm in ['F13','F47','F32']:
        g=d[d.farm==farm].copy()
        available=[c for c in WEATHER if g[c].notna().any()]
        p=g.pivot(index='day',columns='hour',values=available).reindex(columns=pd.MultiIndex.from_product([available,range(24)]))
        p=p.dropna()
        h=pd.util.hash_pandas_object(p,index=False)
        # Independently verify equal hashes are actual array matches.
        for _,indices in h.groupby(h).groups.items():
            block=p.loc[indices].to_numpy()
            assert np.array_equal(block,np.repeat(block[:1],len(block),axis=0))
        key=(h.ne(h.shift())|h.index.to_series().diff().ne(1)).cumsum()
        daily=g.groupby('day')[available+['in_temp','in_hum','in_co2','act_circfan','act_vent']+TARGETS].mean().reindex(p.index)
        daily['signature']=h; daily['run']=key; daily['day']=daily.index
        daily['slot']=daily.groupby('run').cumcount()
        daily['run_size']=daily.groupby('run').day.transform('size')
        counts=h.value_counts(); runs=daily.groupby('run').day.agg(list)
        run_lengths=runs.map(len)
        info=dict(n_days=len(p),weather_variables=available,n_unique_signatures=h.nunique(),
            duplicate_days=int(h.duplicated(False).sum()),multiplicity_histogram=counts.value_counts().sort_index().to_dict(),
            consecutive_run_length_histogram=run_lengths.value_counts().sort_index().to_dict(),
            same_weather_adjacent_pairs=int((h.eq(h.shift())&h.index.to_series().diff().eq(1)).sum()),
            runs=[dict(days=ds,n=len(ds)) for ds in runs if len(ds)>1],
            nonadjacent_signature_groups=[list(map(int,ids)) for _,ids in h.groupby(h).groups.items() if len(ids)>1 and max(ids)-min(ids)+1!=len(ids)])
        pair_effect=[]
        for ds in runs:
            if len(ds)!=2:continue
            a,b=ds
            item={'first_day':int(a),'second_day':int(b)}
            for c in ['in_temp','in_hum','act_circfan','act_vent']+TARGETS:
                item[c+'_first']=float(daily.loc[a,c]); item[c+'_second']=float(daily.loc[b,c])
                item[c+'_delta']=float(daily.loc[b,c]-daily.loc[a,c])
            pair_effect.append(item)
        pairs=pd.DataFrame(pair_effect)
        if len(pairs):
            info['two_day_pair_effects']={c:dict(n=int(pairs[c+'_delta'].count()),
                mean_first=float(pairs[c+'_first'].mean()),mean_second=float(pairs[c+'_second'].mean()),
                median_delta=float(pairs[c+'_delta'].median()),mean_delta=float(pairs[c+'_delta'].mean()),
                fraction_second_greater=float(pairs[c+'_delta'].gt(0).mean())) for c in ['in_temp','act_circfan']+TARGETS}
            pairs.to_csv(OUT/f'house_pairs_{farm}.csv',index=False)
        # Bridge consecutive two-slot weather groups: compare raw midnight and
        # matched-slot next-group midnight for the exact same endpoints.
        indexed=g.set_index(['day','hour'])
        bridge=[]
        runlist=list(runs)
        for prev,cur in zip(runlist,runlist[1:]):
            if len(prev)!=2 or len(cur)!=2 or cur[0]!=prev[-1]+1:continue
            for slot in [0,1]:
                current=cur[slot]; same=prev[slot]; naive=current-1
                for c in ['in_temp','in_hum','in_co2']+TARGETS:
                    v=indexed.loc[(current,0),c]
                    old_same=indexed.loc[(same,23),c]
                    old_naive=indexed.loc[(naive,23),c]
                    if not np.isfinite([v,old_same,old_naive]).all():continue
                    bridge.append(dict(column=c,slot=slot,current_day=int(current),same_slot_abs_diff=abs(v-old_same),nominal_abs_diff=abs(v-old_naive)))
        b=pd.DataFrame(bridge)
        if len(b):
            info['boundary_continuity']=b.groupby('column')[['same_slot_abs_diff','nominal_abs_diff']].agg(['count','mean','median']).to_dict()
            # Flatten tuple keys for JSON.
            info['boundary_continuity']={'__'.join(k):v for k,v in info['boundary_continuity'].items()}
            b.to_csv(OUT/f'house_continuity_{farm}.csv',index=False)
        daily.to_csv(OUT/f'house_daily_{farm}.csv',index=False)
        daily_frames[farm]=daily; arrays[farm]=p; result[farm]=info
    a=arrays['F13']; b=arrays['F47']
    ha=pd.util.hash_pandas_object(a,index=False); hb=pd.util.hash_pandas_object(b,index=False)
    common=set(ha)&set(hb)
    # Count shared weather dates regardless of pseudoday offset.
    result['cross_F13_F47']=dict(common_signatures=len(common),F13_days_matching=int(ha.isin(common).sum()),F47_days_matching=int(hb.isin(common).sum()))
    matched=[]
    for sig in common:
        aa=ha.index[ha==sig].tolist(); bb=hb.index[hb==sig].tolist()
        assert np.array_equal(a.loc[aa[0]].to_numpy(),b.loc[bb[0]].to_numpy())
        matched.append(dict(F13_days=aa,F47_days=bb))
    result['cross_F13_F47']['matched_groups']=sorted(matched,key=lambda z:z['F13_days'][0])
    # LCS of run-compressed signatures: ordered correspondence versus a constant offset.
    sa=daily_frames['F13'].groupby('run').signature.first().tolist()
    sb=daily_frames['F47'].groupby('run').signature.first().tolist()
    dp=np.zeros((len(sa)+1,len(sb)+1),dtype=int)
    for i,aa in enumerate(sa,1):
        for j,bb in enumerate(sb,1):dp[i,j]=dp[i-1,j-1]+1 if aa==bb else max(dp[i-1,j],dp[i,j-1])
    result['cross_F13_F47']['compressed_sequence_lengths']=[len(sa),len(sb)]
    result['cross_F13_F47']['longest_common_subsequence']=int(dp[-1,-1])
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(z) for k,z in v.items()}
        if isinstance(v,list):return [clean(z) for z in v]
        if isinstance(v,np.integer):return int(v)
        if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
        return v
    (OUT/'house_sequence_audit.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2),encoding='utf8')
    for f in ['F13','F47','F32']:
        print(f,json.dumps(clean({k:v for k,v in result[f].items() if k not in ['runs','nonadjacent_signature_groups']}),ensure_ascii=False))
        print('FIRST REPEATED RUNS',result[f]['runs'][:15])
    print('CROSS',json.dumps({k:v for k,v in result['cross_F13_F47'].items() if k!='matched_groups'}))
    print('CROSS FIRST GROUPS',result['cross_F13_F47']['matched_groups'][:12])


if __name__=='__main__':main()
