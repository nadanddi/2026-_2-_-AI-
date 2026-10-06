from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
D=H/'ablation_score_v1'
g=pd.read_csv(D/'groups.csv',float_precision='round_trip',dtype={'seed':str})
d=pd.read_csv(D/'days.csv',float_precision='round_trip',dtype={'seed':str})
b=d[d.arm.eq('BASE')].set_index(['seed','farm','day'])
changes=[];summaries=[]
for arm in ['D1','D2']:
    a=d[d.arm.eq(arm)].set_index(['seed','farm','day']);assert set(a.index)==set(b.index);a=a.loc[b.index]
    c=a[['truth','prediction','sse','raw_et','rmse','ordinary','pass2']].copy();c['delta_sse']=a.sse-b.sse;c['prediction_change']=a.prediction-b.prediction;c['raw_et_change']=a.raw_et-b.raw_et;c['arm']=arm;changes.append(c.reset_index())
    for seed in [*map(str,R.SEEDS),'ensemble']:
        u=c.xs(seed,level='seed');selected=u.loc[('F47',161)];stats=g[g.arm.eq(arm)&g.seed.eq(seed)].set_index('group')
        ordinary_gain=-float(stats.loc['ordinary','delta_sse']);selected_gain=-float(selected.delta_sse)
        summaries.append(dict(arm=arm,seed=seed,ordinary_net_sse_gain=ordinary_gain,F47_161_sse_gain=selected_gain,selected_fraction_of_ordinary_net_gain=selected_gain/ordinary_gain if ordinary_gain else None,ordinary_remaining_delta_sse=float(stats.loc['ordinary_excluding_F47_161','delta_sse']),improved_days=int((u.delta_sse < -1e-12).sum()),worse_days=int((u.delta_sse > 1e-12).sum()),unchanged_days=int((abs(u.delta_sse)<=1e-12).sum()),positive_loss_sum=float(u.loc[u.delta_sse>0,'delta_sse'].sum()),negative_loss_sum=float(u.loc[u.delta_sse<0,'delta_sse'].sum())))
changes=pd.concat(changes,ignore_index=True);changes.to_csv(H/'day_changes_v1.csv',index=False)
changes[changes.seed.eq('ensemble')].sort_values('delta_sse',ascending=False).groupby('arm').head(12).to_csv(H/'largest_losses_v1.csv',index=False)
changes[changes.seed.eq('ensemble')].sort_values('delta_sse').groupby('arm').head(12).to_csv(H/'largest_gains_v1.csv',index=False)
s=pd.read_csv(H/'influence_support_days_v1.csv',float_precision='round_trip');s=s[s.level.eq('smooth_day')];base=s[s.arm.eq('BASE')].set_index(['farm','day']);shifts=[]
for arm in ['D1','D2']:
    a=s[s.arm.eq(arm)].set_index(['farm','day']);keys=base.index.union(a.index);u=pd.DataFrame(index=keys);u['base_weight']=base.weight.reindex(keys,fill_value=0);u['arm_weight']=a.weight.reindex(keys,fill_value=0);u['delta_weight']=u.arm_weight-u.base_weight;u['base_contribution']=base.contribution.reindex(keys,fill_value=0);u['arm_contribution']=a.contribution.reindex(keys,fill_value=0);u['delta_contribution']=u.arm_contribution-u.base_contribution;u['arm']=arm;shifts.append(u.reset_index())
pd.concat(shifts,ignore_index=True).to_csv(H/'support_shifts_v1.csv',index=False)
R.write(H/'diagnostic_summary_v1.json',dict(status='COMPLETE_DIAGNOSTIC_SUMMARY',source_sha=R.sha(__file__),source_scores_sha=R.sha(D/'completion.json'),summaries=summaries,files={name:R.sha(H/name) for name in ['day_changes_v1.csv','largest_losses_v1.csv','largest_gains_v1.csv','support_shifts_v1.csv']},adoption=False))
print('DIAGNOSTIC_SUMMARY_COMPLETE',flush=True)
