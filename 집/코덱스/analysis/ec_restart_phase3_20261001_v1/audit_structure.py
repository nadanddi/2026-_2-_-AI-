"""Run all input/ID-only structural checks independently of model fitting."""
import run_benchmark as b
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    raw,full,lab,lock,sigs,fds=b.prepare()
    audits=[];validation_days=[]
    for name,i,days in fds:
        va=lab[[(f,int(d)) in days for f,d in zip(lab.farm,lab.day)]].reset_index(drop=True)
        tr=lab[b.near_mask(lab,days|lock)].reset_index(drop=True)
        if va.empty:continue
        audits.append(b.audit_fold(name,i,tr,va,sigs,full))
        validation_days.extend((name,i,f,int(day)) for f,day in va[['farm','day']].itertuples(index=False,name=None))
    t=pd.DataFrame(validation_days,columns=['validator','fold','farm','day'])
    counts=t.groupby(['validator','farm','day']).size()
    overlap={name:{'day_occurrences':int(g.sum()),'unique_days':len(g),'days_repeated':int(g.gt(1).sum()),'max_occurrences':int(g.max())}
             for name,g in counts.groupby(level=0)}
    pairs=[]
    for kind in ['weather','full14']:
        groups={}
        for key,z in sigs.items():groups.setdefault(z[kind],[]).append(key)
        repeated=[v for v in groups.values() if len(v)>1]
        pairs.append({'signature':kind,'groups':len(groups),'repeated_groups':len(repeated),'repeated_day_members':sum(map(len,repeated)),
                      'cross_farm_groups':sum(len({k[0] for k in v})>1 for v in repeated)})
    b.save(b.HERE/'structure_full_audit.json',{'folds':audits,'validation_overlap':overlap,'training_day_fingerprint':pairs,
               'structural_ids':b.structural_ids(full,lab,lock),'masked_features':['in_rad','act_side','act_valve','act_cool','act_pump'],
               'features_current18':b.RAW18,'features_v2_38':b.core.FULL,'interpretation':'whole-day fingerprints are audit-only, never model features'})
    print(pd.DataFrame(audits)[['validator','fold','train_days','val_days','weather_duplicate_val_days','full14_duplicate_val_days']].to_string(index=False))
    print(overlap);print(pairs)

if __name__=='__main__':main()
