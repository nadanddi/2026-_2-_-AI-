from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
raw,jobs,prep=R.prepare();M,SG=R.M,R.SG
base=M.features(raw[['row_id']+M.RAW]).set_index('row_id');cols=[c for c in M.FULL_R3 if c!='season'];meta=R.S.M.identify(raw);times=meta.day*24+meta.hour
checks=[]
for farm in ['F13','F47']:
    for fraction in [.2,.5,.8]:
        cut=int(times[meta.farm.eq(farm)].quantile(fraction));allowed=meta.farm.eq(farm)&times.le(cut);changed=raw.copy();changed.loc[~allowed,M.RAW]=changed.loc[~allowed,M.RAW]*13+97
        altered=M.features(changed[['row_id']+M.RAW]).set_index('row_id');ids=meta.loc[allowed,'row_id'];pd.testing.assert_frame_equal(base.loc[ids,cols],altered.loc[ids,cols]);checks.append(dict(kind='nonseason_features',farm=farm,cut=cut,rows=len(ids)))
for farm in ['F13','F47']:
    k=next(k for k,(t,q,p) in jobs.items() if ((q.farm==farm)&(q.day>=179)).any());t,q,_=jobs[k];ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();structure=SG.prepare(raw,ref);cal=SG.ref_calendar(structure,ref);p=np.full(len(q),.7);original=SG.correct(q[['row_id']],p,structure,ec,ref,cal)
    qt=q.day*24+q.hour;refrows=set(t.row_id)
    for fraction in [.2,.5,.8]:
        cut=int(qt[(q.farm==farm)&(q.day>=179)].quantile(fraction));allowed_q=(q.farm==farm)&qt.le(cut);allowed_raw=(meta.farm==farm)&times.le(cut);changed=raw.copy();mask=~changed.row_id.isin(refrows)&~allowed_raw;variables=M.RAW+SG.W;changed.loc[mask,variables]=changed.loc[mask,variables]*13+97
        alt_structure=SG.prepare(changed,ref);pp=p.copy();pp[~allowed_q]=pp[~allowed_q]+1
        altered=SG.correct(q[['row_id']],pp,alt_structure,ec,ref,SG.ref_calendar(alt_structure,ref));np.testing.assert_array_equal(original[allowed_q],altered[allowed_q]);pd.testing.assert_frame_equal(structure['WV'].loc[sorted(ref)],alt_structure['WV'].loc[sorted(ref)])
        checks.append(dict(kind='SG2_future_other_farm_predictions',fold=k,farm=farm,cut=cut,rows=int(allowed_q.sum()),changed_rows_in_tested_prefix=int(np.sum(original[allowed_q]!=p[allowed_q]))))
allref=set(meta[['farm','day']].itertuples(index=False,name=None));a=SG.prepare(raw,allref);b=M.sg2post.prepare(raw);pd.testing.assert_frame_equal(a['WV'],b['WV']);assert a['second']==b['second']
for hour in range(24):pd.testing.assert_frame_equal(a['SIG'][hour],b['SIG'][hour])
R.write(H/'real_checks_v1.json',dict(status='PASS_REAL_INPUT_AND_SG2_CAUSALITY',checks=checks,fullref_package_equivalence=True,fit=0,model_predictions_recomputed=False,season_held_by_training_recipe=True))
print('PASS_REAL_INPUT_AND_SG2_CAUSALITY',len(checks),flush=True)
