"""Intermediate DIAG10 checkpoint verification; no adoption or tuning."""
from pathlib import Path
import sys,csv,json,math,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
locks={(z['farm'],int(z['day'])) for z in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
truth={}
with (Path(env.DATA)/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        farm,day,_=r['row_id'].split('_')
        if farm not in ['F13','F47'] or (farm,int(day)) in locks:continue
        truth[r['row_id']]=float(r['sub_ec'])
frames=[];folds=[]
COLS=['mean','farm_mean','ridge','raw_et','full_et','r3','v2']
for i in range(10):
    p=OUT/f'DIAG10_{i}.npz'
    meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['prediction_sha256']
    with np.load(p,allow_pickle=False) as z:a=pd.DataFrame({k:z[k] for k in ['row_id']+COLS})
    a['y']=a.row_id.map(truth);assert a.y.notna().all()
    for c in COLS:folds.append({'fold':i,'model':c,'rmse':float(np.sqrt(np.mean((a[c]-a.y)**2)))})
    frames.append(a)
d=pd.concat(frames,ignore_index=True)
assert len(d)==8640 and d.row_id.is_unique and set(d.row_id)==set(truth)
f=pd.DataFrame(folds);results=[]
for c in COLS:
    value=float(np.sqrt(np.mean((d[c]-d.y)**2)))
    independent=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(d.y,d[c]))/len(d))
    assert math.isclose(value,independent,rel_tol=1e-12,abs_tol=1e-13)
    fs=f[f.model.eq(c)].rmse
    results.append({'model':c,'rmse':value,'fold_mean':float(fs.mean()),'fold_std':float(fs.std(ddof=0)),'n':len(d)})
pd.DataFrame(results).to_csv(HERE/'intermediate_diag10_scores_v1.csv',index=False,encoding='utf-8-sig')
(HERE/'intermediate_diag10_verification_v1.json').write_text(json.dumps({'status':'PASS','rows':8640,'days':360,'checkpoints':10,'label_alignment':'PASS','unique_day_coverage':'PASS','rmse_independent_fsum':'PASS','checkpoint_hashes':'PASS','candidate_adopted':False,'remaining_validators_pending':True},indent=2),encoding='utf-8')
print(pd.DataFrame(results).to_string(index=False))
