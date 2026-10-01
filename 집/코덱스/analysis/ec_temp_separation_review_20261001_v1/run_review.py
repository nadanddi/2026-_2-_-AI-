from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,hashlib
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
LOCK=Path(env.CODEX)/'ec_final_lock/locked_days.json'
GROUP=ROOT/'집/클로드/research/re17_day_offsets.csv'
CACHE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
locks={(z['farm'],int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
counts=Counter();grecs=[]
with GROUP.open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        counts[r['group']]+=1
        if (r['farm'],int(r['day'])) in locks:continue
        grecs.append((r['farm'],int(r['day']),r['group'],float(r['temp_offset'])))
g=pd.DataFrame(grecs,columns=['farm','day','group','temp_offset'])
assert len(g)==360 and not g[['farm','day']].duplicated().any()
vals=defaultdict(list);truth={}
with (Path(env.DATA)/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        farm,day,_=r['row_id'].split('_');day=int(day)
        if farm not in ('F13','F47') or (farm,day) in locks:continue
        y=float(r['sub_ec']);truth[r['row_id']]=y;vals[(farm,day)].append(y)
manual={k:(math.fsum(a)/len(a),max(a)-min(a)) for k,a in vals.items()}
d=pd.read_csv(ROOT/'집/코덱스/analysis/ec_restart_20261001_v1/unlocked_daily_ec.csv')
assert len(d)==360
for r in d.itertuples():
    a,b=manual[(r.farm,r.day)]
    assert math.isclose(r.mean,a,abs_tol=2e-15) and math.isclose(r.range,b,abs_tol=2e-15)
d=d.rename(columns={'mean':'ec_mean','range':'ec_range'}).merge(g,on=['farm','day'],validate='one_to_one')
d['calendar_previous_change']=[manual[(f,int(day))][0]-manual[(f,int(day)-1)][0] if (f,int(day)-1) in manual else np.nan for f,day in zip(d.farm,d.day)]
d['late']=d.day.ge(179)
feat=pd.read_csv(ROOT/'집/코덱스/analysis/ec_restart_phase2_20261001_v1/features_h6.csv')
d=d.merge(feat[['farm','day','act_heating_mean','act_circfan_mean']],on=['farm','day'],validate='one_to_one')
water=pd.read_csv(ROOT/'집/코덱스/analysis/local/ec_moisture_flux_h20/20260929_123524/input_only_moisture.csv')
water=water[water.hour.eq(6)][['farm','day','closed_rise_cum']]
d=d.merge(water,on=['farm','day'],validate='one_to_one')
parts=[];snap=[]
for p in sorted(CACHE.glob('DIAG10_*.npz')):
    meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['prediction_sha256']
    with np.load(p,allow_pickle=False) as z:
        a=pd.DataFrame({'row_id':z['row_id'],'v2':z['v2']})
    a['y']=a.row_id.map(truth);assert a.y.notna().all()
    a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int)
    a['ec_error']=a.v2-a.y;parts.append(a);snap.append(p.name)
o=pd.concat(parts,ignore_index=True)
assert o.row_id.is_unique
res=o.groupby(['farm','day']).ec_error.mean().rename('ec_residual').reset_index()
d=d.merge(res,on=['farm','day'],how='left',validate='one_to_one')
rows=[]
metrics=['ec_mean','ec_range','calendar_previous_change','ec_residual','closed_rise_cum','act_circfan_mean','act_heating_mean']
for keys,a in [('all',d)]+[(f'{f}/late={late}',a) for (f,late),a in d.groupby(['farm','late'])]:
    for name,a1 in a.groupby('group'):
        for c in metrics:
            v=a1[c].dropna()
            if not len(v):continue
            m=float(v.mean());ind=math.fsum(float(x) for x in v)/len(v)
            assert math.isclose(m,ind,abs_tol=1e-12)
            rows.append({'stratum':keys,'group':name,'metric':c,'days':len(v),'mean':m,'median':float(v.median()),'positive_days':int(v.gt(0).sum())})
contrasts=[]
for c in metrics:
    a=d[d.group.isin(['over','under'])&d[c].notna()].copy()
    X=np.column_stack([np.ones(len(a)),a.group.eq('over'),a.farm.eq('F47'),a.late,a.day])
    if len(a)>X.shape[1] and np.linalg.matrix_rank(X)==X.shape[1]:
        beta=np.linalg.lstsq(X,a[c].to_numpy(float),rcond=None)[0]
        contrasts.append({'metric':c,'n_over':int(a.group.eq('over').sum()),'n_under':int(a.group.eq('under').sum()),'adjusted_over_minus_under':float(beta[1])})
d.to_csv(HERE/'daily_comparison.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(HERE/'group_summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(contrasts).to_csv(HERE/'adjusted_contrasts.csv',index=False,encoding='utf-8-sig')
result={'status':'PASS','groups_original_string_only':dict(counts),'groups_unlocked':dict(d.group.value_counts()),'oof_snapshot':snap,'oof_days':len(res),'oof_rows':len(o),'residual_definition':'prediction minus truth','verification':['EC daily mean/range independent CSV arithmetic','all group means independent math.fsum','checkpoint SHA256 and unique row IDs','lock values excluded before numeric parsing'],'tests':0,'candidate_adopted':False,'limitations':['Partial DIAG10 OOF','Exploratory reused data, no causal identification','Calendar previous day is not verified same-source previous day','Group chosen by temperature model error; not available at prediction time']}
(HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).query("stratum=='all'").to_string(index=False))
print(pd.DataFrame(contrasts).to_string(index=False))
