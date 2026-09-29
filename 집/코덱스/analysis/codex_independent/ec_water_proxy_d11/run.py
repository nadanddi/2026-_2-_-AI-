"""Sensitivity checks for the selected early closed-humidity-rise proxy."""
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib
from datetime import datetime
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
LOCAL=ROOT/'집/코덱스/analysis/local'
H20=LOCAL/'ec_moisture_flux_h20/20260929_123524'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
OUT=LOCAL/'ec_water_proxy_d11'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True,exist_ok=False)
hashes={}
def read(p):
    hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)
frames=[]
for fold in (0,2,4,6,8,9):
    p=(SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv'
    z=read(p);z['v2']=z['blend'] if fold<8 else z['candidate']
    z['fold']=fold
    frames.append(z[['row_id','farm','day','hour','sub_ec','v2','fold']])
v=pd.concat(frames,ignore_index=True)
assert len(v)==5616 and v.row_id.is_unique
f=read(H20/'input_only_moisture.csv')
early=f[f.hour.eq(6)][['row_id','closed_rise_cum']]
day=v.groupby(['farm','day']).sub_ec.mean().rename('day_ec').reset_index()
e=v[v.hour.eq(6)][['row_id','farm','day','fold','v2']].merge(early,on='row_id',validate='one_to_one').merge(day,on=['farm','day'],validate='one_to_one')
assert len(e)==234 and e.row_id.is_unique
e['late']=e.day.ge(179)
e['target']=e.day_ec-e.v2
e['rise']=e.closed_rise_cum.groupby(e.farm).transform(lambda s:s.fillna(s.median()))
e['rise']=e.rise.fillna(e.rise.median())
assert e[['rise','target','v2']].notna().all().all()
def residual(y,c):return y-c@np.linalg.lstsq(c,y,rcond=None)[0]
def partial(frame):
    c=np.column_stack([np.ones(len(frame)),frame.farm.eq('F47').astype(float),frame.late.astype(float),frame.day.astype(float),frame.v2,frame.v2**2])
    x=residual(frame.rise.to_numpy(float),c)
    y=residual(frame.target.to_numpy(float),c)
    if np.linalg.norm(x)<1e-12 or np.linalg.norm(y)<1e-12:return float('nan')
    return float(np.corrcoef(x,y)[0,1])
e=e.reset_index(drop=True)
full=partial(e)
c=np.column_stack([np.ones(len(e)),e.farm.eq('F47').astype(float),e.late.astype(float),e.day.astype(float),e.v2,e.v2**2])
target_residual=residual(e.target.to_numpy(float),c)
top5=np.argsort(np.abs(target_residual))[-5:]
subgroups={'all':e,'without_top5':e.drop(top5),'without_high_ec':e[e.day_ec.lt(1.2)],
           'F13':e[e.farm.eq('F13')],'F47':e[e.farm.eq('F47')],
           'early':e[~e.late],'late':e[e.late]}
sub=[dict(group=name,days=len(g),partial_r=partial(g)) for name,g in subgroups.items()]
blocks={key:[g.index.to_numpy()[i:i+5] for i in range(0,len(g),5)]
        for key,g in e.sort_values('day').groupby(['farm','late'])}
rng=np.random.default_rng(300931)
boots=[]
for _ in range(3000):
    ids=[]
    for bs in blocks.values():
        ids.extend(np.concatenate([bs[j] for j in rng.integers(0,len(bs),len(bs))]))
    boots.append(partial(e.iloc[ids]))
boots=np.asarray(boots,float)
assert np.isfinite(boots).all()
ci=np.quantile(boots,[.025,.975]).tolist()
h20=read(LOCAL/'ec_water_proxy_d10/20260930_025147/daily.csv')
assert len(h20)==468
correction=[]
for seed,z in h20.groupby('seed'):
    joined=e.merge(z[['farm','day','base','h20','y']],on=['farm','day'],validate='one_to_one')
    np.testing.assert_allclose(joined.day_ec,joined.y,atol=1e-12,rtol=0)
    delta=joined.h20-joined.base;resid=joined.y-joined.base
    correction.append(dict(seed=int(seed),corr_rise_delta=float(np.corrcoef(joined.rise,delta)[0,1]),corr_delta_residual=float(np.corrcoef(delta,resid)[0,1]),matching_direction=float(np.mean(np.sign(delta)==np.sign(resid))),mean_abs_delta=float(np.mean(np.abs(delta)))))
result=dict(status='PASS',days=234,overall_partial_r=full,subgroups=sub,block5_bootstrap=dict(draws=3000,seed=300931,ci95=ci,share_positive=float(np.mean(boots>0))),h20_correction=correction,top5_ids=e.iloc[top5][['farm','day','day_ec']].to_dict('records'),sha256=hashes,note='Post-selection sensitivity, not independent confirmation or model adoption.')
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
print('OUTPUT',OUT)
