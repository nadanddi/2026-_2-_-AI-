"""Audit H20 by regime and test predeclared early moisture residual signals."""
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import hashlib,json
from datetime import datetime
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
LOCAL=ROOT/'집/코덱스/analysis/local'
H20=LOCAL/'ec_moisture_flux_h20/20260929_123524'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
OUT=LOCAL/'ec_water_proxy_d10'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True,exist_ok=False)
hashes={}
def read(path):
    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

frames=[];baseline=[]
for fold in (0,2,4,6,8,9):
    b=read((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
    b['v2']=b['blend'] if fold<8 else b['candidate']
    baseline.append(b[['row_id','sub_ec','v2']])
    for seed in (7,101):
        z=read(H20/f'fold{fold}_seed{seed}.csv')
        assert z.row_id.is_unique and set(z.row_id)==set(b.row_id)
        ref=b.set_index('row_id').loc[z.row_id]
        np.testing.assert_allclose(z.sub_ec,ref.sub_ec,rtol=0,atol=1e-12)
        np.testing.assert_allclose(z.v2,ref.v2,rtol=0,atol=1e-12)
        frames.append(z)
a=pd.concat(frames,ignore_index=True)
assert len(a)==11232
v=pd.concat(baseline,ignore_index=True)
assert len(v)==5616 and v.row_id.is_unique
v['farm']=v.row_id.str[:3];v['day']=v.row_id.str[4:7].astype(int)
day=v.groupby(['farm','day']).sub_ec.mean().rename('daily_ec')
assert len(day)==234
features=read(H20/'input_only_moisture.csv')
assert features.row_id.is_unique
early=features[features.hour.eq(6)].set_index(['farm','day'])[['ah_gap','vent_flux_cum','closed_rise_cum']]
assert len(early)==400

raw=read(Path(env.DATA)/'train_X.csv')
raw=raw[raw.row_id.isin(v.row_id)].copy()
raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int)
conditions=raw.groupby(['farm','day']).agg(fan=('act_circfan','mean'),vent_zero=('act_vent',lambda s:float((s==0).mean())))
conditions['sealed']=(conditions.fan<10)&conditions.vent_zero.gt(.85)

a['base_sq']=(a.sub_ec-a.v2)**2;a['h20_sq']=(a.sub_ec-a.h20)**2
daily=a.groupby(['seed','farm','day']).agg(n=('row_id','size'),base_sq=('base_sq','sum'),h20_sq=('h20_sq','sum'),y=('sub_ec','mean'),base=('v2','mean'),h20=('h20','mean'),flux=('flux_hour','sum')).reset_index()
assert daily.n.eq(24).all() and len(daily)==468
daily=daily.join(conditions['sealed'],on=['farm','day'])
daily['late']=daily.day.ge(179);daily['high']=daily.y.ge(1.2);daily['flux_present']=daily.flux.gt(0)
def metric(g):
    sb=float(g.base_sq.sum());sc=float(g.h20_sq.sum());n=int(g.n.sum())
    return dict(days=len(g),baseline=float(np.sqrt(sb/n)),h20=float(np.sqrt(sc/n)),relative_change=float(np.sqrt(sc/sb)-1),improved_days=int((g.h20_sq<g.base_sq).sum()),mean_daily_delta=float((g.h20-g.base).mean()),mean_daily_residual=float((g.y-g.base).mean()))
tables={}
for col in ('farm','late','sealed','high','flux_present'):
    tables[col]=[dict(seed=int(s),group=str(gval),**metric(g)) for (s,gval),g in daily.groupby(['seed',col])]
tables['fold']=[]
for (seed,fold),g in a.groupby(['seed','fold']):
    b=rmse(g.sub_ec,g.v2);c=rmse(g.sub_ec,g.h20)
    tables['fold'].append(dict(seed=int(seed),fold=int(fold),baseline=b,h20=c,relative_change=c/b-1))

e=v[v.row_id.str.endswith('_06')].set_index(['farm','day'])[['v2']].rename(columns={'v2':'v2_6'})
e=e.join(day).join(early).reset_index()
assert len(e)==234 and e[['v2_6','daily_ec']].notna().all().all()
e['late']=e.day.ge(179)
e['target']=e.daily_ec-e.v2_6
controls=np.column_stack([np.ones(len(e)),e.farm.eq('F47').astype(float),e.late.astype(float),e.day.astype(float),e.v2_6,e.v2_6**2])
def residual(y): return y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
y=residual(e.target.to_numpy(float))
xs=[];names=['ah_gap','vent_flux_cum','closed_rise_cum']
for name in names:
    z=e[name].copy()
    z=z.groupby(e.farm).transform(lambda s:s.fillna(s.median()))
    z=z.fillna(z.median()).to_numpy(float)
    q=residual(z)
    q=q-q.mean();q=q/(np.linalg.norm(q)+1e-12)
    xs.append(q)
X=np.column_stack(xs)
norm_y=np.linalg.norm(y)
corr=(X.T@y)/norm_y
rng=np.random.default_rng(300930)
groups=list(e.groupby(['farm','late']).indices.values())
max_abs=np.empty(10000)
for i in range(len(max_abs)):
    perm=y.copy()
    for idx in groups:perm[idx]=y[rng.permutation(idx)]
    max_abs[i]=np.max(np.abs(X.T@perm)/np.linalg.norm(perm))
pmax=(1+int(np.count_nonzero(max_abs>=np.max(np.abs(corr)))))/(len(max_abs)+1)
by_farm={}
for farm,g in e.groupby('farm'):
    idx=g.index.to_numpy();by_farm[farm]={}
    for name in names:
        k=names.index(name);q=X[idx,k];r=y[idx]
        by_farm[farm][name]=float(np.corrcoef(q,r)[0,1])
best=int(np.argmax(np.abs(corr)))
passes=bool(pmax<.01 and all(np.sign(by_farm[f][names[best]])==np.sign(corr[best]) and abs(by_farm[f][names[best]])>=.15 for f in by_farm))
result=dict(status='PASS',rows=5616,days=234,regimes=tables,early_signal=dict(names=names,partial_r=[float(x) for x in corr],p_max=float(pmax),by_farm=by_farm,best=names[best],passes_early_signal_gate=passes,permutations=len(max_abs)),sha256=hashes,note='Diagnostic only. Full-day sealed and label-defined high are not prediction inputs.')
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
daily.to_csv(OUT/'daily.csv',index=False)
print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
print('OUTPUT',OUT)
