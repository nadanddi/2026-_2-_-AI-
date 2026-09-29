"""Independent arithmetic audit of previously scored EC predictions only."""
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd

LOCAL = ROOT / '집/코덱스/analysis/local'
OUT = LOCAL / 'ec_evidence_audit_v1' / datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True, exist_ok=False)
HASHES = {}
def read(path):
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path) if path.suffix == '.csv' else json.loads(path.read_text(encoding='utf-8'))
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
frames = []
for fold in (0,2,4,6,8,9):
    folder = 'ec_three_seed_ensemble_cv/20260928_035914' if fold < 8 else 'ec_locked_confirmation/20260928_044934'
    z = read(LOCAL / folder / f'fold{fold}.csv')
    z['v2'] = z['blend'] if fold < 8 else z['candidate']
    z['fold'] = fold
    frames.append(z[['row_id','farm','day','sub_ec','baseline','v2','fold']])
v = pd.concat(frames, ignore_index=True)
assert len(v)==5616 and v.row_id.is_unique
assert np.isfinite(v[['sub_ec','baseline','v2']]).all().all()
daily = v.assign(error=v.sub_ec-v.v2, sq=(v.sub_ec-v.v2)**2).groupby(['farm','day']).agg(n=('row_id','size'), y=('sub_ec','mean'), e=('error','mean'), sq=('sq','sum'))
assert len(daily)==234 and daily.n.eq(24).all()
ref = v.set_index('row_id')
summary = []
for h in range(11,23):
    folders = list(LOCAL.glob(f'*_h{h}'))
    assert len(folders)==1, (h, folders)
    runs = sorted(folders[0].glob('*/result.json'))
    assert len(runs)==1, (h, runs)
    path = runs[0]
    saved = read(path)
    entries = saved['scores']
    assert len(entries)==12
    assert {(int(s['fold']),int(s['seed'])) for s in entries} == {(f,s) for f in (0,2,4,6,8,9) for s in (7,101)}
    all_rows = []
    for s in entries:
        fold, seed = int(s['fold']), int(s['seed'])
        z = read(path.parent / f'fold{fold}_seed{seed}.csv')
        assert z.row_id.is_unique
        assert set(z.row_id)==set(v.loc[v.fold.eq(fold),'row_id'])
        r = ref.loc[z.row_id]
        assert np.isfinite(z[['sub_ec','v2',f'h{h}']]).all().all()
        for col in ('sub_ec','v2'):
            assert np.allclose(z[col],r[col],rtol=0,atol=1e-12), (h,fold,seed,col)
        base, cand = rmse(z.sub_ec,z.v2), rmse(z.sub_ec,z[f'h{h}'])
        assert abs(base-s['v2']) < 1e-10
        assert abs(cand-s[f'h{h}']) < 1e-10
        assert abs(cand/base-1-s['relative_change']) < 1e-10
        z['seed']=seed
        all_rows.append(z)
    a = pd.concat(all_rows,ignore_index=True)
    by_seed = []
    for seed,g in a.groupby('seed'):
        by_seed.append(dict(seed=int(seed),rmse=rmse(g.sub_ec,g[f'h{h}']),relative_change=rmse(g.sub_ec,g[f'h{h}'])/rmse(g.sub_ec,g.v2)-1,improved_folds=sum(s['relative_change']<0 for s in entries if s['seed']==seed)))
    summary.append(dict(h=h,by_seed=by_seed,all_cells_improved=all(s['relative_change']<0 for s in entries)))
confirm = v[v.fold.isin([8,9])]
record = read(LOCAL/'ec_locked_confirmation/20260928_044934/result.json')
for col,key in [('baseline','baseline'),('v2','candidate')]:
    assert abs(rmse(confirm.sub_ec,confirm[col])-record['pooled'][key])<1e-10
result = dict(status='PASS',rows=len(v),days=len(daily),v2_rmse=rmse(v.sub_ec,v.v2),confirmation_baseline=rmse(confirm.sub_ec,confirm.baseline),confirmation_v2=rmse(confirm.sub_ec,confirm.v2),level_error_share=float((daily.n*daily.e**2).sum()/daily.sq.sum()),high_days=int(daily.y.ge(1.2).sum()),high_error_share=float(daily.loc[daily.y.ge(1.2),'sq'].sum()/daily.sq.sum()),hypotheses=summary,sha256=HASHES,note='Arithmetic audit, not independent validation or candidate adoption. Final lock not read.')
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:val for k,val in result.items() if k!='sha256'},ensure_ascii=True,indent=2))
print('OUTPUT',str(OUT))
