"""Fit-zero source/cache/45-column and retained-median preparation audit."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as N
import numpy as np,pandas as pd
raw,jobs,old=N.B.prepare();p=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
fhash=lambda f,c:hashlib.sha256(pd.util.hash_pandas_object(f[c],index=False).to_numpy().tobytes()).hexdigest()
for key,path in [('source_sha',H/'run_v1.py'),('core_sha',H/'core_v1.py'),('plan_sha',H/'PLAN_v1.md'),('addendum_sha',H/'PLAN_ADDENDUM_v1.md'),('old_preparation_sha',N.OLD/'preparation_v4.json'),('old_completion_sha',N.OLD/'완료상태_v1.json'),('old_receipt_sha',N.OLD/'baseline_receipt_v4.json'),('old_runner_sha',N.OLD/'runner_v4.py'),('parent_trace_sha',N.B.L/'trace/BASE.npz')]:assert p[key]==sha(path)
expected=[c for c in N.B.M.FULL_R3 if c not in ['in_co2_h0','act_heating_h0']]
assert p['columns']==N.COLS==expected and len(expected)==45 and p['dropped']==['in_co2_h0','act_heating_h0'] and p['new_fit']==0
checks=[]
for k,(t,q,pfn) in jobs.items():
    rec=next(r for r in p['records'] if r['k']==k)
    assert len(t)==rec['train_rows'] and len(q)==rec['query_rows'] and fhash(t,['row_id','sub_ec']+expected)==rec['train_hash'] and fhash(q,['row_id','sub_ec']+expected)==rec['query_hash']
    allmedian=np.nanmedian(t[N.B.M.FULL_R3].to_numpy(float),axis=0);keep=[N.B.M.FULL_R3.index(c) for c in expected];med=np.nanmedian(t[expected].to_numpy(float),axis=0)
    assert np.array_equal(med,allmedian[keep]) and np.isfinite(med).all()
    for seed in [7,101,2024]:
        path=N.B.L/'base'/f'{k}_{seed}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert meta['sha']==rec['base_caches'][str(seed)]==sha(path) and meta['prep_sha']==p['old_preparation_sha'] and (meta['k'],meta['seed'])==(k,seed)
        with np.load(path,allow_pickle=False) as z:
            assert z['train_row_id'].tolist()==t.row_id.tolist() and z['row_id'].tolist()==q.row_id.tolist()
            for name in ['et','lgb','mlp']:assert z[name].shape==(len(q),) and np.isfinite(z[name]).all()
        checks.append(dict(fold=k,seed=seed,npz_sha=sha(path),new45_train_hash=rec['train_hash'],new45_query_hash=rec['query_hash']))
    if k==1:
        trace=N.B.L/'trace/BASE.npz';meta=json.loads(trace.with_suffix('.json').read_text(encoding='utf-8'));assert (meta['arm'],meta['seed'],meta['fold'])==('BASE',7,1) and meta['sha']==sha(trace)
        with np.load(trace,allow_pickle=False) as z:
            assert z['train_row_id'].tolist()==t.row_id.tolist();np.testing.assert_allclose(allmedian,z['imputer_median'],rtol=0,atol=1e-12)
            X=t[expected].to_numpy(float);X=np.where(np.isnan(X),med,X).astype(np.float32);assert np.array_equal(X,z['train_X'][:,keep])
            qt=q.set_index('row_id').loc[z['query_row_id']];Q=qt[expected].to_numpy(float);Q=np.where(np.isnan(Q),med,Q).astype(np.float32);assert np.array_equal(Q,z['query_X'][:,keep])
out=dict(status='PASS_TWO_H0_FIT0_PREPARATION_AND_MEDIAN_PROJECTION',new_fit=0,partial_scores=0,folds=10,base_caches=30,retained_columns=45,retained_median_projection='Exact across10fold; existingfold1seed7 stored47median within1e-12 and actual float32X/Q projected exactly.',checks=checks,scope='Current immutable features/season functions reused; bootstrap/score implementation not yet present and not audited.')
with (H/'critic_verify_preparation_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(out['status'],flush=True)
