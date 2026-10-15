"""Independent scalar/identity checks; does not train or access EC values."""
from pathlib import Path
import sys, json, math, hashlib, argparse, os
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np, pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--first',action='store_true');args=ap.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
rs=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
assert reg['source_sha']==sha(H/'n4_run_v2.py')
prep=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
data=Path(prep['temp_code']).parent/'data'
tx=pd.read_csv(data/'train_X.csv',usecols=['row_id','in_temp']).set_index('row_id')
y=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1'
checks=0;gaps=[];result={}
selected=reg['folds'][:1] if args.first else reg['folds']
for fd in selected:
    fn=O/(fd['set']+'_'+str(fd['fold'])+'.npz');receipt=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
    assert receipt['recipe_sha']==rs and receipt['npz_sha']==sha(fn)
    assert receipt['query_ids']==fd['query_ids'] and receipt['train_ids']==fd['train_ids']
    z=np.load(fn,allow_pickle=True);ids=z['row_id'].tolist();assert ids==fd['query_ids'];checks+=3
    assert not set(ids)&set(fd['train_ids'])
    assert len(z['context_ids'])==2000 and len(set(z['context_ids']))==2000 and set(z['context_ids'])<=set(fd['train_ids']);checks+=2
    air=tx.loc[ids,'in_temp'].to_numpy();g=np.where(np.isnan(air),1.,np.minimum(1.,np.maximum(0.,(air-8.)/2.)))
    assert np.array_equal(g,z['gate']) and receipt['seed_noop_max']==0.;checks+=2
    a=.4*z['BASE']+(.6-.4*g)*z['CODEX']+.4*g*z['PFN']
    b=a+(.6-.4*g)*g*(z['IRR']-z['CODEX'])
    gaps.extend([float(np.max(np.abs(a-z['BASELINE']))),float(np.max(np.abs(b-z['N4'])))])
    assert np.allclose(a,z['BASELINE'],rtol=0,atol=1e-13) and np.allclose(b,z['N4'],rtol=0,atol=1e-13)
    assert np.array_equal(z['BASELINE'][g==0],z['N4'][g==0]);checks+=3
    actual=y.loc[ids].to_numpy()
    rm=lambda p:math.sqrt(math.fsum(float(v)*float(v) for v in p-actual)/len(actual))
    result[fd['set']+'_'+str(fd['fold'])]={'base_rmse':rm(z['BASELINE']),'candidate_rmse':rm(z['N4']),'rows':len(ids),'gate0_rows':int((g==0).sum())}
    checks+=2
out={'checks':checks,'max_formula_gap':max(gaps),'fold_results':result,'passed':True,'recipe_sha':rs,'mode':'first' if args.first else 'all','new_fit':0,'full_prediction_causality_test':False}
name='n4_verify_first_v1.json' if args.first else 'n4_verify_all_v1.json'
with (H/name).open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('N4_VERIFY',json.dumps(out))
