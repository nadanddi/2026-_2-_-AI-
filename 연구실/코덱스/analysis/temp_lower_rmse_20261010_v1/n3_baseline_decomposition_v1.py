"""Describe N3 and its baseline by the same post-prediction midnight identity."""
from pathlib import Path
import sys,json,math,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
p=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
y=pd.read_csv(Path(p['temp_code']).parent/'data/train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
reg=json.loads((H/'n3_registration_v1.json').read_text(encoding='utf-8'))
parent=json.loads((H.parent/'temp_repair_20261009_v1/registration_v1.json').read_text(encoding='utf-8'))
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n3_target_v1'
ans={}
def desc(a,b):
    n=len(a);mse=lambda v:math.fsum(float(x)**2 for x in v)/n
    e0=math.fsum(map(float,a))/n;es=math.fsum(map(float,b))/n
    cross=2*math.fsum(float(x)*float(z) for x,z in zip(a,b))/n
    total=mse(a+b);assert abs(mse(a)+mse(b)+cross-total)<1e-12
    return {'level_mse':mse(a),'shape_mse':mse(b),'cross_2E':cross,'total_mse':total,'level_bias':e0,'shape_bias':es,'centered_cov_2':cross-2*e0*es}
for s,fds in parent['folds'].items():
    errs={m:[[],[]] for m in ['BASELINE','N3']}
    for k in range(len(fds)):
        fn=O/(s+'_'+str(k)+'_v1.npz');receipt=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
        assert receipt['npz_sha']==hashlib.sha256(fn.read_bytes()).hexdigest()
        z=np.load(fn,allow_pickle=True);ids=z['row_id'].tolist();h0=[i[:8]+'00' for i in ids]
        truth=y.loc[ids].to_numpy();truth0=y.loc[h0].to_numpy()
        for model in errs:
            pred=z[model];pred0=pd.Series(pred,index=ids).loc[h0].to_numpy()
            e0=pred0-truth0;es=(pred-pred0)-(truth-truth0)
            assert np.allclose(e0+es,pred-truth,rtol=0,atol=5e-14)
            errs[model][0].extend(e0);errs[model][1].extend(es)
    out={m:desc(np.array(a),np.array(b)) for m,(a,b) in errs.items()}
    out['delta']={k:out['N3'][k]-out['BASELINE'][k] for k in out['BASELINE']}
    assert abs(sum(out['delta'][k] for k in ['level_mse','shape_mse','cross_2E'])-out['delta']['total_mse'])<1e-12
    ans[s]=out
with (H/'n3_baseline_decomposition_v1.json').open('x',encoding='utf-8') as f:json.dump({'results':ans,'new_fit':0,'causal_attribution':False,'query_truth':'post-prediction only'},f,indent=2)
print('N3_BASELINE_DECOMPOSITION',json.dumps(ans))
