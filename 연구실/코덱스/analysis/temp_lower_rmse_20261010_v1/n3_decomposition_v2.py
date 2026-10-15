"""Post-prediction only decomposition of midnight level and relative shape errors."""
from pathlib import Path
import sys,json,math
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
p=json.loads((R/'연구실/코덱스/analysis/temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
y=pd.read_csv(Path(p['temp_code']).parent/'data/train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
reg=json.loads((H.parent/'temp_repair_20261009_v1/registration_v1.json').read_text(encoding='utf-8'))
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n3_target_v1';ans={}
for s,fds in reg['folds'].items():
    vec={k:[] for k in ['e0','es','es_raw','total','baseline']}
    drift=[]
    for k in range(len(fds)):
        z=np.load(O/(s+'_'+str(k)+'_v1.npz'),allow_pickle=True);ids=z['row_id'].tolist();truth=y.loc[ids].to_numpy();truth0=y.loc[[i[:8]+'00' for i in ids]].to_numpy()
        e0=z['level']-truth0;es=z['shape']-(truth-truth0);raw=z['shape_raw']-(truth-truth0);et=z['N3']-truth
        assert np.allclose(e0+es,et,rtol=0,atol=5e-14)
        for name,values in [('e0',e0),('es',es),('es_raw',raw),('total',et),('baseline',z['BASELINE']-truth)]:vec[name].extend(map(float,values))
    a={n:np.asarray(v) for n,v in vec.items()};mse=lambda t:math.fsum(float(v*v) for v in t)/len(t)
    m0=mse(a['e0']);ms=mse(a['es']);cov=2*math.fsum(float(x*z) for x,z in zip(a['e0'],a['es']))/len(a['e0']);mt=mse(a['total'])
    assert abs(m0+ms+cov-mt)<1e-12
    ans[s]={'level_mse':m0,'shape_mse':ms,'raw_shape_mse':mse(a['es_raw']),'cross_term':cov,'total_mse':mt,'baseline_mse':mse(a['baseline']),'rows':len(a['e0'])}
with (H/'n3_decomposition_v1.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS_NONORTHOGONAL_IDENTITY','results':ans,'query_truth_used':'post-prediction diagnostics only','new_fit':0},f,indent=2)
print(json.dumps(ans),flush=True)
