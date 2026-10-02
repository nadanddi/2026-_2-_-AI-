"""순차 역변환과 다른 24x24 선형방정식 해 및 실제 core.shrink로 검산."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,importlib.util
from collections import defaultdict
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
with (HERE/'recovered_members.csv').open(encoding='utf-8-sig',newline='') as f:
    rec={(r['validator'],int(r['validation_fold']),r['row_id']):r for r in csv.DictReader(f)}
spec=importlib.util.spec_from_file_location('readonly_original_core',Path(env.CODEX)/'rl_ec_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
S=np.zeros((24,24))
for h in range(24):S[h,:h+1]=.5/(h+1);S[h,h]+=.5
assert np.linalg.matrix_rank(S)==24
max_matrix_error=0.;max_core_error=0.;max_numpy_score_error=0.;n=0
errors=defaultdict(lambda:defaultdict(list))
score_record=json.loads((HERE/'cache_audit.json').read_text(encoding='utf-8'))['score_by_validator']
for path in CACHE.glob('*.npz'):
    name,fold=path.stem.rsplit('_',1);fold=int(fold)
    with np.load(path) as z:
        ids=z['row_id'].tolist();bag=5*z['v2']-4*z['r3'];r3=z['r3'].copy();v2=z['v2'].copy()
    frame=pd.DataFrame({'farm':[r.split('_')[0] for r in ids],'day':[int(r.split('_')[1]) for r in ids],
                        'hour':[int(r.split('_')[2]) for r in ids]})
    restored=np.array([float(rec[(name,fold,r)]['tabpfn_raw_bag']) for r in ids])
    rr=np.array([float(rec[(name,fold,r)]['r3_raw_mean']) for r in ids])
    for ix in frame.groupby(['farm','day']).indices.values():
        ix=ix[np.argsort(frame.loc[ix,'hour'].to_numpy())]
        solved=np.linalg.solve(S,bag[ix]);max_matrix_error=max(max_matrix_error,float(np.max(np.abs(solved-restored[ix]))))
    max_core_error=max(max_core_error,float(np.max(np.abs(core.shrink(restored,frame)-bag))),
                       float(np.max(np.abs(core.shrink(.8*rr+.2*restored,frame)-v2))))
    n+=len(ids)
    errors[name]['r3'].extend(r3.tolist());errors[name]['v2'].extend(v2.tolist())
# Independent fresh NumPy score calculation with OOF labels (locked rows absent in this saved table).
raw=pd.read_csv(CACHE/'oof_predictions.csv')
for name,g in raw.groupby('validator'):
    for k in ['r3','v2']:
        score=float(np.sqrt(np.mean((g[k].to_numpy()-g.sub_ec.to_numpy())**2)))
        max_numpy_score_error=max(max_numpy_score_error,abs(score-score_record[name][k]))
assert max_matrix_error<1e-12 and max_core_error<1e-12 and max_numpy_score_error<1e-12
result={'status':'PASS','cache_occurrences':n,'matrix_rank':int(np.linalg.matrix_rank(S)),
        'matrix_condition_number_2norm':float(np.linalg.cond(S)),
        'matrix_vs_sequential_inverse_max_abs_difference':max_matrix_error,
        'actual_core_shrink_max_abs_difference':max_core_error,
        'numpy_vs_fsum_r3_v2_rmse_max_abs_difference':max_numpy_score_error,
        'model_fit_called':False,'model_predict_called':False,'final_lock_scored':False}
(HERE/'inverse_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
