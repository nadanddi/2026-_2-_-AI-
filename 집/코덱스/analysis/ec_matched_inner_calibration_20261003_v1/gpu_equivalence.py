from pathlib import Path
import sys,os,json,gc,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'));import env_extra_gpu
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1';CKPT=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
assert torch.cuda.is_available();torch.set_num_threads(4)
lab,core,wv,folds,outer=S.loadec();z=dict(np.load(OUT/'DIAG10_0_cpu.npz'));a=lab.set_index('row_id').reindex(z['inner_train_id']).reset_index();b=lab.set_index('row_id').reindex(z['row_id']).reset_index();a,b=S.seasonal(a,b,wv);cols=[c for c in core.FULL if c!='day']+['season'];ix=np.random.default_rng(1).choice(len(a),min(2000,len(a)),replace=False)
pp=[]
for n in [1,2]:
 m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',model_path=str(CKPT),n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
 with threadpool_limits(limits=4):
  m.fit(a[cols].to_numpy(np.float32)[ix],a.sub_ec.to_numpy(float)[ix]);p=np.asarray(m.predict(b[cols].to_numpy(np.float32)),float)
 assert np.isfinite(p).all();pp.append(p);del m;gc.collect();torch.cuda.empty_cache();print('GPU full fit',n,'done',flush=True)
assert np.max(np.abs(pp[0]-pp[1]))<1e-5
np.savez(OUT/'gpu_equivalence_first.npz',row_id=b.row_id.to_numpy(str),context_row_id=a.row_id.iloc[ix].to_numpy(str),prediction=pp[0])
(H/'gpu_reproduction_v1.json').write_text(json.dumps(dict(status='PASS',maxdiff=float(np.max(np.abs(pp[0]-pp[1]))),device=torch.cuda.get_device_name(0),warning='CPU numerical equivalence not established until CPU checkpoint finishes'),ensure_ascii=False,indent=2),encoding='utf-8')
