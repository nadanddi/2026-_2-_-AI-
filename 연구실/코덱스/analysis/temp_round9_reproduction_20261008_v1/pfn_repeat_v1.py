"""One-context diagnostic repeat; no source edits and no reference fitting."""
from pathlib import Path
import sys,os,json,runpy,time,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
P=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'));CODE=Path(P['temp_code'])
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
os.environ['TABPFN_MODEL_CACHE_DIR']=P['cache'];os.environ['AGRI_DATA']=str(CODE.parent/'data')
sys.path.insert(0,str(ROOT/'.analysis-tools/python'));sys.path.append(str(ROOT/'.analysis-tools/extra'))
handles=[os.add_dll_directory(str(d)) for d in [ROOT/'.analysis-tools/msvc',ROOT/'.analysis-tools/extra/torch/lib',Path(P['package'])/'EC/dlls'] if d.exists()]
sys.path.insert(0,str(CODE));os.chdir(CODE);import env
sys.argv=[str(CODE/'make_submission_v13_temp.py')]
V=runpy.run_path(sys.argv[0],run_name='readonly_original_temp_functions')
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_info
start=time.monotonic();lab,test,ct,phc,sX=V['build_frames']();w=V['TF'].row_weights(lab,V['W_FLAG'],w_noisy=V['W_NOISY'])
ctr,cte=V['codex_frame'](lab),V['codex_frame'](test)
Xtr=ctr[list(V['FEATURE_COLUMNS'])].to_numpy().astype(np.float32);Xte=cte[list(V['FEATURE_COLUMNS'])].to_numpy().astype(np.float32);y=lab.sub_temp.to_numpy()
idx=np.random.default_rng(1).choice(len(Xtr),size=min(V['N_CTX'],len(Xtr)),replace=False,p=w/w.sum())
with np.load(Path(P['output'])/'temp_candidate_v13_members.npz',allow_pickle=False) as z:previous=z['pfn_samples'][0].copy()
print('REPEAT_CONTEXT1_READY',len(idx),len(Xte),'no BASE/CODEX rerun',flush=True)
m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',n_estimators=4,random_state=1,ignore_pretraining_limits=True)
m.fit(Xtr[idx],y[idx]);prediction=np.asarray(m.predict(Xte),float)
delta=prediction-previous
result={'status':'PASS_IDENTICAL_FIRST_PFN_CONTEXT_REPEAT' if np.array_equal(prediction,previous) else 'PFN_CONTEXT_REPEAT_DIFFERS',
    'context_seed':1,'full_query_rows':1440,'changed_raw_prediction_rows':int((delta!=0).sum()),'max_abs_raw_difference':float(abs(delta).max()),'rms_raw_difference':float(np.sqrt(np.mean(delta**2))),
    'seconds':time.monotonic()-start,'predictive_context_fits':1,'source_changed':False,'reference_temperature_used_for_fit':False,
    'context_ids_sha':hashlib.sha256('\n'.join(lab.row_id.iloc[idx]).encode()).hexdigest(),'features_sha':{'Xtrain':hashlib.sha256(Xtr.tobytes()).hexdigest(),'Xtest':hashlib.sha256(Xte.tobytes()).hexdigest()},
    'threadpools':threadpool_info(),'limitation':'Within-current-environment first context only, not all members or original submission environment'}
np.savez(H/'repeat_context1_v1.npz',prediction=prediction,previous=previous,context_index=idx)
with (H/'repeat_context1_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
