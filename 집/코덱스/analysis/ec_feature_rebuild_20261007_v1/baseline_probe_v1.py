"""One CPU ET baseline reproduction audit; not a candidate experiment."""
from pathlib import Path
import ast
import hashlib
import json
import os
import platform
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from checkpoint_v1 import digest,atomic,complete,reuse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected_definitions(path, names, namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

core=Path(env.CODEX)/'rl_ec_v1/run.py'
dc4=ROOT/'집/클로드/research/ec2_DC4_exact_twin_anchor_v1.py'
dp1=ROOT/'집/클로드/research/ec3_DP1_daily_operation_pattern_v1.py'
ACTS=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
INDOOR=['in_temp','in_hum','in_co2'];RAW=INDOOR+ACTS
BASE=RAW+['day','hr_sin','hr_cos','midnight']
FP=[v+'_h0' for v in ACTS+INDOOR]+[n for v in ACTS for n in [v+'_tdm',v+'_tdz']]
FULL=BASE+FP;W=['out_temp','out_hum','out_rad','out_wspd']
NEW=['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']
ns=dict(globals())
selected_definitions(core,['identify','features','shrink','et'],ns)
selected_definitions(dc4,['weather_vectors','season_index'],ns)
selected_definitions(dp1,['run_len','day_feats'],ns)

raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+W+RAW)
raw=raw[raw.row_id.str.startswith(('F13_','F47_'))].reset_index(drop=True)
full=ns['identify'](raw)
lab=ns['features'](raw).merge(full[['row_id']+W],on='row_id',validate='one_to_one')
y=pd.read_csv(Path(env.DATA)/'train_y.csv',usecols=['row_id','sub_ec','sub_temp'])
lab=lab.merge(y[['row_id','sub_ec']],on='row_id',validate='one_to_one')
lockpath=Path(env.CODEX)/'ec_final_lock/locked_days.json'
locked={(v['farm'],int(v['day'])) for v in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
lab=lab[[(f,int(d)) not in locked for f,d in zip(lab.farm,lab.day)]].copy()
assert len(lab)==8640
lab=lab.join(pd.concat([ns['day_feats'](g) for _,g in lab.groupby(['farm','day'])]))
storedpath=ROOT/'집/클로드/research/local/ec3_WT1_all.csv'
stored=pd.read_csv(storedpath);saved=stored[(stored.validator=='DIAG10')&(stored.validation_fold==0)].copy()
vd=set(zip(saved.farm,saved.day));forbidden={(f,int(d)+j) for f,d in vd|locked for j in [-1,0,1]}
tr=lab[[(f,int(d)) not in forbidden for f,d in zip(lab.farm,lab.day)]].copy()
va=lab[lab.row_id.isin(saved.row_id)].copy()
assert va.row_id.tolist()==saved.row_id.tolist()
assert not(set(tr.row_id)&set(va.row_id))
np.testing.assert_array_equal(va.sub_ec.to_numpy(),saved.sub_ec.to_numpy())
train_days=tr[['farm','day']].drop_duplicates()
query_days=va[['farm','day']].drop_duplicates().reset_index(drop=True)
season,qseason=ns['season_index'](train_days,query_days,ns['weather_vectors'](full))
tr['season']=[season[(f,d)] for f,d in zip(tr.farm,tr.day)]
qmap={(f,d):v for (f,d),v in zip(query_days.itertuples(index=False,name=None),qseason)}
va['season']=[qmap[(f,d)] for f,d in zip(va.farm,va.day)]
cols=[c for c in FULL if c!='day']+['season']+NEW
assert len(cols)==47
environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'device':'CPU','threadpool_limit':1}
contexts={str(c):tr.row_id.iloc[np.random.default_rng(c).choice(len(tr),size=min(2000,len(tr)),replace=False)].tolist() for c in [5,6,7,8]}
fold={'validator':'DIAG10','fold':0,'train_ids':tr.row_id.tolist(),'query_ids':va.row_id.tolist(),
    'calendar_fit_days':list(map(list,train_days.itertuples(index=False,name=None))),
    'anchor_ids':tr.row_id.tolist(),'potential_PFN_context_ids_NOT_EXECUTED':contexts,
    'hidden_EC_and_temp_ids':y[y.row_id.isin(va.row_id)].row_id.tolist(),'features':cols,
    'purpose':'지정 기준선 ET 구성원 한 셀 재현. PFN/SG2/전체 기준선 재현 아님',
    'environment':environment,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [core,dc4,dp1,lockpath,storedpath]},
    'atol':1e-6,'rtol':0,'fit_query_ids_overlap':0}
p=HERE/'baseline_probe_registration_v1.json';assert not p.exists()
p.write_text(json.dumps(fold,ensure_ascii=False,indent=2),encoding='utf-8')
atomic(HERE/'baseline_probe_STATE.json',json.dumps({'state':'RUNNING','pid':os.getpid(),'command':str(Path(__file__).resolve()),'registration_sha256':sha(p)},ensure_ascii=False))
print('DIAG10/0 seed47 baseline ET fit audit starts',flush=True)
with threadpool_limits(limits=1):
    m=ns['et'](47);m.fit(tr[cols],tr.sub_ec.to_numpy());m.steps[-1][1].n_jobs=1
    pred=ns['shrink'](m.predict(va[cols]),va)
delta=float(np.max(np.abs(pred-saved.et_47.to_numpy())))
passed=delta<=1e-6
contract={'input_sha256':digest(json.loads((HERE/'source_contract_v1.json').read_text(encoding='utf-8'))['SHA256']),
    'code_sha256':sha(__file__),'environment_sha256':digest(environment),'fold_sha256':sha(p),
    'train_ids_sha256':digest(tr.row_id.tolist()),'anchor_ids_sha256':digest(tr.row_id.tolist()),
    'context_ids_sha256':digest([]),'query_ids_sha256':digest(va.row_id.tolist()),
    'baseline_sha256':sha(storedpath),'postprocess_sha256':digest('core.shrink .5 current+.5prefix, unclip ET member'),
    'seed':47,'candidate_id':'BASELINE_ET_REPRODUCTION_ONLY','validator':'DIAG10','fold':0}
receipt=complete(HERE/'checkpoints/baseline_ET_DIAG10_0_seed47',contract,va.row_id.tolist(),va.sub_ec.to_numpy(),pred)
result={'status':'PASS' if passed else 'FAIL','max_abs_difference_ET_shrunk':delta,'atol':1e-6,
    'train_rows':len(tr),'query_rows':len(va),'features':len(cols),'CPU_only':True,
    'candidate_trained':False,'entire_baseline_reproduced':False,'receipt_RMSE':receipt['RMSE'],
    'remaining':['LGB/MLP/PFN 혼합/SG2 재현 및 전체 인과성 감사','후보별 실제학습/참조/문맥/inner fold contract']}
out=HERE/'baseline_probe_result_v1.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
atomic(HERE/'baseline_probe_STATE.json',json.dumps({'state':'VERIFIED' if passed else 'FAILED','pid':os.getpid(),'result_sha256':sha(out)},ensure_ascii=False))
print(json.dumps(result,ensure_ascii=False),flush=True)
assert passed,'baseline differs, preserve receipt and stop candidate fitting'
