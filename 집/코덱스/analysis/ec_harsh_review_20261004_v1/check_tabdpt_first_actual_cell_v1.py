"""First completed actual TabDPT cell: identity/causal postprocess audit only.

No partial RMSE, score, adoption, bootstrap, fit or prediction calls.
Read exactly DIAG10_0_7 candidate CSV/meta/first audit and old public sources.
"""
import ast
import hashlib
import json
import math
import platform
import sys
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path('C:/work/farmai')
HERE=ROOT/'집/코덱스/analysis/ec_harsh_review_20261004_v1'
H=ROOT/'집/코덱스/analysis/ec_tabdpt_20261004_v1'
PREP=ROOT/'집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1'
OUT=ROOT/'집/코덱스/local/ec_tabdpt_20261004_v1'
BASE=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
OLD=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
RESULT=HERE/'tabdpt_first_actual_cell_check_v1.json'
assert not RESULT.exists() and __debug__
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np
import pandas as pd
import sklearn
import lightgbm


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))

expected_run='136f72ce14cf7afc078185f0e1c421052910153d6f156ebe8eed604cc3a6d2f2'
expected_verify='e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9'
assert sha(H/'run_v4.py')==expected_run and sha(H/'verify_full_v2.py')==expected_verify
tree=ast.parse((H/'verify_full_v2.py').read_text(encoding='utf-8-sig'))
fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
           if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
           and n.targets[0].id in ['EXPECTED_DEPS','CORE_RUNTIME','PIN5','SOURCE_COMMIT','WEIGHT_SHA','CSV_COLUMNS']}
ns=dict(np=np,pd=pd,math=math,hashlib=hashlib,json=json,Path=Path,CHECKS=0,
        RAW_ATOL=1e-6,FINAL_ATOL=2e-7,PREP=PREP,SITE=ROOT/'집/코덱스/local/tabdpt130_cpu_v1/site',**constants)
names=['sha','readj','array_sha','ids_sha','compare','prefix_scalar','validate_first','check_runtime']
exec(compile(ast.Module(body=[fs[name] for name in names],type_ignores=[]),'<actual-verifier-audit-only-helpers>','exec'),ns)
compare=ns['compare']

first_path=H/'first_fold_verification_v1.json'
meta_path=OUT/'DIAG10_0_7.json'
csv_path=OUT/'DIAG10_0_7_pred.csv'
first,meta,prepared=readj(first_path),readj(meta_path),readj(H/'preparation_v4.json')
assert meta['status']=='PASS' and first['status']=='PASS'
assert meta['first_audit_sha256']==sha(first_path) and meta['csv_sha256']==sha(csv_path)
assert math.isfinite(meta['fit_predict_seconds']) and meta['fit_predict_seconds']>=0
assert math.isfinite(first['audit_seconds']) and first['audit_seconds']>=0

lab,core,wv,folds,outer=S.loadec()  # train_X + already public OOF y only
selected=[(v,k,tm,vm) for v,k,tm,vm in folds if (v,k)==('DIAG10',0)]
assert len(selected)==1
v,k,tm,vm=selected[0]
tr,q=S.seasonal(lab[tm],lab[vm],wv)
tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
cols=[c for c in core.FULL if c!='day']+['season']
assert len(cols)==38 and cols[-1]=='season' and 'day' not in cols
assert tr.row_id.is_unique and q.row_id.is_unique and not set(tr.row_id)&set(q.row_id)
train_days=set(zip(tr.farm,tr.day));query_days=set(zip(q.farm,q.day))
assert all(not(f==ff and abs(int(day)-int(qday))<=1) for f,day in train_days for ff,qday in query_days)
runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
             sklearn=sklearn.__version__,lightgbm=lightgbm.__version__)
assert runtime==constants['CORE_RUNTIME']
dependencies=dict(core=sha(Path(core.__file__)),season=sha(Path(sys.modules[S.mapping.__module__].__file__)),
                  env=sha(Path(env.__file__)),support=sha(Path(S.__file__)),
                  adapter=sha(PREP/'adapter_draft_v1.py'),runtime_probe=sha(PREP/'runtime_probe_v3.py'),run=sha(H/'run_v4.py'))
assert dependencies==constants['EXPECTED_DEPS']
input_sha=sha(Path(env.DATA)/'train_X.csv');public_sha=sha(OLD/'v2_integration_oof.csv')
assert prepared['status']=='PASS' and prepared['fit_count']==0
assert prepared['input_sha256']==input_sha and prepared['public_cache_sha256']==public_sha
old=dict(np.load(BASE/'DIAG10_0_baseline.npz',allow_pickle=False))
assert np.array_equal(old['row_id'],q.row_id)
compare([old['lo'],old['hi']],[tr.sub_ec.min(),tr.sub_ec.max()])
sig=dict(validator=v,fold=k,runtime=runtime,features=cols,dependencies=dependencies,
         input_sha256=input_sha,public_cache_sha256=public_sha,
         train_ids=ns['ids_sha'](tr.row_id),query_ids=ns['ids_sha'](q.row_id),
         train_features=ns['array_sha'](tr[cols].to_numpy(float)),query_features=ns['array_sha'](q[cols].to_numpy(float)),
         train_targets=ns['array_sha'](tr.sub_ec.to_numpy(float)),query_targets=ns['array_sha'](q.sub_ec.to_numpy(float)),
         bounds=[float(old['lo']),float(old['hi'])],baseline_sha256=sha(BASE/'DIAG10_0_baseline.npz'))
manifest=[x for x in prepared['manifest'] if (x['validator'],x['fold'])==(v,k)]
assert len(manifest)==1 and manifest[0]==sig
provenance=first['signature']['provenance']
ns['check_runtime'](provenance,dependencies,input_sha)  # metadata/reference + seven source hashes; no model import
signature=dict(**sig,seed=7,provenance=provenance)
assert first['signature']==meta['signature']==signature
ns['validate_first'](first,signature,tr,q)

d=pd.read_csv(csv_path,float_precision='round_trip')
assert list(d.columns)==constants['CSV_COLUMNS']
assert np.array_equal(d.row_id,q.row_id) and d.row_id.is_unique
assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==7).all()
for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
public=outer[(outer.validator=='DIAG10')&(outer.seed==7)].set_index('row_id')
assert public.index.is_unique
compare(d.y,q.sub_ec);compare(d.y,public.sub_ec.reindex(d.row_id))
actual=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==7)].set_index('row_id').season_v2.reindex(q.row_id)
compare(d.baseline,actual);compare(d.baseline,old['baseline_7'])
compare(d.r3_raw,old['r3_7']);compare(d.old_pfn_raw,old['old_pfn_raw'])
compare(d.clip_lo,np.repeat(old['lo'],len(q)));compare(d.clip_hi,np.repeat(old['hi'],len(q)))
for column in ['new_tabdpt_raw','r3_raw','old_pfn_raw','candidate','baseline','y']:
    assert np.isfinite(d[column]).all()
lo,hi=float(old['lo']),float(old['hi'])
candidate_vector=np.clip(core.shrink(.8*d.r3_raw.to_numpy()+.2*d.new_tabdpt_raw.to_numpy(),q),lo,hi)
baseline_vector=np.clip(core.shrink(.8*d.r3_raw.to_numpy()+.2*d.old_pfn_raw.to_numpy(),q),lo,hi)
vector_gap=compare(d.candidate,candidate_vector);compare(d.baseline,baseline_vector)
candidate_scalar=ns['prefix_scalar'](q,d.r3_raw.to_numpy(),d.new_tabdpt_raw.to_numpy(),lo,hi)
baseline_scalar=ns['prefix_scalar'](q,d.r3_raw.to_numpy(),d.old_pfn_raw.to_numpy(),lo,hi)
scalar_gap=compare(d.candidate,candidate_scalar);compare(d.baseline,baseline_scalar)

# Independently recompute exact decimal input mixture/prefix averages, without core or prefix_scalar.
decimal_candidate=np.empty(len(d));decimal_baseline=np.empty(len(d))
for _,g in d.groupby(['farm','day'],sort=False):
    g=g.sort_values('hour');new_history=[];old_history=[]
    for row in g.itertuples():
        r=Decimal.from_float(float(row.r3_raw))
        newraw=Decimal('.8')*r+Decimal('.2')*Decimal.from_float(float(row.new_tabdpt_raw))
        oldraw=Decimal('.8')*r+Decimal('.2')*Decimal.from_float(float(row.old_pfn_raw))
        new_history.append(newraw);old_history.append(oldraw)
        n=Decimal(len(new_history))
        newvalue=(newraw+sum(new_history,Decimal(0))/n)/2
        oldvalue=(oldraw+sum(old_history,Decimal(0))/n)/2
        low,high=Decimal.from_float(lo),Decimal.from_float(hi)
        decimal_candidate[row.Index]=float(min(high,max(low,newvalue)))
        decimal_baseline[row.Index]=float(min(high,max(low,oldvalue)))
decimal_gap=compare(d.candidate,decimal_candidate);compare(d.baseline,decimal_baseline)

prefix_records=[]
for farm in ['F13','F47']:
    day=int(q.loc[q.farm==farm,'day'].min())
    for hour in [0,6,12]:
        keep=(q.farm==farm)&(q.day==day)&(q.hour<=hour)
        p=ns['prefix_scalar'](q[keep],d.loc[keep,'r3_raw'].to_numpy(),d.loc[keep,'new_tabdpt_raw'].to_numpy(),lo,hi)
        gap=compare(p,d.loc[keep,'candidate'])
        prefix_records.append(dict(farm=farm,day=day,hour=hour,rows=int(keep.sum()),postprocess_maxdiff=gap))
raw_errors=first['raw_errors']
assert raw_errors['repeat']==raw_errors['reversed']==raw_errors['other_query']==raw_errors['fresh_fit']==0
assert raw_errors['single']==5.960464477539063e-08
raw_max=max(x['raw_maxdiff'] for x in first['prefix'])
final_max=max(x['final_maxdiff'] for x in first['prefix'])
assert raw_max==5.960464477539063e-08 and final_max==1.1920928966180355e-08
assert raw_max<=1e-6 and final_max<=2e-7
assert max(x['feature_maxdiff'] for x in first['feature_causal'])==0

result=dict(status='PASS_FIRST_CELL_INTEGRITY_ONLY',validator=v,fold=k,seed=7,
            source_sha256=expected_run,verifier_sha256=expected_verify,
            first_audit_sha256=sha(first_path),metadata_sha256=sha(meta_path),csv_sha256=sha(csv_path),
            train_rows=len(tr),query_rows=len(q),train_days=len(train_days),query_days=len(query_days),
            fresh_signature_equal=True,fresh_full38_feature_hashes_equal=True,public_label_id_baseline_equal=True,
            source_and_core_runtime_equal=True,stored_overlay_runtime_and_seven_source_hashes_equal=True,
            first_v2_day_validation='PASS',raw_errors=raw_errors,saved_prefix_raw_max=raw_max,
            saved_prefix_final_max=final_max,saved_scalar_maxdiff=first['scalar_maxdiff'],
            vector_candidate_maxdiff=vector_gap,independent_scalar_maxdiff=scalar_gap,
            independent_decimal_maxdiff=decimal_gap,postprocess_prefix=prefix_records,
            numerical_comparison_elements=ns['CHECKS'],fit_predict_seconds=meta['fit_predict_seconds'],
            saved_audit_seconds=first['audit_seconds'],new_fit=0,new_predict=0,partial_rmse_or_score_calculations=0,
            bootstrap_or_adoption_calculations=0,full_experiment_verification=0,
            limitations=['First completed seed/fold only; all 66 cells and model performance remain unverified.',
                         'Saved batch/single/order/other/fresh/prefix raw errors were SHA/signature/day/tolerance checked, not recomputed by fresh model prediction.',
                         'Fresh FULL38 hashes reuse fixed core/season; saved 37-core causality metrics are not independent all-fold season causality proof.',
                         'Runtime metadata and seven installed source hashes checked without reimporting TabDPT/torch or rereading weights.',
                         'Existing public baseline and labels are consistency evidence, not independent untouched holdout validation.'])
with RESULT.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
