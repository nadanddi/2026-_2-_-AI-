"""K1 기존 base-v2 캐시/패키지 입력 대조. 모델 fit/predict 호출 없음."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,hashlib,importlib.util
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
MODEL=ROOT/'집/코덱스/analysis/ec_v2_state_20261001_v1/ec_model.py'
LOCAL=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1'
FINAL=LOCAL/'artifact_numpy253'
STAGE=FINAL/'stage';REPLAY=FINAL/'reproduction_check/replay'
CANON=LOCAL/'artifact/reproduction_check/replay'
LOCK=Path(env.CODEX)/'ec_final_lock/locked_days.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def compare(a,b,cols):
    assert a.row_id.tolist()==b.row_id.tolist()
    x=a[cols].to_numpy(float);y=b[cols].to_numpy(float)
    assert np.array_equal(np.isnan(x),np.isnan(y))
    mask=np.isfinite(x)&np.isfinite(y)
    return {'float64_equal':bool(np.array_equal(x,y,equal_nan=True)),
            'float64_different_finite_cells':int(np.count_nonzero((x!=y)&mask)),
            'float64_max_abs_difference':float(np.max(np.abs(x[mask]-y[mask]))),
            'float32_equal':bool(np.array_equal(x.astype(np.float32),y.astype(np.float32),equal_nan=True))}

def main():
    model=load_module('readonly_state_module_for_k1',MODEL)
    core=load_module('readonly_core_for_k1',Path(env.CODEX)/'rl_ec_v1/run.py')
    assert model.FULL==core.FULL and len(model.FULL)==38 and model.BASE==core.BASE and model.SEEDS==[7,101,2024]
    assert not set(model.STATE)&set(model.FULL)
    tr,q,ids,locks=model.load(STAGE/'data',STAGE/'locked_days.json')
    otr,oq,oids,olocks=model.load(Path(env.DATA),LOCK)
    assert locks==olocks and ids.row_id.tolist()==oids.row_id.tolist()
    assert len(tr)==7344 and len(q)==1440 and len(tr[['farm','day']].drop_duplicates())==306
    assert np.array_equal(tr.sub_ec.to_numpy(),otr.sub_ec.to_numpy())
    assert tr.row_id.tolist()==otr.row_id.tolist()
    assert not any((f,int(d)+j) in locks for f,d in zip(tr.farm,tr.day) for j in [-1,0,1])
    with np.load(REPLAY/'prediction_details.npz',allow_pickle=False) as z:
        keys=z.files;cache_ids=z['row_id'].tolist();baseline=z['baseline'].copy();candidate=z['candidate'].copy()
    assert cache_ids==ids.row_id.tolist() and len(baseline)==1440 and baseline.dtype==np.float64
    assert np.isfinite(baseline).all() and baseline.min()>=tr.sub_ec.min() and baseline.max()<=tr.sub_ec.max()
    fsum_mean=math.fsum(float(v) for v in baseline)/len(baseline)
    assert abs(fsum_mean-float(np.mean(baseline)))<1e-12
    assert min(float(v) for v in baseline)==float(np.min(baseline))
    assert max(float(v) for v in baseline)==float(np.max(baseline))
    with np.load(CANON/'prediction_details.npz',allow_pickle=False) as z:
        assert z['row_id'].tolist()==cache_ids
        canonical=z['baseline'].copy()
    assert np.array_equal(baseline,canonical)
    npz_bits_equal=bool(np.array_equal(baseline.view(np.uint64),canonical.view(np.uint64)))
    manifest=json.loads((REPLAY/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['versions']['numpy']=='2.5.3'
    assert manifest['training_rows']==7344 and manifest['training_days']==306
    assert manifest['code_sha256']==sha(MODEL)==sha(STAGE/'ec_model.py')
    inputs={n:sha(STAGE/'data'/n)==manifest['inputs_sha256'][n] for n in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']}
    assert all(inputs.values())
    assert manifest['lock_sha256']==sha(LOCK)==sha(STAGE/'locked_days.json')
    assert manifest['checkpoint_sha256']==sha(STAGE/'tabpfn-v2-regressor.ckpt')
    train_compare=compare(tr,otr,model.FULL);query_compare=compare(q,oq,model.FULL)
    cache_files=[]
    for p in LOCAL.rglob('prediction_details.npz'):
        cache_files.append({'path':str(p),'bytes':p.stat().st_size})
    result={'status':'PASS','model_fit_called':False,'model_predict_called':False,'final_lock_scored':False,
      'rows':1440,'training_rows':7344,'training_days':306,'features':model.FULL,'seeds':model.SEEDS,
      'pfn_context_seeds':[1,2,3,4],'pfn_context_rows':2000,'pfn_n_estimators_per_context':4,
      'cache_keys':keys,'baseline_numpy253_two_replays_bit_equal':npz_bits_equal,
      'baseline_mean_fsum':fsum_mean,'baseline_minimum':float(baseline.min()),'baseline_maximum':float(baseline.max()),
      'baseline_vs_state_candidate_max_abs_difference':float(np.max(np.abs(baseline-candidate))),
      'package_input_hash_match':inputs,'raw_vs_package_training_features':train_compare,
      'raw_vs_package_query_features':query_compare,'raw_vs_package_training_targets_equal':True,
      'cache_file_inventory':cache_files,'manifest_versions':manifest['versions'],
      'references':{'reference_cache':str(REPLAY/'prediction_details.npz'),'package_data':str(STAGE/'data'),
                    'canonical_numpy253_cache':str(CANON/'prediction_details.npz')},
      'hashes':{'cache':sha(REPLAY/'prediction_details.npz'),'canonical_cache':sha(CANON/'prediction_details.npz'),
                'manifest':sha(REPLAY/'manifest.json'),'model':sha(MODEL),'audit_code':sha(__file__)},
      'limitations':['기존 캐시 baseline은 상태 후보와 별개이며 학습한 모델을 다시 예측한 결과를 감사한 것은 아님',
                     '새 패키지의 정확 재현은 동일 필터 입력/환경/행순서와 새 ZIP 깨끗한 추출 검산이 필요']}
    p=HERE/'k1_base_cache_audit_v2.json';assert not p.exists()
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['features','references','hashes','cache_file_inventory','limitations']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
