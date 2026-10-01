"""원실험 후반 12fold의 3멤버 예측 캐시만 사전계산한다.

고정 그룹 A=A0..4+B0, BEXT=B1..4+EXT10/12. 후보/점수 생성 없음.
원 run_ablation.py의 동일 get_members/fit_members/recipe/key를 호출한다.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import argparse,json,hashlib,importlib.util,time,platform
import numpy as np
import pandas as pd
import sklearn,lightgbm
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
RUN_PATH=HERE/'run_ablation.py'
GROUPS={'A':[('A',0),('A',1),('A',2),('A',3),('A',4),('B',0)],
        'BEXT':[('B',1),('B',2),('B',3),('B',4),('EXT10',0),('EXT12',0)]}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def import_run():
    spec=importlib.util.spec_from_file_location('ec_original_ablation_precache_readonly',RUN_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def assert_equal(got,want,name):
    if got!=want:raise RuntimeError(f'Original experiment manifest mismatch: {name}')
def manifest_check(run,common,core):
    path=HERE/'manifest.json'
    manifest=load_json(path)
    checks={'code_sha256':sha(RUN_PATH),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
            'common_sha256':sha(run.COMMON_PATH),'core_sha256':sha(core.__file__),
            'split_sha256':sha(common.SPLIT),'phase3_manifest_sha256':sha(common.CACHE/'manifest.json'),
            'lock_sha256':sha(common.LOCK),'inputs_sha256':{n:sha(Path(env.DATA)/n) for n in ['train_X.csv','train_y.csv']},
            'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,
            'seeds':run.SEEDS,'weights':run.WEIGHTS,'arms':run.ARMS,'total_comparison_family':5,
            'p_worse_limit':.005,'bootstrap_reps':20000,'threads':2,
            'features':{'et':list(core.FULL),'lgb_mlp':list(core.BASE)},'fold_count':22,'locked_days':40,
            'model_final_training':False,'test_predictions_created':False}
    assert_equal(set(manifest),set(checks),'manifest fields')
    for key,value in checks.items():assert_equal(manifest[key],value,key)
    original=load_json(common.CACHE/'manifest.json')
    assert_equal(original['inputs'],manifest['inputs_sha256'],'phase3 original inputs')
    for old,new in [('core','core_sha256'),('lock','lock_sha256'),('splits','split_sha256')]:
        assert_equal(original[old],manifest[new],f'phase3 original {old}')
    keyprefix=json.dumps(manifest,sort_keys=True,ensure_ascii=False)
    return manifest,keyprefix
def main(group):
    start=time.monotonic()
    run=import_run()
    common=run.load_common()
    data,lab,folds,locks,core=common.prepare()
    assert len(lab)==8640 and lab.row_id.is_unique and len(locks)==40
    assert not any((str(f),int(d)) in locks for f,d in zip(lab.farm,lab.day))
    manifest,keyprefix=manifest_check(run,common,core)
    wanted=GROUPS[group]
    lookup={(name,index):fold for fold in folds for name,index,_ in [fold]}
    chosen=[lookup[key] for key in wanted]
    assert len(chosen)==6 and [(name,index) for name,index,_ in chosen]==wanted
    entries=[]
    def status(stage,**detail):
        payload={'stage':stage,'group':group,'worker_code_sha256':sha(__file__),
                 'original_code_sha256':manifest['code_sha256'],'original_manifest_sha256':sha(HERE/'manifest.json'),
                 'original_hypothesis_or_recipe_changed':False,'candidate_scores_created':False,
                 'elapsed_seconds':time.monotonic()-start,'entries':entries,**detail}
        tmp=HERE/f'precache_{group}_index.tmp'
        tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(tmp,HERE/f'precache_{group}_index.json')
        message=json.dumps({k:v for k,v in payload.items() if k!='entries'},ensure_ascii=False)
        print(message,flush=True)
        with (HERE/f'precache_{group}.log').open('a',encoding='utf-8') as f:f.write(message+'\n')
    status('READY',fixed_fold_order=wanted)
    for fold in chosen:
        name,index,_=fold
        tr,va=common.split_fold(data,lab,fold,locks)
        for seed in run.SEEDS:
            path=run.OUT/f'{name}_{index}_seed{seed}_members.npz'
            meta=path.with_suffix('.json')
            # Original and worker should not calculate the same target concurrently.
            # Root supervises active_target against the unchanged original runner.
            status('FITTING_OR_REUSING',active_target=f'{name}/{index}/seed{seed}',
                   target_path=str(path),already_exists=path.exists())
            predictions=run.get_members(core,tr,va,name,index,seed,keyprefix)
            assert set(predictions)=={'et','lgb','mlp'} and all(len(p)==len(va) and np.isfinite(p).all() for p in predictions.values())
            assert path.exists() and meta.exists()
            expected_key=keyprefix+run.dataset_hash(tr,core.FULL)+run.dataset_hash(va,core.FULL)
            metadata=load_json(meta)
            assert_equal(metadata['key'],expected_key,'prediction cache key')
            assert_equal(metadata['npz_sha256'],sha(path),'prediction cache SHA256')
            with np.load(path) as z:assert_equal(z['row_id'].tolist(),va.row_id.tolist(),'prediction row order')
            entries.append({'validator':name,'validation_fold':index,'seed':seed,'train_n':len(tr),'val_n':len(va),
                            'cache_path':str(path),'cache_sha256':sha(path),'metadata_sha256':sha(meta)})
            status('CACHED',completed_target=f'{name}/{index}/seed{seed}',completed_fits=len(entries))
    assert len(entries)==18
    status('COMPLETE',completed_fits=len(entries),completed_folds=len(chosen))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=sorted(GROUPS),required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.group)
