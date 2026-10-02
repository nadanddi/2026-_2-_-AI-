"""기존 phase3 캐시와 선형 후처리 역변환 감사. fit/predict 모델 호출 없음."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import env_extra
import csv,json,math,hashlib,platform
from collections import defaultdict
import numpy as np
import sklearn,lightgbm,torch,tabpfn,pandas

HERE=Path(__file__).resolve().parent
CACHE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
CORE=Path(env.CODEX)/'rl_ec_v1/run.py'
LOCK=Path(env.CODEX)/'ec_final_lock/locked_days.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def meta(rid):
    f,d,h=rid.split('_');return f,int(d),int(h)
def mean(a):return math.fsum(a)/len(a)
def rmse(a):return math.sqrt(math.fsum(z*z for z in a)/len(a))
def grouped(ids):
    groups=defaultdict(list)
    for i,rid in enumerate(ids):
        f,d,h=meta(rid);groups[(f,d)].append((h,i))
    return {k:[i for h,i in sorted(v)] for k,v in groups.items()}
def shrink(p,ids):
    out=np.zeros(len(p))
    for inds in grouped(ids).values():
        previous=[]
        for i in inds:
            previous.append(float(p[i]));out[i]=.5*float(p[i])+.5*mean(previous)
    return out
def inverse(q,ids):
    out=np.zeros(len(q))
    for inds in grouped(ids).values():
        previous=[]
        for n,i in enumerate(inds,1):
            out[i]=(2*n*float(q[i])-math.fsum(previous))/(n+1);previous.append(float(out[i]))
    return out

def main():
    lock={(r['farm'],int(r['day'])) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    raw={};skip=0
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,h=meta(r['row_id'])
        if f not in ['F13','F47']:continue
        if (f,d) in lock:skip+=1;continue
        raw[r['row_id']]=float(r['sub_ec'])
    assert len(raw)==8640 and skip==960
    manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
    current={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
             'lightgbm':lightgbm.__version__,'torch':torch.__version__,'tabpfn':tabpfn.__version__,
             'pandas':pandas.__version__,'executable':sys.executable,
             'module_paths':{m.__name__:m.__file__ for m in [np,sklearn,lightgbm,torch,tabpfn,pandas]}}
    recorded={k:manifest[k] for k in ['python','numpy','sklearn','lightgbm','torch','tabpfn']}
    version_match={k:current[k]==v for k,v in recorded.items()}
    files=[];recoveries=[];totals=defaultdict(int);max_mixture_diff=0.;max_forward_diff=0.;max_seedbag_diff=0.
    oof={(r['validator'],int(r['validation_fold']),r['row_id']):r for r in read(CACHE/'oof_predictions.csv')}
    baseline=defaultdict(lambda:defaultdict(list))
    paths=sorted(CACHE.glob('*.npz'));assert len(paths)==22
    for path in paths:
        name,fold=path.stem.rsplit('_',1);fold=int(fold)
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        ids=arrays.pop('row_id').tolist();validation={(meta(rid)[0],meta(rid)[1]) for rid in ids}
        forbidden={(f,d+j) for f,d in validation|lock for j in [-1,0,1]}
        tr=[v for rid,v in raw.items() if (meta(rid)[0],meta(rid)[1]) not in forbidden]
        lo,hi=min(tr),max(tr)
        side=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert side['prediction_sha256']==sha(path)
        groups=grouped(ids);assert all(len(ix)==24 for ix in groups.values())
        boundary={}
        for k in ['r3','v2']+[f'{m}_{s}' for m in ['r3','v2'] for s in [7,101,2024]]:
            p=arrays[k];assert np.isfinite(p).all() and p.min()>=lo and p.max()<=hi
            low=int(np.sum(p==lo));high=int(np.sum(p==hi));near=int(np.sum((p-lo<=1e-12)|(hi-p<=1e-12)))
            boundary[k]={'lower_equal':low,'upper_equal':high,'within_1e12_boundary':near,
                         'prediction_min':float(p.min()),'prediction_max':float(p.max())}
            totals[k+'_boundary_equal']+=low+high
            for rid,value in zip(ids,p):
                saved=float(oof[(name,fold,rid)][k]);assert abs(saved-float(value))<=1e-12
                baseline[name][k].append(float(value)-raw[rid])
        clipfree=all(v['within_1e12_boundary']==0 for v in boundary.values())
        bag_post=5*arrays['v2']-4*arrays['r3']
        seedbags=[5*arrays[f'v2_{s}']-4*arrays[f'r3_{s}'] for s in [7,101,2024]]
        spread=max(float(np.max(np.abs(v-bag_post))) for v in seedbags)
        max_seedbag_diff=max(max_seedbag_diff,spread)
        assert clipfree, f'{name}/{fold}: boundary prevents exact identification'
        assert spread<1e-12
        bag_raw=inverse(bag_post,ids);roundtrip=shrink(bag_raw,ids)
        diff=float(np.max(np.abs(roundtrip-bag_post)));max_forward_diff=max(max_forward_diff,diff)
        assert diff<1e-12
        r3_raw=inverse(arrays['r3'],ids)
        mix=np.clip(shrink(.8*r3_raw+.2*bag_raw,ids),lo,hi)
        md=float(np.max(np.abs(mix-arrays['v2'])));max_mixture_diff=max(max_mixture_diff,md);assert md<1e-12
        for rid,bp,br,rr in zip(ids,bag_post,bag_raw,r3_raw):
            recoveries.append({'validator':name,'validation_fold':fold,'row_id':rid,
                               'tabpfn_post_shrink':float(bp),'tabpfn_raw_bag':float(br),'r3_raw_mean':float(rr)})
        files.append({'validator':name,'fold':fold,'rows':len(ids),'days':len(groups),'training_rows':len(tr),
                      'clip_lower':lo,'clip_upper':hi,'all_r3_v2_cache_interiors':clipfree,'boundaries':boundary,
                      'seed_recovered_bag_max_abs_difference':spread,'inverse_forward_max_abs_difference':diff,
                      'mixture_recreation_max_abs_difference':md,'cache_sha256':sha(path)})
    with (HERE/'recovered_members.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(recoveries[0]));w.writeheader();w.writerows(recoveries)
    checkpoint=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
    hashes={'core':sha(CORE),'code':sha(ROOT/'집/코덱스/analysis/ec_restart_phase3_20261001_v1/run_benchmark.py'),
            'lock':sha(LOCK),'checkpoint':sha(checkpoint),'splits':sha(ROOT/'집/코덱스/analysis/local/rl_ec_v1/20260927_173801/splits.csv')}
    hashmatches={k:v==manifest[k] for k,v in hashes.items()}
    report={'status':'PASS','training_executed':False,'prediction_model_called':False,'locked_rows_skipped_before_float':skip,
            'current_environment':current,'recorded_environment':recorded,'environment_version_match':version_match,
            'recipe_hash_match':hashmatches,'cache_files':len(files),'cache_occurrences':len(recoveries),
            'boundary_totals':dict(totals),'max_seed_bag_difference':max_seedbag_diff,
            'max_inverse_forward_difference':max_forward_diff,'max_mixture_recreation_difference':max_mixture_diff,
            'score_by_validator':{v:{k:rmse(er) for k,er in p.items()} for v,p in baseline.items()},
            'folds':files,'hashes':hashes,'independent_audit_code_sha256':sha(__file__),
            'limitations':['후처리 평균 bag만 식별, 4개 TabPFN 문맥별 예측·모델상태는 복원불가',
                           '학습폴드 예측 없음: OOF bag를 학습행 target/feature로 잘못 재사용하면 누수',
                           '원코드 fit이나 모델 predict를 재실행하지 않았으므로 전체 재학습 결정성은 미확인']}
    (HERE/'cache_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['folds','hashes']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
