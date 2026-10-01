from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
os.environ['PYTHONPATH']=''
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,hashlib,importlib.util,time,platform,traceback
import numpy as np
import pandas as pd
import sklearn,lightgbm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'
COMMON_PATH=ROOT/'집/코덱스/analysis/ec_model_common_20261002_v1/common.py'
SEEDS=[7,101,2024]
WEIGHTS={'et':.48,'lgb':.24,'mlp':.08,'pfn':.20}
ARMS=['drop_et','drop_lgb','drop_mlp','drop_pfn']
R3_TOL=1e-6
PFN_TOL=1e-9

class GateFailure(RuntimeError):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def log(s):
    print(s,flush=True)
    with (HERE/'progress.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
def final(core,tr,va,p):return np.clip(core.shrink(np.asarray(p,float),va),tr.sub_ec.min(),tr.sub_ec.max())
def maximum_difference(a,b):return float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float))))
def check(ok,msg):
    if not ok:raise GateFailure(msg)
def load_common():
    spec=importlib.util.spec_from_file_location('ec_ablation_common_readonly',COMMON_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def fit_members(core,tr,va,seed):
    et=core.et(seed);et.set_params(extratreesregressor__n_jobs=2)
    lg=core.lg(seed,'tweedie');lg.set_params(n_jobs=2)
    ml=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
        core.MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-2,learning_rate_init=1e-3,
            max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))
    preds={}
    for name,model,columns in [('et',et,core.FULL),('lgb',lg,core.BASE),('mlp',ml,core.BASE)]:
        model.fit(tr[columns],tr.sub_ec.to_numpy(float))
        preds[name]=np.asarray(model.predict(va[columns]),float)
        check(np.isfinite(preds[name]).all(),f'{name} nonfinite')
    return preds
def dataset_hash(frame,columns):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[['row_id','sub_ec']+columns],index=False).values.tobytes()).hexdigest()
def get_members(core,tr,va,name,index,seed,keyprefix):
    key=keyprefix+dataset_hash(tr,core.FULL)+dataset_hash(va,core.FULL)
    path=OUT/f'{name}_{index}_seed{seed}_members.npz';meta=path.with_suffix('.json')
    if path.exists():
        check(meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['key']==key,'Member cache hash mismatch')
        with np.load(path) as z:
            check(z['row_id'].tolist()==va.row_id.tolist(),'Member cache row order mismatch')
            return {n:z[n] for n in ['et','lgb','mlp']}
    preds=fit_members(core,tr,va,seed)
    np.savez(path,row_id=va.row_id.to_numpy(str),**preds)
    save(meta,{'key':key,'npz_sha256':sha(path),'validator':name,'fold':index,'seed':seed})
    return preds
def clipping_stats(q,lo,hi):
    q=np.asarray(q,float)
    return {'below_train_range':int((q<lo).sum()),'above_train_range':int((q>hi).sum()),
            'minimum':float(q.min()),'maximum':float(q.max())}

def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    common=load_common()
    data,lab,folds,locks,core=common.prepare()
    check(len(folds)==22,'Expected phase3 22 folds')
    check(len(lab)==8640 and lab.row_id.is_unique,'Expected unlocked360day/8640row data')
    check(not any((str(f),int(d)) in locks for f,d in zip(lab.farm,lab.day)),'Locked target present')
    corepath=Path(core.__file__)
    checks=common.feature_checks(data)
    save(HERE/'feature_checks.json',checks)
    phase_manifest_path=common.CACHE/'manifest.json'
    phase_manifest=json.loads(phase_manifest_path.read_text(encoding='utf-8'))
    for filename in ['train_X.csv','train_y.csv']:
        check(phase_manifest['inputs'][filename]==sha(Path(env.DATA)/filename),'Original phase3 input hash changed')
    check(phase_manifest['core']==sha(corepath),'Original phase3 core hash changed')
    check(phase_manifest['lock']==sha(common.LOCK),'Original phase3 lock hash changed')
    check(phase_manifest['splits']==sha(common.SPLIT),'Original phase3 split hash changed')
    manifest={'code_sha256':sha(__file__),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
              'common_sha256':sha(COMMON_PATH),'core_sha256':sha(corepath),
              'split_sha256':sha(common.SPLIT),'phase3_manifest_sha256':sha(phase_manifest_path),
              'inputs_sha256':{name:sha(Path(env.DATA)/name) for name in ['train_X.csv','train_y.csv']},
              'lock_sha256':sha(Path(env.CODEX)/'ec_final_lock/locked_days.json'),
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,
              'seeds':SEEDS,'weights':WEIGHTS,'arms':ARMS,'total_comparison_family':5,
              'p_worse_limit':.005,'bootstrap_reps':20000,'threads':2,
              'features':{'et':list(core.FULL),'lgb_mlp':list(core.BASE)},
              'fold_count':len(folds),'locked_days':len(locks),'model_final_training':False,'test_predictions_created':False}
    manifestpath=HERE/'manifest.json'
    if manifestpath.exists():check(json.loads(manifestpath.read_text(encoding='utf-8'))==manifest,'Manifest differs on resume')
    else:save(manifestpath,manifest)
    keyprefix=json.dumps(manifest,sort_keys=True,ensure_ascii=False)
    output=[];audits=[];foldrecords=[]
    for fold in folds:
        name,index,_=fold
        tr,va=common.split_fold(data,lab,fold,locks)
        check(not va.empty and not tr.empty,'Empty fold')
        check(set(tr.row_id).isdisjoint(set(va.row_id)),'Train-validation ID overlap')
        lo=float(tr.sub_ec.min());hi=float(tr.sub_ec.max())
        pfn_reference=None
        log(f'START {name}/{index} train={len(tr)} val={len(va)} elapsed={time.monotonic()-started:.1f}s')
        for seed in SEEDS:
            preds=get_members(core,tr,va,name,index,seed,keyprefix)
            r3raw=.6*preds['et']+.3*preds['lgb']+.1*preds['mlp']
            r3refit=final(core,tr,va,r3raw)
            cached=common.baseline(va,name,index,seed)
            cached_r3=np.asarray(cached['r3'],float);cached_v2=np.asarray(cached['v2'],float)
            gap=maximum_difference(r3refit,cached_r3)
            audit={'validator':name,'validation_fold':index,'seed':seed,'n':len(va),'train_n':len(tr),
                   'train_label_min':lo,'train_label_max':hi,'refit_finished_r3_max_difference':gap,
                   'cache_hash':cached['cache_hash'],'cache_pfn_safe':bool(cached['cache_pfn_safe']),
                   'clip_r3_count':int(cached['clip_r3_count']),'clip_v2_count':int(cached['clip_v2_count'])}
            audits.append(audit);save(HERE/'reconstruction_audit.json',audits)
            check(gap<=R3_TOL,f'R3_REPRODUCTION_FAILED {name}/{index}/seed{seed}: {gap} > {R3_TOL}; cached PFN forbidden')
            check(cached['cache_pfn_safe'] and cached['clip_r3_count']==0 and cached['clip_v2_count']==0,
                  f'PFN_CACHE_CLIPPED {name}/{index}/seed{seed}: linear reconstruction is not identified')
            check(((cached_r3>lo)&(cached_r3<hi)&(cached_v2>lo)&(cached_v2<hi)).all(),
                  f'PFN_CACHE_BOUNDARY {name}/{index}/seed{seed}: conservative strict-interior condition failed')
            pfn_finished=np.asarray(cached['pfn_finished'],float)
            linear=(cached_v2-.8*cached_r3)/.2
            linear_gap=maximum_difference(linear,pfn_finished)
            check(linear_gap<=PFN_TOL,'Common PFN reconstruction differs from direct linear formula')
            pfn_raw=np.asarray(common.inverse_shrink(va,pfn_finished),float)
            check(np.isfinite(pfn_raw).all(),'Reconstructed PFN nonfinite')
            roundtrip=maximum_difference(core.shrink(pfn_raw,va),pfn_finished)
            check(roundtrip<=PFN_TOL,'PFN inverse shrink roundtrip failed')
            if pfn_reference is None:pfn_reference=pfn_finished.copy()
            seed_gap=maximum_difference(pfn_reference,pfn_finished)
            check(seed_gap<=PFN_TOL,'PFN cache should be independent of R3 seed')
            restored_v2=final(core,tr,va,.8*r3raw+.2*pfn_raw)
            v2gap=maximum_difference(restored_v2,cached_v2)
            check(v2gap<=R3_TOL,'Restored original V2 differs from cache')
            audit.update({'pfn_linear_formula_max_difference':linear_gap,'pfn_inverse_shrink_roundtrip_max_difference':roundtrip,
                          'pfn_cross_r3_seed_max_difference':seed_gap,'restored_v2_max_difference':v2gap,
                          'pfn_finished_range':clipping_stats(pfn_finished,lo,hi),'pfn_raw_range':clipping_stats(pfn_raw,lo,hi)})
            preds['pfn']=pfn_raw
            rec=va[['row_id','farm','day','hour','block','sub_ec']].copy()
            rec['validator']=name;rec['validation_fold']=index;rec['seed']=seed;rec['v2']=cached_v2
            for arm in ARMS:
                removed=arm[5:]
                denom=1-WEIGHTS[removed]
                raw=sum(WEIGHTS[k]*v for k,v in preds.items() if k!=removed)/denom
                shrunk=core.shrink(raw,va)
                rec[arm]=np.clip(shrunk,lo,hi)
                audit[arm+'_preclip_range']=clipping_stats(shrunk,lo,hi)
                check(np.isfinite(rec[arm]).all(),f'{arm} nonfinite')
                foldrecords.append({'validator':name,'validation_fold':index,'seed':seed,'arm':arm,'n':len(va),
                                    'baseline_rmse':float(np.sqrt(np.mean((va.sub_ec-cached_v2)**2))),
                                    'candidate_rmse':float(np.sqrt(np.mean((va.sub_ec-rec[arm])**2)))})
            output.append(rec)
            save(HERE/'reconstruction_audit.json',audits)
            log(f'DONE {name}/{index} seed={seed} r3_match={gap:.3g} pfn_roundtrip={roundtrip:.3g}')
        pd.DataFrame(foldrecords).to_csv(HERE/'fold_scores.csv',index=False,encoding='utf-8-sig')
    combined=pd.concat(output,ignore_index=True)
    check(combined.groupby(['validator','validation_fold','seed','row_id']).size().max()==1,'Duplicate OOF occurrence')
    diag=combined[combined.validator.eq('DIAG10')]
    check(all(len(g)==8640 and g.row_id.is_unique for _,g in diag.groupby('seed')),'DIAG10 expected once per row per seed')
    combined.to_csv(OUT/'oof_predictions.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    result=common.evaluate(combined,ARMS)
    save(HERE/'evaluation.json',result)
    for key in ['scores','bootstrap','decisions']:
        pd.DataFrame(result[key]).to_csv(HERE/f'{key}.csv',index=False,encoding='utf-8-sig')
    unique=combined.groupby(['validator','seed','row_id','farm','day','hour','block'],sort=True)[['sub_ec','v2']+ARMS].mean().reset_index()
    unique_scores=[]
    for (name,seed),g in unique.groupby(['validator','seed']):
        for arm in ['v2']+ARMS:
            unique_scores.append({'validator':name,'seed':int(seed),'arm':arm,'n_unique_rows':len(g),
                                  'rmse':float(np.sqrt(np.mean((g[arm]-g.sub_ec)**2))),
                                  'aggregation':'row_id mean over validation fold predictions'})
    pd.DataFrame(unique_scores).to_csv(HERE/'unique_row_scores.csv',index=False,encoding='utf-8-sig')
    save(HERE/'completion.json',{'status':'COMPLETE','folds':len(folds),'seeds':SEEDS,'rows':len(combined),
                                 'elapsed_seconds':time.monotonic()-started,'manifest':manifest,
                                 'all_baseline_reconstruction_gates_passed':True,'lock_scored':False,
                                 'test_predictions_created':False,'final_model_trained':False})
    log('MEMBER ABLATION COMPLETE')

if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except GateFailure as ex:
        save(HERE/'gate_failure.json',{'status':'BLOCKED','reason':str(ex),'cached_pfn_use_after_failure':False,
                                      'candidate_scores_accepted':False,'requires_new_baseline_reproduction':True})
        log(str(ex))
        raise
