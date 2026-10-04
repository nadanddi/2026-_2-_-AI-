"""family20: replace only .2 PFN with CPU TabDPT mean. No auto-download or tuning."""
from pathlib import Path
import sys, os, json, argparse, importlib.util, hashlib, math, gc, time
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1')); import support as S
import numpy as np
import pandas as pd
import sklearn, lightgbm, platform
from threadpoolctl import threadpool_limits

OUT = ROOT/'집/코덱스/local'/H.name
SITE = ROOT/'집/코덱스/local/tabdpt130_cpu_v1/site'
WEIGHT = SITE.parent/'weights/tabdpt1_3.safetensors'
BASE = ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
OLD = ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
PREP = H.parent/'ec_tabdpt_preparation_20261004_v1'
SEEDS = [7,101,2024]
RAW_ATOL = 1e-6
FINAL_ATOL = 2e-7
S_SHA = '82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed'
assert S.sha(Path(S.__file__)) == S_SHA
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path); obj=importlib.util.module_from_spec(sp); sp.loader.exec_module(obj); return obj
A = module('tabdpt_fixed_spec',PREP/'adapter_draft_v1.py')
def compare(a,b,tol=1e-12):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    error=float(np.max(np.abs(a-b))); assert error<=tol,(error,tol); return error
def compare_nullable(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
    assert not np.isinf(a).any() and not np.isinf(b).any()
    mask=np.isfinite(a)
    return compare(a[mask],b[mask]) if mask.any() else 0.
def dependency_hashes(core):
    return dict(core=S.sha(Path(core.__file__)),season=S.sha(Path(sys.modules[S.mapping.__module__].__file__)),
                env=S.sha(Path(env.__file__)),support=S_SHA,adapter=S.sha(PREP/'adapter_draft_v1.py'),
                runtime_probe=S.sha(PREP/'runtime_probe_v3.py'),run=S.sha(Path(__file__)))
def validate_first(record,signature):
    assert record['status']=='PASS' and record['signature']==signature
    assert record['raw_atol']==RAW_ATOL and record['final_atol']==FINAL_ATOL
    assert set(record['raw_errors'])=={'repeat','single','reversed','other_query','fresh_fit'}
    assert all(np.isfinite(x) and 0<=x<=RAW_ATOL for x in record['raw_errors'].values())
    assert np.isfinite(record['scalar_maxdiff']) and 0<=record['scalar_maxdiff']<=1e-12
    expected={(f,h) for f in ['F13','F47'] for h in [0,6,12]}
    assert len(record['prefix'])==6 and {(x['farm'],x['hour']) for x in record['prefix']}==expected
    assert all(0<=x['raw_maxdiff']<=RAW_ATOL and 0<=x['final_maxdiff']<=FINAL_ATOL for x in record['prefix'])
    assert len(record['feature_causal'])==6 and {(x['farm'],x['hour']) for x in record['feature_causal']}==expected
    assert all(0<=x['feature_maxdiff']<=1e-12 for x in record['feature_causal'])
def array_sha(x):
    x=np.asarray(x); return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ids_sha(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def runtime_core():
    found=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__)
    assert found==dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0'),found
    return found
def preflight(lab,core,wv,folds,outer):
    cols=[c for c in core.FULL if c!='day']+['season']; assert tuple(cols)==A.FULL38
    runtime=runtime_core(); refs={}; manifests=[]
    deps=dependency_hashes(core); input_sha=S.sha(Path(env.DATA)/'train_X.csv'); public_sha=S.sha(OLD/'v2_integration_oof.csv')
    for v,k,tm,vm in folds:
        tr,q=S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
        assert len(tr)>512 and not set(tr.row_id)&set(q.row_id)
        assert not np.isnan(tr[cols].to_numpy(float)).all(axis=0).any()
        old=dict(np.load(BASE/f'{v}_{k}_baseline.npz',allow_pickle=False))
        assert np.array_equal(old['row_id'],q.row_id)
        compare([old['lo'],old['hi']],[tr.sub_ec.min(),tr.sub_ec.max()])
        pfn=[]
        for c in [1,2,3,4]:
            bag=dict(np.load(OLD/f'{v}_{k}_pfn_{c}.npz',allow_pickle=False))
            assert np.array_equal(bag['row_id'],q.row_id) and np.array_equal(bag['context_row_id'],old[f'context_{c}'])
            assert set(bag['context_row_id'])<=set(tr.row_id)
            compare(bag['sub_ec'],q.sub_ec);pfn.append(bag['raw_pfn'])
        compare(old['old_pfn_raw'],np.mean(pfn,axis=0))
        for seed in SEEDS:
            original_meta=json.loads((OLD/f'{v}_{k}_r3_{seed}.json').read_text(encoding='utf-8'))
            assert original_meta['provenance']['shared']['input_sha256']['train_X.csv']==input_sha
            assert original_meta['provenance']['shared']['core_sha256']==deps['core']
            assert all(original_meta['provenance']['environment'][name]==value for name,value in runtime.items())
            r3=dict(np.load(OLD/f'{v}_{k}_r3_{seed}.npz',allow_pickle=False))
            assert np.array_equal(r3['row_id'],q.row_id);compare(old[f'r3_{seed}'],r3['raw_r3'])
            actual=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
            compare(actual,old[f'baseline_{seed}'])
            compare(np.clip(core.shrink(.8*old[f'r3_{seed}']+.2*old['old_pfn_raw'],q),old['lo'],old['hi']),actual)
        sig=dict(validator=v,fold=k,runtime=runtime,features=cols,
                 dependencies=deps,input_sha256=input_sha,public_cache_sha256=public_sha,
                 train_ids=ids_sha(tr.row_id),query_ids=ids_sha(q.row_id),
                 train_features=array_sha(tr[cols].to_numpy(float)),query_features=array_sha(q[cols].to_numpy(float)),
                 train_targets=array_sha(tr.sub_ec.to_numpy(float)),query_targets=array_sha(q.sub_ec.to_numpy(float)),
                 bounds=[float(old['lo']),float(old['hi'])],baseline_sha256=S.sha(BASE/f'{v}_{k}_baseline.npz'))
        refs[v,k]=(tr,q,old,sig);manifests.append(sig)
    assert len(refs)==22
    return cols,refs,dict(status='PASS',folds=22,manifest=manifests,runtime=runtime,fit_count=0,dependencies=deps,input_sha256=input_sha,public_cache_sha256=public_sha)
def feature_audit(lab,core,cols):
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+core.RAW)
    raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
    original=core.features(raw).set_index('row_id')
    records=[]
    for farm in ['F13','F47']:
        day=int(lab.loc[lab.farm==farm,'day'].min())
        for hour in [0,6,12]:
            keep=(original.farm==farm)&(original.day==day)&(original.hour<=hour)
            ids=original.index[keep]
            changed=raw.copy(); other=~changed.row_id.isin(ids)
            changed.loc[other,core.RAW]=changed.loc[other,core.RAW]*17+1000
            new=core.features(changed).set_index('row_id')
            gap=compare_nullable(original.loc[ids,cols[:-1]],new.loc[ids,cols[:-1]])
            records.append(dict(farm=farm,day=day,hour=hour,feature_maxdiff=gap))
    return records
def first_audit(model,tr,q,cols,seed,raw,core,old,make,predict):
    start=time.perf_counter()
    original=raw[:8]
    repeat=predict(model,q[cols].to_numpy(float)[:8],seed)
    single=np.array([predict(model,q[cols].to_numpy(float)[i:i+1],seed)[0] for i in range(8)])
    reverse=predict(model,q[cols].to_numpy(float)[:8][::-1].copy(),seed)[::-1]
    changed=q[cols].to_numpy(float)[:8].copy();changed[1:]=np.where(np.isfinite(changed[1:]),changed[1:]+1e4,0.)
    other=predict(model,changed,seed)
    fresh=make(tr,cols,seed); fresh_raw=predict(fresh,q[cols].to_numpy(float)[:8],seed)
    errors=dict(repeat=compare(original,repeat,RAW_ATOL),single=compare(original,single,RAW_ATOL),
                reversed=compare(original,reverse,RAW_ATOL),other_query=compare(original[:1],other[:1],RAW_ATOL),
                fresh_fit=compare(original,fresh_raw,RAW_ATOL))
    del fresh;gc.collect()
    combined=.8*old[f'r3_{seed}']+.2*raw
    final=np.clip(core.shrink(combined,q),old['lo'],old['hi'])
    scalar_max=0.
    for _,g in q.assign(raw=combined,final=final).groupby(['farm','day']):
        history=[]
        for r in g.sort_values('hour').itertuples():
            history.append(float(r.raw));expected=min(float(old['hi']),max(float(old['lo']),.5*r.raw+.5*math.fsum(history)/len(history)))
            scalar_max=max(scalar_max,compare([expected],[r.final]))
    causal=[]
    for farm in ['F13','F47']:
        day=int(q.loc[q.farm==farm,'day'].min())
        for hour in [0,6,12]:
            keep=(q.farm==farm)&(q.day==day)&(q.hour<=hour)
            prefix=q[keep].copy()
            raw_prefix=predict(model,prefix[cols].to_numpy(float),seed)
            rgap=compare(raw[keep],raw_prefix,RAW_ATOL)
            selected=np.clip(core.shrink(.8*old[f'r3_{seed}'][keep]+.2*raw_prefix,prefix),old['lo'],old['hi'])
            fgap=compare(final[keep],selected,FINAL_ATOL)
            causal.append(dict(farm=farm,day=day,hour=hour,raw_maxdiff=rgap,final_maxdiff=fgap))
    return dict(status='PASS',raw_atol=RAW_ATOL,final_atol=FINAL_ATOL,raw_errors=errors,
                scalar_maxdiff=scalar_max,feature_causal=feature_audit(tr,core,cols),prefix=causal,audit_seconds=time.perf_counter()-start)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    lab,core,wv,folds,outer=S.loadec();cols,refs,prepared=preflight(lab,core,wv,folds,outer)
    if args.prepare:
        S.savej(H/'preparation_v2.json',prepared);print('PREPARATION_PASS_NO_FIT',flush=True);return
    assert json.loads((H/'preparation_v2.json').read_text(encoding='utf-8'))==prepared,'Prepared inputs/code changed: stop before fitting.'
    assert not (H/'fit_audit_v1.json').exists(),'Complete experiment: verify without retraining.'
    R=module('tabdpt_import_guard',PREP/'runtime_probe_v3.py')
    imports=R.import_probe(SITE)
    A.verify_weight(WEIGHT,execution_authorized=True)
    import torch, faiss
    from tabdpt.regressor import TabDPTRegressor
    torch.set_num_threads(1);torch.set_num_interop_threads(1);faiss.omp_set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[name]='1'
    def make(tr,cols,seed):
        torch.manual_seed(seed);np.random.seed(seed)
        model=TabDPTRegressor(model_weight_path=str(WEIGHT),**A.CONSTRUCTOR)
        assert model.max_features>=len(cols)
        with threadpool_limits(limits=1):model.fit(tr[cols].to_numpy(float),tr.sub_ec.to_numpy(float))
        return model
    def predict(model,x,seed):
        with threadpool_limits(limits=1):p=np.asarray(model.predict(x,seed=seed,**A.PREDICT),float)
        assert p.shape==(len(x),) and np.isfinite(p).all();return p
    provenance=dict(run_sha256=S.sha(Path(__file__)),support_sha256=S_SHA,adapter_sha256=S.sha(PREP/'adapter_draft_v1.py'),
                    runtime_probe_sha256=S.sha(PREP/'runtime_probe_v3.py'),weight_sha256=S.sha(WEIGHT),
                    input_sha256=S.sha(Path(env.DATA)/'train_X.csv'),runtime=imports)
    OUT.mkdir(parents=True,exist_ok=True);audit=[];outputs=[]
    for v,k,_,_ in folds:
        tr,q,old,sig=refs[v,k]
        for seed in SEEDS:
            dest=OUT/f'{v}_{k}_{seed}_pred.csv';meta=OUT/f'{v}_{k}_{seed}.json'
            first=H/'first_fold_verification_v1.json'
            signature=dict(**sig,seed=seed,provenance=provenance)
            files=[dest,meta]+([first] if (v,k,seed)==('DIAG10',0,7) else [])
            present=[p.exists() for p in files];assert not any(present) or all(present),'Partial cell: preserve and stop.'
            if all(present):
                saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['signature']==signature and saved['csv_sha256']==S.sha(dest) and saved['status']=='PASS'
                d=pd.read_csv(dest,float_precision='round_trip');assert np.array_equal(d.row_id,q.row_id)
                compare(d.y,q.sub_ec);compare(d.baseline,old[f'baseline_{seed}'])
                for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
                assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==seed).all()
                compare(d.r3_raw,old[f'r3_{seed}']);compare(d.old_pfn_raw,old['old_pfn_raw'])
                compare(d.clip_lo,np.repeat(old['lo'],len(q)));compare(d.clip_hi,np.repeat(old['hi'],len(q)))
                if len(files)==3:validate_first(json.loads(first.read_text(encoding='utf-8')),signature)
            else:
                start=time.perf_counter();model=make(tr,cols,seed);raw=predict(model,q[cols].to_numpy(float),seed)
                check=first_audit(model,tr,q,cols,seed,raw,core,old,make,predict) if (v,k,seed)==('DIAG10',0,7) else None
                d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=old[f'baseline_{seed}']
                d['candidate']=np.clip(core.shrink(.8*old[f'r3_{seed}']+.2*raw,q),old['lo'],old['hi'])
                d['new_tabdpt_raw']=raw;d['old_pfn_raw']=old['old_pfn_raw'];d['r3_raw']=old[f'r3_{seed}']
                d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=float(old['lo']);d['clip_hi']=float(old['hi'])
                d.to_csv(dest,index=False)
                saved=dict(status='PASS',signature=signature,csv_sha256=S.sha(dest),fit_predict_seconds=time.perf_counter()-start)
                S.savej(meta,saved)
                if check is not None:
                    first_record=dict(**check,signature=signature);validate_first(first_record,signature);S.savej(first,first_record)
                del model;gc.collect()
            compare(d.candidate,np.clip(core.shrink(.8*d.r3_raw.to_numpy()+.2*d.new_tabdpt_raw.to_numpy(),q),old['lo'],old['hi']))
            audit.append(saved);outputs.append(d);print(v,k,seed,'COMPLETE',flush=True)
    assert len(audit)==66
    pd.concat(outputs,ignore_index=True).to_csv(OUT/'oof.csv',index=False)
    S.savej(H/'fit_audit_v1.json',dict(status='PASS',family=20,preparation=prepared,cells=audit))
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
