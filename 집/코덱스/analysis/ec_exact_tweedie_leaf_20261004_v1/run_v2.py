"""family24 EXACT_TWEEDIE_LEAF: one Tweedie member, fixed original recipe."""
from pathlib import Path
import sys, os, json, argparse, hashlib, math, gc, time
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
BASE = ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
OLD = ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
SEEDS = [7,101,2024]
S_SHA = '82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed'
assert S.sha(Path(S.__file__)) == S_SHA
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

import ast
GUARD_SHA = 'e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9'
CORE_SHA = '057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2'
NEW=[]
FOLD_KEYS = [(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
FAMILY=24
ALPHA=.025/FAMILY

def savej(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def array_sha(x):
    x=np.asarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ids_sha(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()

def runtime_core():
    found=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__)
    assert found==dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0'),found
    return found

def extract(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def operations(frame):return frame.copy()

BUILD=json.loads((H/'build_result_v3.json').read_text(encoding='utf-8'))
DLL=Path(BUILD['dll'])
def library_guard():
    assert BUILD['status']=='PASS_BUILD_ONLY' and S.sha(DLL)==BUILD['dll_sha256']
    overlay=json.loads((H/'overlay_manifest_v1.json').read_text(encoding='utf-8'))
    assert S.sha(H/'overlay_manifest_v1.json')==BUILD['overlay_manifest_sha256']
    source=OUT/'overlay_v1'
    for name,item in overlay['changes'].items():assert S.sha(source/name)==item['overlay_sha256']
    assert S.sha(source/'src/objective/farmai_leaf_audit.hpp')==overlay['extra_header_sha256']
    return dict(dll=str(DLL),dll_sha256=S.sha(DLL),build_sha256=S.sha(H/'build_result_v3.json'),overlay_sha256=S.sha(H/'overlay_manifest_v1.json'),synthetic_sha256=S.sha(H/'synthetic_result_v4.json'),cpp_audit_source_sha256=S.sha(H/'first_cpp_audit_v2.py'))
def bind_library():
    import ctypes,lightgbm.basic as basic
    global DLL_HANDLES
    DLL_HANDLES=[os.add_dll_directory(str(OUT/'toolchain_v1/mingw64/bin')),os.add_dll_directory(str(DLL.parent))]
    original=Path(basic._LIB._name)
    assert S.sha(original)=='7e366d2e49cd061aac3ab21676b2f99b0c7a758dc3e888d4b23812af1b7d301c'
    basic._LIB=ctypes.cdll.LoadLibrary(str(DLL));basic._LIB.LGBM_GetLastError.restype=ctypes.c_char_p
    assert Path(basic._LIB._name).resolve()==DLL.resolve()
    return original

def preflight():
    runtime=runtime_core();lab,core,wv,folds,outer=S.loadec()
    assert S.sha(Path(core.__file__))==CORE_SHA
    assert [(v,k) for v,k,_,_ in folds]==FOLD_KEYS
    lab=operations(lab)
    fs=[c for c in core.FULL if c!='day']+['season']
    bs=[c for c in core.BASE if c!='day']+['season'];cols=bs+NEW
    assert len(fs)==38 and len(bs)==14 and len(cols)==14 and bs[-1]=='season' and 'day' not in cols
    deps=dict(run=S.sha(Path(__file__)),core=CORE_SHA,support=S_SHA,season=S.sha(Path(sys.modules[S.mapping.__module__].__file__)),env=S.sha(Path(env.__file__)))
    label_map=outer[(outer.validator=='DIAG10')&(outer.seed==7)].set_index('row_id').sub_ec
    assert len(label_map)==8640 and label_map.index.is_unique
    compare(lab.sub_ec,label_map.reindex(lab.row_id))
    guard_path=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
    assert S.sha(guard_path)==GUARD_SHA
    guard_ns=extract(guard_path,['guard_original_r3'],dict(np=np,hashlib=hashlib,json=json,compare=compare,ids_sha=ids_sha,array_sha=array_sha))
    deps['original_guard']=S.sha(guard_path)
    inputs=dict(train_X=S.sha(Path(env.DATA)/'train_X.csv'),public_oof=S.sha(OLD/'v2_integration_oof.csv'))
    refs={};manifests=[];guards=[]
    for v,k,tm,vm in folds:
        tr,q=S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
        banned={(f,int(d)+j) for f,d in q[['farm','day']].itertuples(index=False,name=None) for j in [-1,0,1]}
        assert not set(tr[['farm','day']].itertuples(index=False,name=None))&banned
        base_path=BASE/f'{v}_{k}_baseline.npz';old=dict(np.load(base_path,allow_pickle=False))
        assert np.array_equal(old['row_id'],q.row_id)
        compare([old['lo'],old['hi']],[tr.sub_ec.min(),tr.sub_ec.max()])
        pfn=[];cache_hashes={str(base_path.relative_to(ROOT)):S.sha(base_path)};r3s={}
        for c in [1,2,3,4]:
            p=OLD/f'{v}_{k}_pfn_{c}.npz';bag=dict(np.load(p,allow_pickle=False));cache_hashes[str(p.relative_to(ROOT))]=S.sha(p)
            assert np.array_equal(bag['row_id'],q.row_id) and np.array_equal(bag['context_row_id'],old[f'context_{c}'])
            assert set(bag['context_row_id'])<=set(tr.row_id)
            compare(bag['sub_ec'],q.sub_ec);pfn.append(bag['raw_pfn'])
        compare(old['old_pfn_raw'],np.mean(pfn,axis=0))
        for seed in SEEDS:
            p=OLD/f'{v}_{k}_r3_{seed}.npz';meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));r3=dict(np.load(p,allow_pickle=False))
            assert meta['provenance']['feature_order']=='remove day; append season'
            assert meta['provenance']['shared']['input_sha256']['train_X.csv']==inputs['train_X']
            assert meta['provenance']['shared']['core_sha256']==CORE_SHA
            assert all(meta['provenance']['environment'][n]==value for n,value in runtime.items())
            guards.append(guard_ns['guard_original_r3'](r3,meta,S.sha(p),tr,q,seed,v,k,label_map))
            cache_hashes[str(p.relative_to(ROOT))]=S.sha(p);cache_hashes[str(p.with_suffix('.json').relative_to(ROOT))]=S.sha(p.with_suffix('.json'))
            compare(old[f'r3_{seed}'],r3['raw_r3'])
            actual=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
            compare(actual,old[f'baseline_{seed}'])
            compare(actual,np.clip(core.shrink(.8*r3['raw_r3']+.2*old['old_pfn_raw'],q),old['lo'],old['hi']))
            r3s[seed]=r3
        signature=dict(library=library_guard(),validator=v,fold=k,runtime=runtime,dependencies=deps,inputs=inputs,cache_hashes=cache_hashes,full_features=fs,old_lgb_features=bs,new_lgb_features=cols,train_ids=ids_sha(tr.row_id),query_ids=ids_sha(q.row_id),bounds=[float(old['lo']),float(old['hi'])])
        for part,frame in [('train',tr),('query',q)]:
            signature[part+'_targets']=array_sha(frame.sub_ec.to_numpy(float))
            for name,c in [('full38',fs),('old14',bs),('new14',cols)]:signature[part+'_'+name]=array_sha(frame[c].to_numpy(float))
        refs[v,k]=(tr,q,old,r3s,signature);manifests.append(signature)
    assert len(guards)==66
    prepared=dict(status='PASS',family=FAMILY,alpha=ALPHA,fit_count=0,predict_count=0,score_count=0,runtime=runtime,dependencies=deps,inputs=inputs,manifest=manifests,original_r3_guard=guards,params={str(s):core.lg(s,'tweedie').get_params() for s in SEEDS},threadpool_limits=2)
    return core,lab,fs,bs,cols,refs,prepared

def combined(old,r3,new):return .8*(.6*r3['raw_et']+.3*new+.1*r3['raw_mlp'])+.2*old['old_pfn_raw']
def final(core,q,old,r3,new):return np.clip(core.shrink(combined(old,r3,new),q),old['lo'],old['hi'])

def causal_features(core,lab,fs):
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+core.RAW)
    raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    original=operations(core.features(raw)).set_index('row_id');records=[]
    for farm in ['F13','F47']:
        day=int(lab.loc[lab.farm==farm,'day'].min())
        for hour in [0,6,12]:
            ids=original.index[(original.farm==farm)&(original.day==day)&(original.hour<=hour)]
            changed=raw.copy();mask=~changed.row_id.isin(ids);changed.loc[mask,core.RAW]=changed.loc[mask,core.RAW]*17+1000
            other=operations(core.features(changed)).set_index('row_id')
            gap=compare_nullable(original.loc[ids,fs[:-1]+NEW],other.loc[ids,fs[:-1]+NEW])
            records.append(dict(farm=farm,day=day,hour=hour,feature_maxdiff=gap))
    return records

def first_audit(core,lab,fs,bs,cols,tr,q,old,r3,seed,model,raw,make,predict,reproduction):
    fresh=make(tr,cols,seed)
    n=min(8,len(q));small=q.iloc[:n].copy();expected=raw[:n]
    changed=small.copy();changed.loc[changed.index[1:],cols]=changed.loc[changed.index[1:],cols].fillna(0)+10000
    errors=dict(repeat=compare(expected,predict(model,small,cols)),fresh_fit=compare(raw,predict(fresh,q,cols)),single=compare(expected,np.array([predict(model,small.iloc[i:i+1],cols)[0] for i in range(n)])),reversed=compare(expected,predict(model,small.iloc[::-1],cols)[::-1]),other_query=compare(expected[:1],predict(model,changed,cols)[:1]))
    output=final(core,q,old,r3,raw);rawmix=combined(old,r3,raw)
    compare(rawmix,.48*r3['raw_et']+.24*raw+.08*r3['raw_mlp']+.2*old['old_pfn_raw'])
    scalar_gap=0.
    for _,g in q.assign(rawmix=rawmix,output=output).groupby(['farm','day']):
        hist=[]
        for r in g.sort_values('hour').itertuples():
            hist.append(float(r.rawmix));value=min(float(old['hi']),max(float(old['lo']),.5*r.rawmix+.5*math.fsum(hist)/len(hist)))
            scalar_gap=max(scalar_gap,compare([value],[r.output]))
    prefixes=[]
    for farm in ['F13','F47']:
        day=int(q.loc[q.farm==farm,'day'].min())
        for hour in [0,6,12]:
            keep=(q.farm==farm)&(q.day==day)&(q.hour<=hour);p=q[keep];pred=predict(model,p,cols)
            gap=compare(raw[keep],pred)
            sliced={name:r3[name][keep] for name in ['raw_et','raw_mlp']};oldsub=dict(old_pfn_raw=old['old_pfn_raw'][keep],lo=old['lo'],hi=old['hi'])
            prefixes.append(dict(farm=farm,day=day,hour=hour,raw_maxdiff=gap,final_maxdiff=compare(output[keep],final(core,p,oldsub,sliced,pred))))
    del fresh
    return dict(status='PASS',original_lgb_maxdiff=reproduction,raw_errors=errors,scalar_maxdiff=scalar_gap,prefix=prefixes,feature_causal=causal_features(core,lab,fs),atol=1e-12)

def validate_first(record,signature):
    assert record['cpp_audit']['status']=='PASS' and record['cpp_trace_sha256']==S.sha(OUT/'first_exact_cpp_trace_v1.ndjson')
    assert record['status']=='PASS' and record['signature']==signature and record['atol']==1e-12
    assert 0<=record['original_lgb_maxdiff']<=1e-12 and 0<=record['scalar_maxdiff']<=1e-12
    assert set(record['raw_errors'])=={'repeat','fresh_fit','single','reversed','other_query'}
    assert all(0<=x<=1e-12 for x in record['raw_errors'].values())
    for key,fields in [('prefix',['raw_maxdiff','final_maxdiff']),('feature_causal',['feature_maxdiff'])]:
        assert {(x['farm'],x['hour']) for x in record[key]}=={(f,h) for f in ['F13','F47'] for h in [0,6,12]} and len(record[key])==6
        assert all(0<=r[c]<=1e-12 for r in record[key] for c in fields)

def freshness(core,signature,prepared_hash,prereg_hash):
    assert library_guard()==signature['library']
    assert runtime_core()==signature['runtime']
    paths=dict(run=Path(__file__),core=Path(core.__file__),support=Path(S.__file__),season=Path(sys.modules[S.mapping.__module__].__file__),env=Path(env.__file__),original_guard=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py')
    assert set(paths)==set(signature['dependencies'])
    for name,path in paths.items():assert S.sha(path)==signature['dependencies'][name],('Source changed before fit',name)
    assert S.sha(H/'preparation_v2.json')==prepared_hash and S.sha(H/'preregistration_v1.md')==prereg_hash,'Preparation or preregistration changed before fit'
    assert S.sha(Path(env.DATA)/'train_X.csv')==signature['inputs']['train_X']
    assert S.sha(OLD/'v2_integration_oof.csv')==signature['inputs']['public_oof']
    for relative,expected in signature['cache_hashes'].items():assert S.sha(ROOT/relative)==expected,('Original cache changed before fit',relative)

def preserve_failed_raw(kind,v,k,seed,q,raw):
    path=OUT/f'failure_{v}_{k}_{seed}_{kind}_raw.npz'
    with path.open('xb') as handle:np.savez_compressed(handle,row_id=q.row_id.to_numpy(str),raw=np.asarray(raw,float))
    return path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    core,lab,fs,bs,cols,refs,prepared=preflight()
    if args.prepare:
        savej(H/'preparation_v2.json',prepared);print('PREPARATION_PASS_FIT0_PREDICT0_SCORE0',flush=True);return
    prereg=H/'preregistration_v1.md';assert prereg.is_file()
    assert json.loads((H/'preparation_v2.json').read_text(encoding='utf-8'))==prepared,'Source/input/runtime/metadata changed: stop before fitting.'
    assert not (H/'fit_audit_v1.json').exists() and not (OUT/'oof.csv').exists(),'Preserve completed aggregate; no fit.'
    prepared_hash=S.sha(H/'preparation_v2.json');prereg_hash=S.sha(prereg)
    assert not list(H.glob('failure_*.json')), 'A failed run is preserved: fresh reviewed version required'
    assert not list(OUT.glob('failure_*_raw.npz')),'Failed raw artifacts preserved: do not restart.'
    assert json.loads((H/'synthetic_result_v4.json').read_text())['status']=='PASS_SYNTHETIC_ONLY'
    bind_library()
    active_sig=None
    fit_mode='tweedie'
    def make(tr,features,seed):
        freshness(core,active_sig,prepared_hash,prereg_hash)
        model=core.lg(seed,'tweedie');assert model.get_params()==prepared['params'][str(seed)]
        if fit_mode!='tweedie':model.set_params(objective=fit_mode)
        with threadpool_limits(limits=2):model.fit(tr[features],tr.sub_ec)
        return model
    def predict(model,q,features):
        with threadpool_limits(limits=2):p=np.asarray(model.predict(q[features]),float)
        assert p.shape==(len(q),) and np.isfinite(p).all();return p
    OUT.mkdir(parents=True,exist_ok=True)
    controls=[]
    for v,k in FOLD_KEYS:
        tr,q,old,r3s,sig=refs[v,k];active_sig=sig
        for seed in SEEDS:
            dest=OUT/f'{v}_{k}_{seed}_newton_control.npz';meta=OUT/f'{v}_{k}_{seed}_newton_control.json'
            present=[dest.exists(),meta.exists()];assert not any(present) or all(present),'Partial Newton control preserved'
            if all(present):
                record=json.loads(meta.read_text(encoding='utf-8'));assert record['status']=='PASS' and record['seed']==seed and record['signature']==sig and record['preparation_sha256']==prepared_hash and record['preregistration_sha256']==prereg_hash and record['npz_sha256']==S.sha(dest)
                raw=dict(np.load(dest,allow_pickle=False));assert np.array_equal(raw['row_id'],q.row_id)
                compare(raw['raw'],r3s[seed]['raw_lgb'])
            else:
                model=make(tr,bs,seed);raw=predict(model,q,bs)
                try:error=compare(raw,r3s[seed]['raw_lgb'])
                except BaseException:preserve_failed_raw('compiled_newton',v,k,seed,q,raw);raise
                with dest.open('xb') as handle:np.savez_compressed(handle,row_id=q.row_id.to_numpy(str),raw=raw)
                record=dict(status='PASS',signature=sig,seed=seed,preparation_sha256=prepared_hash,preregistration_sha256=prereg_hash,npz_sha256=S.sha(dest),max_error=error);savej(meta,record)
                del model;gc.collect()
            controls.append(record);print(v,k,seed,'NEWTON_REPRODUCE_PASS',flush=True)
    assert len(controls)==66
    control_path=H/'compiled_newton_all66_v1.json'
    control_summary=dict(status='PASS',cells=controls,atol=1e-12,score_count=0)
    if control_path.exists():assert json.loads(control_path.read_text(encoding='utf-8'))==control_summary
    else:savej(control_path,control_summary)
    fit_mode='tweedie_exact_leaf'
    outputs=[];audits=[];first=H/'first_fold_verification_v1.json'

    for v,k in FOLD_KEYS:
        tr,q,old,r3s,sig=refs[v,k]
        active_sig=sig
        for seed in SEEDS:
            r3=r3s[seed];dest=OUT/f'{v}_{k}_{seed}_pred.csv';meta=OUT/f'{v}_{k}_{seed}.json'
            signature=dict(**sig,seed=seed,preparation_sha256=S.sha(H/'preparation_v2.json'),preregistration_sha256=S.sha(prereg))
            files=[dest,meta]+([first] if (v,k,seed)==('DIAG10',0,7) else [])
            present=[p.exists() for p in files];assert not any(present) or all(present),'Partial cell preserved: stop.'
            if all(present):
                saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['status']=='PASS' and saved['signature']==signature and saved['csv_sha256']==S.sha(dest)
                d=pd.read_csv(dest,float_precision='round_trip')
                if len(files)==3:
                    assert saved['first_audit_sha256']==S.sha(first);validate_first(json.loads(first.read_text(encoding='utf-8')),signature)
            else:
                start=time.perf_counter();reproduction=None
                reproduction=controls[0]['max_error']
                trace=None
                if (v,k,seed)==('DIAG10',0,7):
                    trace=OUT/'first_exact_cpp_trace_v1.ndjson';assert not trace.exists();trace.open('x').close()
                    os.chdir(OUT);os.environ['FARMAI_LGB_AUDIT_PATH']=trace.name
                try:model=make(tr,cols,seed)
                finally:os.environ.pop('FARMAI_LGB_AUDIT_PATH',None)
                raw=predict(model,q,cols)

                try:
                    check=first_audit(core,lab,fs,bs,cols,tr,q,old,r3,seed,model,raw,make,predict,reproduction) if (v,k,seed)==('DIAG10',0,7) else None
                    if check is not None:
                        from first_cpp_audit_v2 import audit_trace
                        check['cpp_audit']=audit_trace(trace,model,tr,cols)
                        check['cpp_trace_sha256']=S.sha(trace)
                except BaseException:
                    preserve_failed_raw('candidate_lgb',v,k,seed,q,raw)
                    raise
                d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=old[f'baseline_{seed}'];d['candidate']=final(core,q,old,r3,raw)
                d['new_lgb_raw']=raw
                for name in ['raw_et','raw_lgb','raw_mlp']:d[name]=r3[name]
                d['old_pfn_raw']=old['old_pfn_raw'];d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=float(old['lo']);d['clip_hi']=float(old['hi'])
                with dest.open('x',encoding='utf-8',newline='') as f:d.to_csv(f,index=False)
                saved=dict(status='PASS',signature=signature,first_audit_sha256=S.sha(first) if first.exists() else None,csv_sha256=S.sha(dest),fit_predict_seconds=time.perf_counter()-start)
                if check is not None:
                    record=dict(**check,signature=signature);validate_first(record,signature);savej(first,record);saved['first_audit_sha256']=S.sha(first)
                savej(meta,saved);del model;gc.collect()
            assert np.array_equal(d.row_id,q.row_id) and d.row_id.is_unique
            for name in ['farm','day','hour']:assert np.array_equal(d[name],q[name])
            assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==seed).all()
            compare(d.y,q.sub_ec);compare(d.baseline,old[f'baseline_{seed}'])
            for name in ['raw_et','raw_lgb','raw_mlp']:compare(d[name],r3[name])
            compare(d.old_pfn_raw,old['old_pfn_raw']);compare(d.clip_lo,np.repeat(old['lo'],len(q)));compare(d.clip_hi,np.repeat(old['hi'],len(q)))
            compare(d.candidate,final(core,q,old,r3,d.new_lgb_raw.to_numpy()))
            outputs.append(d);audits.append(saved);print(v,k,seed,'COMPLETE',flush=True)
    assert len(audits)==66 and not (OUT/'oof.csv').exists()
    with (OUT/'oof.csv').open('x',encoding='utf-8',newline='') as f:pd.concat(outputs,ignore_index=True).to_csv(f,index=False)
    savej(H/'fit_audit_v1.json',dict(status='PASS',family=FAMILY,preparation=prepared,compiled_newton_all66_sha256=S.sha(control_path),cells=audits,score_count=0))
    print('ALL_FITS_COMPLETE_SCORE0',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        failure=H/('failure_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.json')
        savej(failure,dict(status='STOP_PRESERVE',error_type=type(error).__name__,error=str(error),source_sha256=S.sha(Path(__file__)),traceback=__import__('traceback').format_exc()))
        raise
