"""family21 LGB_OPERATION_ONLY: one Tweedie member, fixed original recipe."""
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
SOURCE = ROOT/'집/클로드/research/ec3_DP1_daily_operation_pattern_v1.py'
SOURCE_SHA = '40d650c639550c09ef96a6be3aae3ef0f969fa1cbd899c787e7724a7f563e86d'
CORE_SHA = '057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2'
NEW = ['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']
CONTROLS = ['act_vent','act_thermal','act_shade','act_heating','act_co2']
FOLD_KEYS = [(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
FAMILY=21
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

def operations(frame):
    assert S.sha(SOURCE)==SOURCE_SHA
    funcs=extract(SOURCE,['run_len','day_feats'],{'np':np,'pd':pd})
    assert frame.row_id.is_unique
    assert not frame[CONTROLS].isna().any().any()
    assert np.isfinite(frame[CONTROLS].to_numpy(float)).all()
    assert (frame[CONTROLS]>=0).all().all()
    for _,g in frame.groupby(['farm','day']):assert np.array_equal(np.sort(g.hour),np.arange(24))
    f=pd.concat([funcs['day_feats'](g) for _,g in frame.groupby(['farm','day'])])
    out=frame.copy();out[NEW]=f[NEW].reindex(frame.index)
    assert np.isfinite(out[NEW].to_numpy(float)).all()
    return out

def preflight():
    runtime=runtime_core();lab,core,wv,folds,outer=S.loadec()
    assert S.sha(Path(core.__file__))==CORE_SHA
    assert [(v,k) for v,k,_,_ in folds]==FOLD_KEYS
    lab=operations(lab)
    fs=[c for c in core.FULL if c!='day']+['season']
    bs=[c for c in core.BASE if c!='day']+['season'];cols=bs+NEW
    assert len(fs)==38 and len(bs)==14 and len(cols)==23 and bs[-1]=='season' and 'day' not in cols
    deps=dict(run=S.sha(Path(__file__)),core=CORE_SHA,support=S_SHA,dp1=SOURCE_SHA,season=S.sha(Path(sys.modules[S.mapping.__module__].__file__)),env=S.sha(Path(env.__file__)))
    label_map=outer[(outer.validator=='DIAG10')&(outer.seed==7)].set_index('row_id').sub_ec
    assert len(label_map)==8640 and label_map.index.is_unique
    compare(lab.sub_ec,label_map.reindex(lab.row_id))
    guard_path=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
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
        signature=dict(validator=v,fold=k,runtime=runtime,dependencies=deps,inputs=inputs,cache_hashes=cache_hashes,full_features=fs,old_lgb_features=bs,new_lgb_features=cols,train_ids=ids_sha(tr.row_id),query_ids=ids_sha(q.row_id),bounds=[float(old['lo']),float(old['hi'])])
        for part,frame in [('train',tr),('query',q)]:
            signature[part+'_targets']=array_sha(frame.sub_ec.to_numpy(float))
            for name,c in [('full38',fs),('old14',bs),('new23',cols)]:signature[part+'_'+name]=array_sha(frame[c].to_numpy(float))
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

def first_audit(core,lab,fs,bs,cols,tr,q,old,r3,seed,model,raw,make,predict):
    original=make(tr,bs,seed)
    reproduction=compare(predict(original,q,bs),r3['raw_lgb'])
    del original
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
    assert record['status']=='PASS' and record['signature']==signature and record['atol']==1e-12
    assert 0<=record['original_lgb_maxdiff']<=1e-12 and 0<=record['scalar_maxdiff']<=1e-12
    assert set(record['raw_errors'])=={'repeat','fresh_fit','single','reversed','other_query'}
    assert all(0<=x<=1e-12 for x in record['raw_errors'].values())
    for key,fields in [('prefix',['raw_maxdiff','final_maxdiff']),('feature_causal',['feature_maxdiff'])]:
        assert {(x['farm'],x['hour']) for x in record[key]}=={(f,h) for f in ['F13','F47'] for h in [0,6,12]} and len(record[key])==6
        assert all(0<=r[c]<=1e-12 for r in record[key] for c in fields)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    core,lab,fs,bs,cols,refs,prepared=preflight()
    if args.prepare:
        savej(H/'preparation_v1.json',prepared);print('PREPARATION_PASS_FIT0_PREDICT0_SCORE0',flush=True);return
    prereg=H/'preregistration_v1.md';assert prereg.is_file()
    assert json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))==prepared,'Source/input/runtime/metadata changed: stop before fitting.'
    assert not (H/'fit_audit_v1.json').exists() and not (OUT/'oof.csv').exists(),'Preserve completed aggregate; no fit.'
    def make(tr,features,seed):
        model=core.lg(seed,'tweedie');assert model.get_params()==prepared['params'][str(seed)]
        with threadpool_limits(limits=2):model.fit(tr[features],tr.sub_ec)
        return model
    def predict(model,q,features):
        with threadpool_limits(limits=2):p=np.asarray(model.predict(q[features]),float)
        assert p.shape==(len(q),) and np.isfinite(p).all();return p
    OUT.mkdir(parents=True,exist_ok=True);outputs=[];audits=[];first=H/'first_fold_verification_v1.json'
    for v,k in FOLD_KEYS:
        tr,q,old,r3s,sig=refs[v,k]
        for seed in SEEDS:
            r3=r3s[seed];dest=OUT/f'{v}_{k}_{seed}_pred.csv';meta=OUT/f'{v}_{k}_{seed}.json'
            signature=dict(**sig,seed=seed,preparation_sha256=S.sha(H/'preparation_v1.json'),preregistration_sha256=S.sha(prereg))
            files=[dest,meta]+([first] if (v,k,seed)==('DIAG10',0,7) else [])
            present=[p.exists() for p in files];assert not any(present) or all(present),'Partial cell preserved: stop.'
            if all(present):
                saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['status']=='PASS' and saved['signature']==signature and saved['csv_sha256']==S.sha(dest)
                d=pd.read_csv(dest,float_precision='round_trip')
                if len(files)==3:
                    assert saved['first_audit_sha256']==S.sha(first);validate_first(json.loads(first.read_text(encoding='utf-8')),signature)
            else:
                start=time.perf_counter();model=make(tr,cols,seed);raw=predict(model,q,cols)
                check=first_audit(core,lab,fs,bs,cols,tr,q,old,r3,seed,model,raw,make,predict) if (v,k,seed)==('DIAG10',0,7) else None
                d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=old[f'baseline_{seed}'];d['candidate']=final(core,q,old,r3,raw)
                d['new_lgb_raw']=raw
                for name in ['raw_et','raw_lgb','raw_mlp']:d[name]=r3[name]
                d['old_pfn_raw']=old['old_pfn_raw'];d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=float(old['lo']);d['clip_hi']=float(old['hi'])
                with dest.open('x',encoding='utf-8',newline='') as f:d.to_csv(f,index=False)
                saved=dict(status='PASS',signature=signature,csv_sha256=S.sha(dest),fit_predict_seconds=time.perf_counter()-start)
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
    savej(H/'fit_audit_v1.json',dict(status='PASS',family=FAMILY,preparation=prepared,cells=audits,score_count=0))
    print('ALL_FITS_COMPLETE_SCORE0',flush=True)
if __name__=='__main__':main()
