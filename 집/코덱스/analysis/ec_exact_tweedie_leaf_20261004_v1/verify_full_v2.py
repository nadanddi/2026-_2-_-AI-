"""family24 독립 whole 감사. runner/preparation/prereg SHA 고정. 전체완료 후 부모만 --verify 실행.
Runner import/fit 없음; checkpoint prediction only. 공개 support loadec/seasonal 및 R3 guard만 재사용.
"""
from pathlib import Path
import sys, json, math, hashlib, ast, argparse, platform, importlib.metadata
sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python removes audit assertions; stop')
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
OUT=ROOT/'집/코덱스/local'/H.name
OLD=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
BASE=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
RUN_FILE='run_v3.py'; PREP_FILE='preparation_v3.json'
RUN_SHA='74246a838fa27e2f83e5cbb6a50f705bf0669f0a3ae0f578b9a6cd1aaf4eea63'
PREP_SHA='9f25648fcdda6ee06c3414a91edba15803df5ee1f1e5e2e3cf3e7cdeb9f7ea30'
PREREG_SHA='54e75b181e9054eb479991589dd8b84cc6317b404ae88f24ee57e6a34a7311ae'
CPP_HELPER_SHA='c38232636990570a125d5c6e0bfdda91cc7552d18914be99afca5e4ad3408fd4'
GUARD_PATH=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
DP1=ROOT/'집/클로드/research/ec3_DP1_daily_operation_pattern_v1.py'
EXPECTED_DEPS=dict(core='057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2',
 season='7d58feeb6a0e7653796a6775f762b659cab3e64078b299d36b52ae69faeff162',
 env='82ca7b4bda2e9f50069b53b92d8418a7c0d4dd5ce7127fade16dd6ea78a8dcd2',
 support='82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed',
 original_guard='e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9')
RUNTIME=dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0')
SEEDS=(7,101,2024); VALIDATORS=('DIAG10','A','B','EXT10','EXT12')
FOLD_KEYS=tuple((v,k) for v,n in zip(VALIDATORS,(10,5,5,1,1)) for k in range(n))
FAMILY=24; ALPHA=.025/FAMILY; ATOL=1e-12
NEW=[]
CSV_COLUMNS=['row_id','farm','day','hour','y','baseline','candidate','new_lgb_raw','raw_et','raw_lgb','raw_mlp','old_pfn_raw','validator','fold','seed','clip_lo','clip_hi']
KEY_COLUMNS=['row_id','farm','day','hour','validator','fold','seed']
CHECKS=0; np=pd=env=S=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    def unique(items):
        d={}
        for k,v in items:
            assert k not in d,('duplicate JSON key',k)
            d[k]=v
        return d
    return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def array_sha(x):
    x=np.asarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ids_sha(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def compare(a,b,tol=ATOL):
    global CHECKS
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    gap=float(np.max(np.abs(a-b)));assert gap<=tol,(gap,tol)
    CHECKS+=a.size;return gap
def exact_keys(d,keys):assert isinstance(d,dict) and set(d)==set(keys),(set(d),set(keys))
def small_error(x):assert isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<=ATOL,x
def extract(path,names,ns):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==len(names) and {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def bootstrap(support=False):
    global np,pd,env,S
    sys.path.insert(0,str(ROOT/'집/클로드/research'));import env as e
    import numpy as n,pandas as p
    np,pd,env=n,p,e
    if support:
        sys.path.insert(0,str(H.parent/'statistical_experiments_20261003_v1'))
        import support as s
        S=s;assert sha(Path(S.__file__))==EXPECTED_DEPS['support']
def runtime():
    import sklearn, lightgbm
    assert sklearn.__version__==RUNTIME['sklearn'] and lightgbm.__version__==RUNTIME['lightgbm']
    found=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        sklearn=importlib.metadata.version('scikit-learn'),lightgbm=importlib.metadata.version('lightgbm'))
    assert found==RUNTIME,found;return found
def operations(lab):return lab.copy()

def library():
    import os,ctypes,lightgbm.basic as basic
    build=readj(H/'build_result_v3.json');assert build['status']=='PASS_BUILD_ONLY'
    dll=Path(build['dll']);assert sha(dll)==build['dll_sha256']=='25806c5faae08b55142c8d7670be8a88461604865eba353f7094f008e3c94ac8'
    assert sha(H/'overlay_manifest_v1.json')==build['overlay_manifest_sha256']
    overlay=readj(H/'overlay_manifest_v1.json')
    for name,item in overlay['changes'].items():assert sha(OUT/'overlay_v1'/name)==item['overlay_sha256']
    assert sha(OUT/'overlay_v1/src/objective/farmai_leaf_audit.hpp')==overlay['extra_header_sha256']
    tool=OUT/'toolchain_v1/mingw64/bin'
    assert sha(tool/'g++.exe')==build['compiler_sha256'] and sha(tool/'cmake.exe')==build['cmake_sha256']
    assert sha(OUT/'winlibs.zip')=='d5dbafc4a170e762ca6143151ec918fb9e2c72736fb14cd704abebc6bdd5276a'
    assert Path(build['alias']).resolve()==Path(build['alias_resolved_target']).resolve()==OUT.resolve()
    assert readj(H/'synthetic_result_v4.json')['status']=='PASS_SYNTHETIC_ONLY'
    original=Path(basic._LIB._name);assert sha(original)=='7e366d2e49cd061aac3ab21676b2f99b0c7a758dc3e888d4b23812af1b7d301c'
    global DLL_HANDLES
    DLL_HANDLES=[os.add_dll_directory(str(tool)),os.add_dll_directory(str(dll.parent))]
    basic._LIB=ctypes.cdll.LoadLibrary(str(dll));basic._LIB.LGBM_GetLastError.restype=ctypes.c_char_p
    assert Path(basic._LIB._name).resolve()==dll.resolve()
    return dict(dll=str(dll),dll_sha256=sha(dll),build_sha256=sha(H/'build_result_v3.json'),overlay_sha256=sha(H/'overlay_manifest_v1.json'),synthetic_sha256=sha(H/'synthetic_result_v4.json'),cpp_audit_source_sha256=sha(H/'first_cpp_audit_v2.py'))

def scalar_candidate(q,et,lgb,mlp,pfn,lo,hi):
    assert math.isfinite(float(lo)) and math.isfinite(float(hi)) and lo<=hi
    values={};result=np.empty(len(q))
    for i in sorted(range(len(q)),key=lambda i:(q.farm.iloc[i],int(q.day.iloc[i]),int(q.hour.iloc[i]))):
        raw=math.fsum([.48*float(et[i]),.24*float(lgb[i]),.08*float(mlp[i]),.2*float(pfn[i])])
        prior=values.setdefault((q.farm.iloc[i],int(q.day.iloc[i])),[]);prior.append(raw)
        result[i]=min(float(hi),max(float(lo),.5*raw+.5*math.fsum(prior)/len(prior)))
    return result
def validate_first(first,signature,lab,q):
    exact_keys(first,{'status','original_lgb_maxdiff','raw_errors','scalar_maxdiff','prefix','feature_causal','atol','signature','cpp_audit','cpp_trace_sha256'})
    assert first['status']=='PASS' and first['signature']==signature and first['atol']==ATOL
    assert (signature['validator'],signature['fold'],signature['seed'])==('DIAG10',0,7)
    assert first['cpp_audit']['status']=='PASS' and first['cpp_trace_sha256']==sha(OUT/'first_exact_cpp_trace_v1.ndjson')
    small_error(first['original_lgb_maxdiff']);small_error(first['scalar_maxdiff'])
    exact_keys(first['raw_errors'],{'repeat','fresh_fit','single','reversed','other_query'})
    for x in first['raw_errors'].values():small_error(x)
    for key,frame,fields in [('prefix',q,['raw_maxdiff','final_maxdiff']),('feature_causal',lab,['feature_maxdiff'])]:
        records=first[key];assert isinstance(records,list) and len(records)==6
        expected=[(f,int(frame.loc[frame.farm==f,'day'].min()),h) for f in ['F13','F47'] for h in [0,6,12]]
        assert [(r['farm'],r['day'],r['hour']) for r in records]==expected
        for r in records:
            exact_keys(r,['farm','day','hour']+fields)
            assert type(r['day']) is int and type(r['hour']) is int
            assert len(frame[(frame.farm==r['farm'])&(frame.day==r['day'])&(frame.hour<=r['hour'])])==r['hour']+1
            for c in fields:small_error(r[c])
def validate_meta(meta,signature,csv_digest,first_digest=None):
    keys={'status','signature','csv_sha256','fit_predict_seconds','first_audit_sha256','model_sha256'}
    if first_digest is not None:keys.add('first_audit_sha256')
    exact_keys(meta,keys)
    assert meta['status']=='PASS' and meta['signature']==signature and meta['csv_sha256']==csv_digest
    t=meta['fit_predict_seconds'];assert isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) and t>=0
    if first_digest is not None:assert meta['first_audit_sha256']==first_digest
def validate_csv(d,q,v,k,seed):
    assert list(d.columns)==CSV_COLUMNS and len(d)==len(q) and d.row_id.is_unique
    assert np.array_equal(d.row_id,q.row_id)
    for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
    for c,val in [('validator',v),('fold',k),('seed',seed)]:assert (d[c]==val).all()
    for c in ['day','hour','fold','seed']:assert pd.api.types.is_integer_dtype(d[c].dtype)
    assert np.isfinite(d[[c for c in CSV_COLUMNS if c not in KEY_COLUMNS]].to_numpy(float)).all()
def compare_frames(a,b):
    assert list(a.columns)==list(b.columns)==CSV_COLUMNS and len(a)==len(b)
    for c in CSV_COLUMNS:
        if c in KEY_COLUMNS:assert np.array_equal(a[c],b[c])
        else:compare(a[c],b[c])
def rmse_pair(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float);assert y.shape==p.shape and y.size
    assert np.isfinite(y).all() and np.isfinite(p).all()
    scalar=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))
    vector=float(np.sqrt(np.mean((y-p)**2)));compare([scalar],[vector]);return scalar,vector
def summary_boot(sample):
    assert sample.shape==(20000,) and np.isfinite(sample).all()
    return dict(p_worse=float(np.mean(sample>=0)),ci_adjusted=np.quantile(sample,[ALPHA,1-ALPHA],method='linear').tolist(),ci95=np.quantile(sample,[.025,.975],method='linear').tolist())
def boot(d,seed):
    assert seed in SEEDS and set(d.farm)=={'F13','F47'} and len(d)==8640
    assert len(d[['farm','day']].drop_duplicates())==360
    assert all(set(g.hour)==set(range(24)) and len(g)==24 for _,g in d.groupby(['farm','day']))
    daily=d.assign(loss_diff=(d.y-d.candidate)**2-(d.y-d.baseline)**2).groupby(['farm','day'],sort=True).loss_diff.agg(['sum','count'])
    rng=np.random.default_rng(20261003+seed);total=np.zeros(20000);count=np.zeros(20000);block_counts={}
    for farm in ['F13','F47']:
        g=daily.loc[farm].sort_index();blocks=[g.iloc[i:i+5] for i in range(0,len(g),5)]
        a=np.array([x['sum'].sum() for x in blocks]);n=np.array([x['count'].sum() for x in blocks]);block_counts[farm]=len(blocks)
        ix=rng.integers(len(a),size=(20000,len(a)));total+=a[ix].sum(axis=1);count+=n[ix].sum(axis=1)
    sample=total/count;records={}
    for r in d.itertuples():records.setdefault((r.farm,int(r.day)),[]).append((r.y-r.candidate)**2-(r.y-r.baseline)**2)
    rng=np.random.default_rng(20261003+seed);ss=np.zeros(20000);nn=np.zeros(20000)
    for farm in ['F13','F47']:
        days=sorted(day for f,day in records if f==farm)
        blocks=[[x for day in days[i:i+5] for x in records[farm,day]] for i in range(0,len(days),5)]
        a=np.array([math.fsum(b) for b in blocks]);n=np.array([len(b) for b in blocks]);assert len(blocks)==block_counts[farm]
        ix=rng.integers(len(a),size=(20000,len(a)));ss+=a[ix].sum(axis=1);nn+=n[ix].sum(axis=1)
    compare(count,nn);compare(sample,ss/nn)
    a,b=summary_boot(sample),summary_boot(ss/nn)
    assert a['p_worse']==b['p_worse']
    compare(a['ci_adjusted'],b['ci_adjusted']);compare(a['ci95'],b['ci95'])
    assert (a['p_worse']<ALPHA and a['ci_adjusted'][1]<0)==(b['p_worse']<ALPHA and b['ci_adjusted'][1]<0)
    return dict(**a,independent=b,draws=20000,rng_seed=20261003+seed,block_days=5,blocks=block_counts,quantile_method='linear',adjusted_quantiles=[ALPHA,1-ALPHA])

def verify():
    global CHECKS
    CHECKS=0
    destinations=[H/'full_verification_v2.json',H/'full_scores_v2.csv',H/'full_segments_v2.csv']
    assert all(not p.exists() for p in destinations),'Preserve existing outputs; create a new version'
    for digest in [RUN_SHA,PREP_SHA,PREREG_SHA]:assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Final pins not fixed'
    assert sha(H/RUN_FILE)==RUN_SHA and sha(H/PREP_FILE)==PREP_SHA and sha(H/'preregistration_v1.md')==PREREG_SHA
    assert sha(H/'whole_cpp_recheck_v1.py')==CPP_HELPER_SHA
    prepared=readj(H/PREP_FILE);fit=readj(H/'fit_audit_v1.json');first_path=H/'first_fold_verification_v1.json';first=readj(first_path)
    exact_keys(prepared,{'status','family','alpha','fit_count','predict_count','score_count','runtime','dependencies','inputs','manifest','original_r3_guard','params','threadpool_limits'})
    assert prepared['status']=='PASS' and prepared['family']==FAMILY and prepared['alpha']==ALPHA
    assert all(prepared[n]==0 for n in ['fit_count','predict_count','score_count']) and prepared['threadpool_limits']==2
    exact_keys(fit,{'status','family','preparation','cells','score_count','compiled_newton_all66_sha256'})
    assert fit['status']=='PASS' and fit['family']==FAMILY and fit['preparation']==prepared and fit['score_count']==0 and len(fit['cells'])==66
    bootstrap(True);rt=runtime();assert prepared['runtime']==rt
    lib=library()
    controls=readj(H/'compiled_newton_all66_v1.json')
    assert controls['status']=='PASS' and controls['atol']==ATOL and controls['score_count']==0 and len(controls['cells'])==66
    assert fit['compiled_newton_all66_sha256']==sha(H/'compiled_newton_all66_v1.json')
    lab,core,wv,folds,outer=S.loadec()
    deps=dict(EXPECTED_DEPS,run=RUN_SHA)
    found=dict(core=sha(Path(core.__file__)),support=sha(Path(S.__file__)),season=sha(Path(sys.modules[S.mapping.__module__].__file__)),env=sha(Path(env.__file__)),original_guard=sha(GUARD_PATH),run=sha(H/RUN_FILE))
    assert found==deps==prepared['dependencies']
    inputs=dict(train_X=sha(Path(env.DATA)/'train_X.csv'),public_oof=sha(OLD/'v2_integration_oof.csv'));assert inputs==prepared['inputs']
    assert len(lab)==8640 and lab.row_id.is_unique and [(v,k) for v,k,_,_ in folds]==list(FOLD_KEYS)
    lab=operations(lab);labels=outer[(outer.validator=='DIAG10')&(outer.seed==7)].set_index('row_id').sub_ec
    assert len(labels)==8640 and labels.index.is_unique;compare(lab.sub_ec,labels.reindex(lab.row_id))
    fs=[c for c in core.FULL if c!='day']+['season'];bs=[c for c in core.BASE if c!='day']+['season'];cols=bs+NEW
    assert len(fs)==38 and len(bs)==14 and len(cols)==14 and len(set(cols))==14 and bs[-1]=='season' and 'day' not in cols
    # Creating fixed recipe instances to read params only: fit/predict never invoked.
    params={str(s):core.lg(s,'tweedie').get_params() for s in SEEDS};assert params==prepared['params']
    guard=extract(GUARD_PATH,{'guard_original_r3'},dict(np=np,hashlib=hashlib,json=json,compare=compare,ids_sha=ids_sha,array_sha=array_sha))['guard_original_r3']
    expected={(v,k,s) for v,k in FOLD_KEYS for s in SEEDS}
    assert {p.name for p in OUT.glob('*_pred.csv')}=={f'{v}_{k}_{s}_pred.csv' for v,k,s in expected}
    assert {p.name for p in OUT.glob('*.json')}==({f'{v}_{k}_{s}.json' for v,k,s in expected}|{f'{v}_{k}_{s}_newton_control.json' for v,k,s in expected})
    assert {p.name for p in OUT.glob('*_model.txt')}=={f'{v}_{k}_{s}_model.txt' for v,k,s in expected}
    frames=[];metas=[];manifest=[];guards=[];scalar_error=0.
    for i,(v,k,tm,vm) in enumerate(folds):
        tr,q=S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
        banned={(f,int(d)+j) for f,d in q[['farm','day']].itertuples(index=False,name=None) for j in [-1,0,1]}
        assert not set(tr[['farm','day']].itertuples(index=False,name=None))&banned
        assert not set(tr.row_id)&set(q.row_id) and tr.row_id.is_unique and q.row_id.is_unique
        p=BASE/f'{v}_{k}_baseline.npz';old=dict(np.load(p,allow_pickle=False));cache_hashes={str(p.relative_to(ROOT)):sha(p)}
        assert np.array_equal(old['row_id'],q.row_id);compare([old['lo'],old['hi']],[tr.sub_ec.min(),tr.sub_ec.max()]);assert old['lo']<=old['hi']
        pfns=[]
        for c in [1,2,3,4]:
            p=OLD/f'{v}_{k}_pfn_{c}.npz';bag=dict(np.load(p,allow_pickle=False));cache_hashes[str(p.relative_to(ROOT))]=sha(p)
            assert np.array_equal(bag['row_id'],q.row_id) and np.array_equal(bag['context_row_id'],old[f'context_{c}'])
            assert len(set(bag['context_row_id']))==len(bag['context_row_id']) and set(bag['context_row_id'])<=set(tr.row_id)
            compare(bag['sub_ec'],q.sub_ec);assert bag['raw_pfn'].shape==(len(q),) and np.isfinite(bag['raw_pfn']).all();pfns.append(bag['raw_pfn'])
        compare(old['old_pfn_raw'],np.mean(pfns,axis=0));r3s={}
        for s in SEEDS:
            p=OLD/f'{v}_{k}_r3_{s}.npz';meta=readj(p.with_suffix('.json'));r3=dict(np.load(p,allow_pickle=False))
            prov=meta['provenance'];assert prov['feature_order']=='remove day; append season'
            assert prov['shared']['input_sha256']['train_X.csv']==inputs['train_X'] and prov['shared']['core_sha256']==deps['core']
            assert all(prov['environment'][n]==value for n,value in rt.items())
            g=guard(r3,meta,sha(p),tr,q,s,v,k,labels);guards.append(g)
            cache_hashes[str(p.relative_to(ROOT))]=sha(p);cache_hashes[str(p.with_suffix('.json').relative_to(ROOT))]=sha(p.with_suffix('.json'))
            compare(old[f'r3_{s}'],r3['raw_r3']);compare(old[f'baseline_{s}'],np.clip(core.shrink(.8*r3['raw_r3']+.2*old['old_pfn_raw'],q),old['lo'],old['hi']))
            r3s[s]=r3
        sig=dict(library=lib,validator=v,fold=k,runtime=rt,dependencies=deps,inputs=inputs,cache_hashes=cache_hashes,full_features=fs,old_lgb_features=bs,new_lgb_features=cols,train_ids=ids_sha(tr.row_id),query_ids=ids_sha(q.row_id),bounds=[float(old['lo']),float(old['hi'])])
        for part,d in [('train',tr),('query',q)]:
            sig[part+'_targets']=array_sha(d.sub_ec.to_numpy(float))
            for name,c in [('full38',fs),('old14',bs),('new14',cols)]:sig[part+'_'+name]=array_sha(d[c].to_numpy(float))
        assert sig==prepared['manifest'][i];manifest.append(sig)
        for s in SEEDS:
            p=OUT/f'{v}_{k}_{s}_pred.csv';meta=readj(OUT/f'{v}_{k}_{s}.json')
            signature=dict(**sig,seed=s,preparation_sha256=PREP_SHA,preregistration_sha256=PREREG_SHA)
            is_first=(v,k,s)==('DIAG10',0,7)
            validate_meta(meta,signature,sha(p),sha(first_path))
            checkpoint=OUT/f'{v}_{k}_{s}_model.txt';assert meta['model_sha256']==sha(checkpoint)
            control=readj(OUT/f'{v}_{k}_{s}_newton_control.json');npz=OUT/f'{v}_{k}_{s}_newton_control.npz'
            assert control['status']=='PASS' and control['signature']==sig and control['seed']==s and control['npz_sha256']==sha(npz)
            assert control['preparation_sha256']==PREP_SHA and control['preregistration_sha256']==PREREG_SHA
            small_error(control['max_error']);assert control==controls['cells'][i*3+SEEDS.index(s)]
            cached=dict(np.load(npz,allow_pickle=False));assert np.array_equal(cached['row_id'],q.row_id);compare(cached['raw'],r3s[s]['raw_lgb'])
            import lightgbm
            model=lightgbm.Booster(model_file=str(checkpoint));assert model.dump_model()['objective']=='tweedie_exact_leaf rho:1.5 lambda_l2:1'
            if is_first:
                validate_first(first,signature,lab,q)
                from whole_cpp_recheck_v1 import independent_cpp
                cpp_independent=independent_cpp(OUT/'first_exact_cpp_trace_v1.ndjson',model,tr,bs)

            d=pd.read_csv(p,float_precision='round_trip');validate_csv(d,q,v,k,s)
            compare(d.new_lgb_raw,model.predict(q[bs]));del model
            compare(d.y,q.sub_ec);compare(d.y,labels.reindex(d.row_id));r3=r3s[s]
            baseline=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==s)].set_index('row_id').season_v2.reindex(q.row_id)
            compare(d.baseline,baseline);compare(d.baseline,old[f'baseline_{s}'])
            for name in ['raw_et','raw_lgb','raw_mlp']:compare(d[name],r3[name])
            compare(d.old_pfn_raw,old['old_pfn_raw']);compare(d.clip_lo,np.repeat(old['lo'],len(q)));compare(d.clip_hi,np.repeat(old['hi'],len(q)))
            raw=.8*(.6*r3['raw_et']+.3*d.new_lgb_raw.to_numpy()+.1*r3['raw_mlp'])+.2*old['old_pfn_raw']
            compare(raw-(.8*r3['raw_r3']+.2*old['old_pfn_raw']),.24*(d.new_lgb_raw.to_numpy()-r3['raw_lgb']))
            compare(d.candidate,np.clip(core.shrink(raw,q),old['lo'],old['hi']))
            scalar_error=max(scalar_error,compare(d.candidate,scalar_candidate(q,r3['raw_et'],d.new_lgb_raw.to_numpy(),r3['raw_mlp'],old['old_pfn_raw'],old['lo'],old['hi'])))
            compare(d.baseline,scalar_candidate(q,r3['raw_et'],r3['raw_lgb'],r3['raw_mlp'],old['old_pfn_raw'],old['lo'],old['hi']))
            frames.append(d);metas.append(meta)
    assert manifest==prepared['manifest'] and len(manifest)==22 and metas==fit['cells'] and guards==prepared['original_r3_guard'] and len(guards)==66
    all_rows=pd.concat(frames,ignore_index=True);aggregate=pd.read_csv(OUT/'oof.csv',float_precision='round_trip')
    assert len(all_rows)==len(aggregate)==83160 and not all_rows.duplicated(['validator','fold','seed','row_id']).any()
    assert set(zip(all_rows.validator,all_rows.fold,all_rows.seed))==expected;compare_frames(all_rows,aggregate)
    # ALL source/runtime/first/manifest/66cells/aggregate guards above finish BEFORE any score.
    scores=[];segments=[];boots={}
    for v in VALIDATORS:
        for s in SEEDS:
            d=all_rows[(all_rows.validator==v)&(all_rows.seed==s)]
            rb,nb=rmse_pair(d.y,d.baseline);rc,nc=rmse_pair(d.y,d.candidate)
            assert (rc<rb)==(nc<nb) and rb>0
            scores.append(dict(validator=v,seed=s,n=len(d),baseline=rb,candidate=rc,baseline_numpy=nb,candidate_numpy=nc,delta_rmse=rc-rb,change_pct=100*(rc/rb-1)))
    for s in SEEDS:
        d=all_rows[(all_rows.validator=='DIAG10')&(all_rows.seed==s)];boots[s]=boot(d,s)
        high=d.groupby(['farm','day']).y.transform('mean')>=1
        assert high.sum()==744 and len(d.loc[high,['farm','day']].drop_duplicates())==31 and len(d.loc[~high,['farm','day']].drop_duplicates())==329
        for name,mask in [('high',high),('ordinary',~high),('late',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47'),('hour0',d.hour==0)]:
            g=d[mask];rb,_=rmse_pair(g.y,g.baseline);rc,_=rmse_pair(g.y,g.candidate)
            bb=math.fsum(g.baseline-g.y)/len(g);cb=math.fsum(g.candidate-g.y)/len(g);compare([bb,cb],[(g.baseline-g.y).mean(),(g.candidate-g.y).mean()])
            segments.append(dict(seed=s,segment=name,n=len(g),days=len(g[['farm','day']].drop_duplicates()),baseline=rb,candidate=rc,baseline_bias=bb,candidate_bias=cb))
    sc=pd.DataFrame(scores);assert len(sc)==15 and set(zip(sc.validator,sc.seed))=={(v,s) for v in VALIDATORS for s in SEEDS}
    passed=bool((sc.delta_rmse<0).all() and all(b['p_worse']<ALPHA and b['ci_adjusted'][1]<0 for b in boots.values()))
    for table,target in [(sc,destinations[1]),(pd.DataFrame(segments),destinations[2])]:
        with target.open('x',encoding='utf-8',newline='') as handle:table.to_csv(handle,index=False)
    result=dict(status='PASS',decision='PUBLIC_PASS_PENDING_REVIEW' if passed else 'REJECT',public_pass=passed,family=FAMILY,alpha=ALPHA,rows=83160,cells=66,score_cells=15,checks=CHECKS,bootstrap=boots,scalar_maxdiff=scalar_error,dependencies=deps,runtime=rt,inputs=inputs,manifest=manifest,r3_cache_audit=guards,
        verifier_sha256=sha(Path(__file__)),runner_sha256=RUN_SHA,preparation_sha256=PREP_SHA,preregistration_sha256=PREREG_SHA,fit_audit_sha256=sha(H/'fit_audit_v1.json'),first_audit_sha256=sha(first_path),aggregate_sha256=sha(OUT/'oof.csv'),scores_sha256=sha(destinations[1]),segments_sha256=sha(destinations[2]),verification_fit=0,verification_predict=68,cpp_independent=cpp_independent,raw_ec_target_reads=0,test_reads=0,EL1_rescore=0,
        limitations=['Repeated public validation; not untouched holdout','First tree trace independently checked; every model checkpoint predicted without retraining','SHA/source/manifest replay does not independently prove every fold causal','Preregistration timing requires parent git-registration evidence','Partial output write failures are preserved and require new output version'])
    with destinations[0].open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(sc.to_string(index=False));print(result['decision'])

def synthetic():
    global CHECKS
    CHECKS=0;bootstrap(False)
    q=pd.DataFrame(dict(row_id=[f'F13_{d:03d}_{h:02d}' for d in [1,2] for h in range(24)]+[f'F47_{d:03d}_{h:02d}' for d in [1,2] for h in range(24)],farm=['F13']*48+['F47']*48,day=[1]*24+[2]*24+[1]*24+[2]*24,hour=list(range(24))*4))
    rng=np.random.default_rng(21);et,lgb,mlp,pfn=rng.normal(size=(4,len(q)));raw=.8*(.6*et+.3*lgb+.1*mlp)+.2*pfn
    avg=pd.Series(raw).groupby([q.farm,q.day]).expanding().mean().reset_index(level=[0,1],drop=True).sort_index().to_numpy()
    compare(np.clip(.5*raw+.5*avg,-.2,.4),scalar_candidate(q,et,lgb,mlp,pfn,-.2,.4))
    signature=dict(validator='DIAG10',fold=0,seed=7)
    record=dict(cpp_audit={'status':'PASS'},cpp_trace_sha256='MOCK',status='PASS',original_lgb_maxdiff=0.,raw_errors={k:0. for k in ['repeat','fresh_fit','single','reversed','other_query']},scalar_maxdiff=0.,prefix=[dict(farm=f,day=1,hour=h,raw_maxdiff=0.,final_maxdiff=0.) for f in ['F13','F47'] for h in [0,6,12]],feature_causal=[dict(farm=f,day=1,hour=h,feature_maxdiff=0.) for f in ['F13','F47'] for h in [0,6,12]],atol=ATOL,signature=signature)
    original_sha=globals()['sha'];globals()['sha']=lambda p:'MOCK'
    validate_first(record,signature,q,q);rejected=0
    import copy
    for kind in ['nan','negative','over','day','missing','signature','extra']:
        bad=copy.deepcopy(record)
        if kind in ['nan','negative','over']:bad['raw_errors']['single']={'nan':float('nan'),'negative':-1.,'over':1e-9}[kind]
        elif kind=='day':bad['prefix'][0]['day']=2
        elif kind=='missing':del bad['raw_errors']['single']
        elif kind=='signature':bad['signature']['seed']=101
        else:bad['extra']=0
        try:validate_first(bad,signature,q,q)
        except (AssertionError,KeyError):rejected+=1
        else:raise AssertionError(('corruption accepted',kind))
    globals()['sha']=original_sha
    d=q.copy()
    for c in CSV_COLUMNS:
        if c not in d:d[c]=0.
    d['validator']='DIAG10';d['fold']=0;d['seed']=7;d=d[CSV_COLUMNS];validate_csv(d,q,'DIAG10',0,7)
    for kind in ['extra','duplicate_id','float_key','nonfinite','aggregate_value','aggregate_order']:
        bad=d.copy()
        if kind=='extra':bad['extra']=0
        elif kind=='duplicate_id':bad.loc[1,'row_id']=bad.loc[0,'row_id']
        elif kind=='float_key':bad['hour']=bad.hour.astype(float)
        elif kind=='nonfinite':bad.loc[1,'new_lgb_raw']=np.inf
        elif kind=='aggregate_value':bad.loc[0,'candidate']=1.
        else:bad=bad.iloc[::-1].reset_index(drop=True)
        try:
            if kind.startswith('aggregate'):compare_frames(d,bad)
            else:validate_csv(bad,q,'DIAG10',0,7)
        except (AssertionError,KeyError):rejected+=1
        else:raise AssertionError(('corruption accepted',kind))
    bootq=pd.DataFrame(dict(farm=['F13']*4320+['F47']*4320,day=np.tile(np.repeat(np.arange(180),24),2),hour=np.tile(np.arange(24),360),y=np.ones(8640),baseline=np.zeros(8640),candidate=np.full(8640,.1)))
    b=boot(bootq,7);assert b['p_worse']==0. and b['ci_adjusted'][1]<0
    target=H/'synthetic_verify_full_v2.json'
    with target.open('x',encoding='utf-8') as handle:json.dump(dict(status='PASS_SYNTHETIC_ONLY',corruption_rejected=rejected,checks=CHECKS,bootstrap=b,fit=0,predict=0,actual_candidate_reads=0,real_score=0),handle,indent=2)
    print('PASS_SYNTHETIC_ONLY',rejected,CHECKS)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();assert args.synthetic!=args.verify
    synthetic() if args.synthetic else verify()
