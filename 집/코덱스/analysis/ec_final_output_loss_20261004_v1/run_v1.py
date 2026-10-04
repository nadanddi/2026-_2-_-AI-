"""Family22 FINAL_LOSS /23 RAW_LOSS: zero-head residual MLP.
--prepare: public/source/cache audit and b-only preprocessing; no EC model training.
--synthetic: artificial arithmetic/autograd/model training only.
Default actual training requires parent preregistration and exact preparation replay.
"""
from pathlib import Path
import sys,json,hashlib,ast,argparse,platform,math,time,os,gc
sys.dont_write_bytecode=True
if sys.flags.optimize:raise RuntimeError('Audit requires assertions')
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
OUT=ROOT/'집/코덱스/local'/H.name
INNER=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
IH=H.parent/'ec_matched_inner_calibration_20261003_v1'
N_PATH=H.parent/'ec_nested_high_specialist_20261004_v1/run_v2.py'
G_PATH=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
OLD=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
BASE=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
SEEDS=(7,101,2024);MODES=('FINAL_LOSS','RAW_LOSS');FAMILIES=dict(FINAL_LOSS=22,RAW_LOSS=23)
FOLD_KEYS=tuple((v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n))
ALPHA=.025/24;ATOL=1e-12
CSV_COLUMNS=['row_id','farm','day','hour','y','baseline','candidate','base_raw','delta','validator','fold','seed','clip_lo','clip_hi','mode']
CONFIG=dict(modes=list(MODES),families=FAMILIES,alpha=ALPHA,features='original BASE: remove day; append season; exactly14',hidden=[64,32],activation='ReLU',head='zero weight and bias at initialization',delta='unbounded additive residual to complete raw mixture',dtype='float64',device='cpu',optimizer='torch.optim.Adam coupled weight_decay',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,amsgrad=False,foreach=False,fused=False,capturable=False,maximize=False,differentiable=False,epochs=400,batch='full inner-query b',loss='mean squared error; FINAL clip(shrink(base_raw+delta)), RAW base_raw+delta',preprocessing='SimpleImputer median keep_empty_features=True -> StandardScaler; fit b only',train_target='inner-query b public hourly sub_ec',train_bounds='inner-training a target min/max',query_bounds='outer-training tr target min/max',query_base='unchanged cached original .8R3+.2PFN4bag',train_base='matched-inner heldout .8R3+.2PFN4bag',inner_selection='existing full_inner single heldout subset; no claim complete crossfit of all outer rows',threads=2,deterministic=True,selection='last step400 only; no validation stopping/best epoch',strict_15=True,bootstrap='DIAG each seed farm sortedday5 block20k rng20261003+seed adjusted quantile[alpha,1-alpha]')
PINS=dict(support='82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed',core='057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2',env='82ca7b4bda2e9f50069b53b92d8418a7c0d4dd5ce7127fade16dd6ea78a8dcd2',env_extra='8b83d9e2651b3ac2787db59042267951f42b7f5e098a91d90785dc81dd6cbffb',season='7d58feeb6a0e7653796a6775f762b659cab3e64078b299d36b52ae69faeff162',nested_source='0ceb89e38692f76466000a4b3a83e056c99a04df10c519ddba59e70a55c2bbaf',matched_source='b1041713e093ca9b8f17437e05b194f284a7e44e8a681a06ce033bc6fc8a7e40',original_guard='e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9')
RUNTIME=dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0',torch='2.14.0+cpu')
np=pd=torch=env=env_extra=S=sklearn=lightgbm=None
SimpleImputer=StandardScaler=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    def unique(items):
        d={}
        for k,v in items:assert k not in d;d[k]=v
        return d
    return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def savej(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def arrayfile(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def array_sha(x):
    x=np.asarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ids_sha(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def compare(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    gap=float(np.max(abs(a-b)));assert gap<=ATOL,(gap,ATOL);return gap
def extract(path,names,ns):
    nodes=[x for x in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==len(names) and {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def bootstrap(support=False):
    global np,pd,torch,env,env_extra,S,SimpleImputer,StandardScaler,sklearn,lightgbm
    sys.path.insert(0,str(ROOT/'집/클로드/research'));import env as e;import env_extra as ex
    import numpy as n,pandas as p,torch as t,sklearn as sk,lightgbm as lg
    from sklearn.impute import SimpleImputer as imp
    from sklearn.preprocessing import StandardScaler as scaler
    env,env_extra,np,pd,torch,sklearn,lightgbm=e,ex,n,p,t,sk,lg;SimpleImputer,StandardScaler=imp,scaler
    torch.set_num_threads(CONFIG['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if support:
        sys.path.insert(0,str(H.parent/'statistical_experiments_20261003_v1'));import support as s
        S=s;assert sha(Path(S.__file__))==PINS['support']
def runtime():
    found=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,torch=torch.__version__)
    assert found==RUNTIME,found
    return dict(versions=found,module_paths={name:str(Path(mod.__file__).resolve()) for name,mod in [('numpy',np),('pandas',pd),('sklearn',sklearn),('lightgbm',lightgbm),('torch',torch)]},module_init_sha256={name:sha(Path(mod.__file__)) for name,mod in [('numpy',np),('pandas',pd),('sklearn',sklearn),('lightgbm',lightgbm),('torch',torch)]},torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),deterministic=torch.are_deterministic_algorithms_enabled())
def ordered_groups(frame):
    assert frame.row_id.is_unique
    groups=[]
    for _,g in frame.reset_index(drop=True).groupby(['farm','day'],sort=True):
        g=g.sort_values('hour');assert np.array_equal(g.hour.to_numpy(),np.arange(24))
        groups.append(g.index.to_numpy(int))
    assert sum(map(len,groups))==len(frame);return groups
def smooth_numpy(raw,frame):
    out=np.empty(len(raw))
    for ix in ordered_groups(frame):out[ix]=.5*raw[ix]+.5*np.cumsum(raw[ix])/np.arange(1,25)
    return out
def smooth_scalar(raw,frame):
    out=np.empty(len(raw))
    for ix in ordered_groups(frame):
        history=[]
        for i in ix:history.append(float(raw[i]));out[i]=.5*float(raw[i])+.5*math.fsum(history)/len(history)
    return out
def smooth_tensor(raw,groups):
    # scatter writes preserve autograd while grouping is based on fixed input IDs/hours.
    out=torch.empty_like(raw)
    for ix in groups:
        j=torch.as_tensor(ix,dtype=torch.long);v=raw[j]
        out[j]=.5*v+.5*torch.cumsum(v,dim=0)/torch.arange(1,len(ix)+1,dtype=torch.float64)
    return out
def final(raw,frame,bounds):return np.clip(smooth_numpy(raw,frame),*bounds)
def preprocessing(b,cols):
    x=b[cols].to_numpy(float);assert not np.isinf(x).any()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);filled=imputer.fit_transform(x)
    scaler=StandardScaler();xb=scaler.fit_transform(filled)
    assert xb.shape==(len(b),14) and np.isfinite(xb).all()
    stats=dict(imputer_statistics=imputer.statistics_.copy(),scaler_mean=scaler.mean_.copy(),scaler_scale=scaler.scale_.copy(),scaler_var=scaler.var_.copy(),scaler_n_samples_seen=np.asarray(scaler.n_samples_seen_))
    return stats,xb
def transform(frame,cols,stats):
    x=frame[cols].to_numpy(float).copy();assert not np.isinf(x).any()
    missing=np.isnan(x);x[missing]=np.broadcast_to(stats['imputer_statistics'],x.shape)[missing]
    x=(x-stats['scaler_mean'])/stats['scaler_scale'];assert np.isfinite(x).all();return x
def make_model(seed):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(14,64,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(64,32,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(32,1,dtype=torch.float64))
    with torch.no_grad():net[4].weight.zero_();net[4].bias.zero_()
    assert all(p.dtype==torch.float64 and p.device.type=='cpu' for p in net.parameters());return net
def predict(net,x):
    net.eval()
    with torch.no_grad():out=net(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float64))).reshape(-1).numpy()
    assert out.shape==(len(x),) and np.isfinite(out).all();return out
def train(x,y,base_raw,frame,bounds,seed,mode,epochs=400):
    assert mode in MODES and epochs==CONFIG['epochs'];net=make_model(seed)
    xt=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float64));yt=torch.from_numpy(np.asarray(y,dtype=np.float64));bt=torch.from_numpy(np.asarray(base_raw,dtype=np.float64));groups=ordered_groups(frame)
    optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],betas=tuple(CONFIG['betas']),eps=CONFIG['eps'],amsgrad=False,foreach=False,fused=False,capturable=False,maximize=False,differentiable=False)
    def objective():
        raw=bt+net(xt).reshape(-1)
        pred=torch.clamp(smooth_tensor(raw,groups),min=bounds[0],max=bounds[1]) if mode=='FINAL_LOSS' else raw
        return torch.mean((pred-yt)**2)
    zero=predict(net,x);assert np.array_equal(zero,np.zeros(len(x)))
    with torch.no_grad():initial=float(objective())
    trace=[initial]
    net.train()
    for step in range(epochs):
        optimizer.zero_grad(set_to_none=True);loss=objective();assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step();assert all(torch.isfinite(p).all() for p in net.parameters())
        with torch.no_grad():trace.append(float(objective()))
    assert len(trace)==401 and all(math.isfinite(x) for x in trace)
    net.eval();return net,np.asarray(trace,float)
def checkpoint_payload(net,stats,cols,trace):
    payload=dict(stats);payload.update({f'model__{k}':v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()})
    payload.update(feature_names=np.asarray(cols,str),format=np.asarray('RESIDUAL_MLP_FLOAT64_V1'),loss_trace=trace)
    return payload
def load_checkpoint(path,seed,cols):
    payload=arrayfile(path);assert str(payload['format'])=='RESIDUAL_MLP_FLOAT64_V1' and np.array_equal(payload['feature_names'],np.asarray(cols,str))
    stats={k:payload[k] for k in ['imputer_statistics','scaler_mean','scaler_scale','scaler_var','scaler_n_samples_seen']}
    net=make_model(seed);expected={f'model__{k}' for k in net.state_dict()}
    assert set(payload)==set(stats)|expected|{'feature_names','format','loss_trace'}
    net.load_state_dict({k:torch.from_numpy(payload['model__'+k].copy()) for k in net.state_dict()},strict=True)
    assert payload['loss_trace'].shape==(401,) and np.isfinite(payload['loss_trace']).all()
    return net,stats,payload['loss_trace']
def guard_current(deps,inputs,caches):
    paths=dict(run=Path(__file__),support=Path(S.__file__),core=ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py',env=Path(env.__file__),env_extra=Path(env_extra.__file__),season=Path(sys.modules[S.mapping.__module__].__file__),nested_source=N_PATH,matched_source=IH/'run.py',original_guard=G_PATH)
    assert {k:sha(p) for k,p in paths.items()}==deps
    assert inputs==dict(train_X=sha(Path(env.DATA)/'train_X.csv'),public_oof=sha(OLD/'v2_integration_oof.csv'))
    for relative,digest in caches.items():assert sha(ROOT/relative)==digest,relative
def preflight():
    rt=runtime();lab,core,wv,folds,outer=S.loadec()
    assert len(lab)==8640 and lab.row_id.is_unique and [(v,k) for v,k,_,_ in folds]==list(FOLD_KEYS)
    paths=dict(run=Path(__file__),support=Path(S.__file__),core=Path(core.__file__),env=Path(env.__file__),env_extra=Path(env_extra.__file__),season=Path(sys.modules[S.mapping.__module__].__file__),nested_source=N_PATH,matched_source=IH/'run.py',original_guard=G_PATH)
    deps={k:sha(p) for k,p in paths.items()};assert {k:v for k,v in deps.items() if k!='run'}==PINS
    inputs=dict(train_X=sha(Path(env.DATA)/'train_X.csv'),public_oof=sha(OLD/'v2_integration_oof.csv'))
    assert (INNER/'source_sha.txt').read_text()==PINS['matched_source']
    fit=readj(IH/'fit_audit_v1.json');assert fit['status']=='PASS' and fit['checks']==22*3*24
    for filename,tol in [('cpu_reproduction_v1.json',1e-9),('pfn_reproduction_v1.json',1e-5)]:
        r=readj(IH/filename);assert r['status']=='PASS' and math.isfinite(r['maxdiff']) and 0<=r['maxdiff']<tol
    inner=extract(N_PATH,{'arrayfile','full_inner'},dict(np=np,S=S,INNER=INNER))['full_inner']
    original=extract(G_PATH,{'guard_original_r3'},dict(np=np,hashlib=hashlib,json=json,compare=compare,ids_sha=ids_sha,array_sha=array_sha))['guard_original_r3']
    labels=outer[(outer.validator=='DIAG10')&(outer.seed==7)].set_index('row_id').sub_ec;assert len(labels)==8640 and labels.index.is_unique;compare(lab.sub_ec,labels.reindex(lab.row_id))
    cols=[c for c in core.BASE if c!='day']+['season'];assert len(cols)==14 and cols[-1]=='season' and 'day' not in cols
    idx=lab.set_index('row_id');refs={};manifests=[];r3guards=[]
    for v,k,tm,vm in folds:
        rawtr=lab[tm].reset_index(drop=True);a,b,z,bag=inner(v,k,rawtr,idx)
        a,b=S.seasonal(a,b,wv);a,b=a.reset_index(drop=True),b.reset_index(drop=True)
        tr,q=S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
        for frame in [a,b,tr,q]:ordered_groups(frame)
        for trainframe,queryframe in [(a,b),(tr,q)]:
            forbidden={(f,int(d)+j) for f,d in queryframe[['farm','day']].itertuples(index=False,name=None) for j in [-1,0,1]}
            assert not set(trainframe[['farm','day']].itertuples(index=False,name=None))&forbidden
        assert set(a.row_id)|set(b.row_id)<=set(tr.row_id) and not (set(a.row_id)|set(b.row_id))&set(q.row_id)
        ip=(float(z['lo']),float(z['hi']));op=(float(tr.sub_ec.min()),float(tr.sub_ec.max()))
        compare(ip,[a.sub_ec.min(),a.sub_ec.max()])
        cache_paths=[INNER/f'{v}_{k}_cpu.npz',S.OUT/f'E_{v}_{k}_cpu.npz']+[INNER/f'{v}_{k}_pfn_{c}.npz' for c in [1,2,3,4]]+[IH/'fit_audit_v1.json',IH/'cpu_reproduction_v1.json',IH/'pfn_reproduction_v1.json',INNER/'source_sha.txt']
        base_path=BASE/f'{v}_{k}_baseline.npz';old=arrayfile(base_path);cache_paths.append(base_path)
        assert np.array_equal(old['row_id'],q.row_id);compare([old['lo'],old['hi']],op)
        pfns=[]
        for c in [1,2,3,4]:
            p=OLD/f'{v}_{k}_pfn_{c}.npz';cache_paths.append(p);pf=arrayfile(p)
            assert np.array_equal(pf['row_id'],q.row_id) and np.array_equal(pf['context_row_id'],old[f'context_{c}']) and set(pf['context_row_id'])<=set(tr.row_id)
            compare(pf['sub_ec'],q.sub_ec);pfns.append(pf['raw_pfn'])
        compare(old['old_pfn_raw'],np.mean(pfns,axis=0));trainbase={};querybase={};baseline={}
        for seed in SEEDS:
            p=OLD/f'{v}_{k}_r3_{seed}.npz';cache_paths.extend([p,p.with_suffix('.json')]);r3=arrayfile(p);m=readj(p.with_suffix('.json'));prov=m['provenance']
            assert prov['feature_order']=='remove day; append season' and prov['shared']['input_sha256']['train_X.csv']==inputs['train_X'] and prov['shared']['core_sha256']==deps['core']
            assert all(prov['environment'][n]==rt['versions'][n] for n in ['python','numpy','pandas','sklearn','lightgbm'])
            r3guards.append(original(r3,m,sha(p),tr,q,seed,v,k,labels));compare(old[f'r3_{seed}'],r3['raw_r3'])
            trainbase[seed]=.8*z[f'r3_{seed}']+.2*bag;querybase[seed]=.8*r3['raw_r3']+.2*old['old_pfn_raw']
            compare(final(trainbase[seed],b,ip),np.clip(core.shrink(trainbase[seed],b),*ip))
            baseline[seed]=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy(float)
            compare(baseline[seed],old[f'baseline_{seed}']);compare(baseline[seed],final(querybase[seed],q,op));compare(baseline[seed],np.clip(smooth_scalar(querybase[seed],q),*op))
        stats,xb=preprocessing(b,cols);compare(xb,transform(b,cols,stats))
        xq=transform(q,cols,stats);caches={str(p.relative_to(ROOT)):sha(p) for p in cache_paths}
        sig=dict(validator=v,fold=k,dependencies=deps,inputs=inputs,runtime=rt,config=CONFIG,features=cols,cache_hashes=caches,inner_bounds=list(ip),outer_bounds=list(op),preprocessing={name:array_sha(value) for name,value in stats.items()})
        for part,frame in [('inner_a',a),('inner_b',b),('outer_train',tr),('outer_query',q)]:
            sig[part+'_ids']=ids_sha(frame.row_id);sig[part+'_rows']=len(frame);sig[part+'_days']=len(frame[['farm','day']].drop_duplicates());sig[part+'_features']=array_sha(frame[cols].to_numpy(float));sig[part+'_targets']=array_sha(frame.sub_ec.to_numpy(float))
        sig['inner_b_transformed_features']=array_sha(xb);sig['outer_query_transformed_features']=array_sha(xq)
        sig['inner_base_raw']={str(seed):array_sha(trainbase[seed]) for seed in SEEDS};sig['outer_base_raw']={str(seed):array_sha(querybase[seed]) for seed in SEEDS};sig['outer_baseline']={str(seed):array_sha(baseline[seed]) for seed in SEEDS}
        refs[v,k]=dict(a=a,b=b,tr=tr,q=q,stats=stats,xb=xb,xq=xq,trainbase=trainbase,querybase=querybase,baseline=baseline,inner_bounds=ip,outer_bounds=op,signature=sig);manifests.append(sig)
    prepared=dict(status='PASS_PREPARATION_MODEL_FIT0_PREDICT0_SCORE0',source_sha256=sha(Path(__file__)),config=CONFIG,runtime=rt,dependencies=deps,inputs=inputs,features=cols,manifest=manifests,original_r3_guard=r3guards,model_fit_count=0,model_predict_count=0,real_score_count=0,preprocessing_fit_count=22,required_cpu=22,required_pfn=88,outer_cells_per_mode=66,limitations=['inner residual learns only existing heldout b subset, not all outer train rows','original saved inner reproduction receipt checked; no ET/MLP/PFN refit','source/manifest audit does not independently replay all causal input generation'])
    return refs,prepared,cols
def validate_csv(d,q,mode,v,k,seed):
    assert list(d.columns)==CSV_COLUMNS and d.row_id.is_unique and np.array_equal(d.row_id,q.row_id)
    for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
    for c,value in [('mode',mode),('validator',v),('fold',k),('seed',seed)]:assert (d[c]==value).all()
    assert np.isfinite(d[['y','baseline','candidate','base_raw','delta','clip_lo','clip_hi']].to_numpy(float)).all()
def audit_first(ref,net,mode,seed,trace,cols):
    q,b=ref['q'],ref['b'];xq=ref['xq'];rawbase=ref['querybase'][seed];op=ref['outer_bounds'];rawtrain=ref['trainbase'][seed];ip=ref['inner_bounds']
    zero=make_model(seed);zero_delta=predict(zero,xq);assert np.array_equal(zero_delta,np.zeros(len(q)))
    zero_query=compare(final(rawbase+zero_delta,q,op),ref['baseline'][seed]);compare(final(rawbase,q,op),np.clip(smooth_scalar(rawbase,q),*op))
    zero_inner=compare(final(rawtrain+predict(zero,ref['xb']),b,ip),np.clip(smooth_scalar(rawtrain,b),*ip))
    delta=predict(net,xq);pred=final(rawbase+delta,q,op)
    fresh,fresh_trace=train(ref['xb'],b.sub_ec.to_numpy(float),rawtrain,b,ip,seed,mode)
    errors=dict(repeat=compare(delta,predict(net,xq)),fresh=compare(delta,predict(fresh,xq)),reversed=compare(delta,predict(net,xq[::-1].copy())[::-1]),single=compare(delta[:8],np.array([predict(net,xq[i:i+1])[0] for i in range(8)])))
    changed=xq[:8].copy();changed[1:]+=10000;errors['other_query']=compare(delta[:1],predict(net,changed)[:1]);compare(trace,fresh_trace)
    scalar=compare(pred,np.clip(smooth_scalar(rawbase+delta,q),*op));prefix=[]
    for farm in ['F13','F47']:
        day=int(q.loc[q.farm==farm,'day'].min())
        for hour in [0,6,12]:
            mask=((q.farm==farm)&(q.day==day)&(q.hour<=hour)).to_numpy();qp=q[mask].reset_index(drop=True);dp=predict(net,xq[mask]);history=[];pp=[]
            for r in rawbase[mask]+dp:history.append(float(r));pp.append(np.clip(.5*r+.5*math.fsum(history)/len(history),*op))
            prefix.append(dict(farm=farm,day=day,hour=hour,rows=len(qp),raw_delta_maxdiff=compare(delta[mask],dp),final_maxdiff=compare(pred[mask],pp)))
    del fresh,zero
    return dict(status='PASS',atol=ATOL,zero_query_baseline_maxdiff=zero_query,zero_inner_baseline_maxdiff=zero_inner,errors=errors,scalar_maxdiff=scalar,prefix=prefix,loss_initial=float(trace[0]),loss_final=float(trace[-1]),fresh_loss_trace_maxdiff=compare(trace,fresh_trace),epochs=400,limitations=['audit fresh/repeat/order/prefix model; not full raw-feature causal regeneration'])
def validate_first(first,signature):
    assert first['status']=='PASS' and first['signature']==signature and first['atol']==ATOL and first['epochs']==400
    assert set(first['errors'])=={'repeat','fresh','reversed','single','other_query'}
    for x in list(first['errors'].values())+[first['zero_query_baseline_maxdiff'],first['zero_inner_baseline_maxdiff'],first['scalar_maxdiff'],first['fresh_loss_trace_maxdiff']]:assert math.isfinite(x) and 0<=x<=ATOL
    assert len(first['prefix'])==6
    for r in first['prefix']:
        assert r['rows']==r['hour']+1
        for c in ['raw_delta_maxdiff','final_maxdiff']:assert math.isfinite(r[c]) and 0<=r[c]<=ATOL
def actual(refs,prepared,cols):
    assert readj(H/'preparation_v1.json')==prepared and (H/'preregistration_v1.md').is_file()
    assert not (H/'fit_audit_v1.json').exists() and not (OUT/'oof.csv').exists(),'Preserve completed run'
    OUT.mkdir(parents=True,exist_ok=True);allframes=[];audits=[]
    for mode in MODES:
        modeframes=[]
        for v,k in FOLD_KEYS:
            ref=refs[v,k];q,b=ref['q'],ref['b']
            for seed in SEEDS:
                guard_current(prepared['dependencies'],prepared['inputs'],ref['signature']['cache_hashes']);assert readj(H/'preparation_v1.json')==prepared
                name=f'{mode}_{v}_{k}_{seed}';csvpath=OUT/(name+'_pred.csv');metapath=OUT/(name+'.json');ckpt=OUT/(name+'_model.npz');firstpath=H/f'first_{mode}_v1.json';is_first=(v,k,seed)==('DIAG10',0,7)
                signature=dict(**ref['signature'],seed=seed,mode=mode,family=FAMILIES[mode],preparation_sha256=sha(H/'preparation_v1.json'),preregistration_sha256=sha(H/'preregistration_v1.md'))
                files=[csvpath,metapath,ckpt]+([firstpath] if is_first else []);present=[p.exists() for p in files];assert not any(present) or all(present),'Partial output preserved: stop'
                if all(present):
                    meta=readj(metapath);assert meta['status']=='PASS' and meta['signature']==signature and meta['csv_sha256']==sha(csvpath) and meta['checkpoint_sha256']==sha(ckpt)
                    net,stats,trace=load_checkpoint(ckpt,seed,cols)
                    for key in stats:compare(stats[key],ref['stats'][key])
                    delta=predict(net,ref['xq']);d=pd.read_csv(csvpath,float_precision='round_trip');compare(d.delta,delta)
                    if is_first:assert meta['first_audit_sha256']==sha(firstpath);validate_first(readj(firstpath),signature)
                else:
                    started=time.perf_counter();net,trace=train(ref['xb'],b.sub_ec.to_numpy(float),ref['trainbase'][seed],b,ref['inner_bounds'],seed,mode)
                    delta=predict(net,ref['xq']);payload=checkpoint_payload(net,ref['stats'],cols,trace)
                    with ckpt.open('xb') as f:np.savez(f,**payload)
                    reloaded,stats,rt=load_checkpoint(ckpt,seed,cols);compare(delta,predict(reloaded,ref['xq']));compare(trace,rt);del reloaded
                    d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=ref['baseline'][seed];d['base_raw']=ref['querybase'][seed];d['delta']=delta;d['candidate']=final(d.base_raw.to_numpy()+delta,q,ref['outer_bounds']);d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=ref['outer_bounds'][0];d['clip_hi']=ref['outer_bounds'][1];d['mode']=mode;d=d[CSV_COLUMNS]
                    with csvpath.open('x',encoding='utf-8',newline='') as f:d.to_csv(f,index=False)
                    meta=dict(status='PASS',signature=signature,csv_sha256=sha(csvpath),checkpoint_sha256=sha(ckpt),train_seconds=time.perf_counter()-started,train=dict(epochs=400,loss_initial=float(trace[0]),loss_final=float(trace[-1]),loss_trace_sha256=array_sha(trace),training_rows=len(b),training_days=len(b[['farm','day']].drop_duplicates()),inner_bounds=list(ref['inner_bounds']),outer_bounds=list(ref['outer_bounds']),config=CONFIG))
                    if is_first:
                        first=dict(**audit_first(ref,net,mode,seed,trace,cols),signature=signature);validate_first(first,signature);savej(firstpath,first);meta['first_audit_sha256']=sha(firstpath)
                    savej(metapath,meta)
                validate_csv(d,q,mode,v,k,seed);compare(d.y,q.sub_ec);compare(d.baseline,ref['baseline'][seed]);compare(d.base_raw,ref['querybase'][seed]);compare(d.candidate,final(d.base_raw.to_numpy()+d.delta.to_numpy(),q,ref['outer_bounds']));compare(d.candidate,np.clip(smooth_scalar(d.base_raw.to_numpy()+d.delta.to_numpy(),q),*ref['outer_bounds']))
                for key,val in [('clip_lo',ref['outer_bounds'][0]),('clip_hi',ref['outer_bounds'][1])]:compare(d[key],np.full(len(q),val))
                modeframes.append(d);allframes.append(d);audits.append(meta);print(mode,v,k,seed,'COMPLETE_SCORE0',flush=True);del net;gc.collect()
        target=OUT/f'oof_{mode}.csv';assert not target.exists();pd.concat(modeframes,ignore_index=True).to_csv(target,index=False,mode='x')
    assert len(audits)==132
    pd.concat(allframes,ignore_index=True).to_csv(OUT/'oof.csv',index=False,mode='x')
    savej(H/'fit_audit_v1.json',dict(status='PASS',config=CONFIG,preparation=prepared,cells=audits,rows=166320,cells_per_mode=66,score_count=0,aggregate_sha256=sha(OUT/'oof.csv'),mode_aggregate_sha256={m:sha(OUT/f'oof_{m}.csv') for m in MODES}))
    print('ALL_FITS_COMPLETE_SCORE0',flush=True)
def synthetic():
    target=H/'synthetic_run_v1.json';assert not target.exists();runtime()
    rng=np.random.default_rng(20261004);q=pd.DataFrame(dict(row_id=[f'{f}_{d:03d}_{h:02d}' for f in ['F13','F47'] for d in [1,2] for h in range(24)],farm=['F13']*48+['F47']*48,day=[1]*24+[2]*24+[1]*24+[2]*24,hour=list(range(24))*4))
    x=rng.normal(size=(96,14));raw=rng.normal(.5,.1,96);y=raw+rng.normal(0,.03,96);bounds=(.1,.9)
    xt=torch.from_numpy(raw.copy()).requires_grad_(True);groups=ordered_groups(q)
    compare(smooth_tensor(xt,groups).detach().numpy(),smooth_scalar(raw,q));compare(final(raw,q,bounds),np.clip(smooth_scalar(raw,q),*bounds))
    obj=torch.mean((torch.clamp(smooth_tensor(xt,groups),min=bounds[0],max=bounds[1])-torch.from_numpy(y))**2);obj.backward();autograd=xt.grad.numpy();fd=[]
    for i in range(96):
        delta=np.zeros(96);delta[i]=1e-6;fd.append((np.mean((final(raw+delta,q,bounds)-y)**2)-np.mean((final(raw-delta,q,bounds)-y)**2))/2e-6)
    fd_gap=float(np.max(abs(autograd-fd)));assert fd_gap<1e-8
    trainings={}
    for mode in MODES:
        net,trace=train(x,y,raw,q,bounds,7,mode);p=predict(net,x);fresh,t2=train(x,y,raw,q,bounds,7,mode)
        gap=compare(p,predict(fresh,x));compare(trace,t2);compare(p,predict(net,x[::-1].copy())[::-1]);compare(p[:8],np.array([predict(net,x[i:i+1])[0] for i in range(8)]))
        zero=make_model(7);assert np.array_equal(predict(zero,x),np.zeros(96))
        trainings[mode]=dict(initial=float(trace[0]),final=float(trace[-1]),fresh_maxdiff=gap,steps=400)
    assert not np.array_equal(predict(net,x),predict(make_model(7),x))
    # Synthetic preprocessing only, including all-NaN feature retained at zero.
    cols=[f'x{i}' for i in range(14)];xb=x.copy();xb[0,0]=np.nan;xb[:,1]=np.nan;frame=pd.DataFrame(xb,columns=cols);stats,filled=preprocessing(frame,cols);compare(filled,transform(frame,cols,stats));assert stats['imputer_statistics'][1]==0
    savej(target,dict(status='PASS_SYNTHETIC_ONLY',config=CONFIG,source_sha256=sha(Path(__file__)),runtime=runtime(),autograd_fd_maxdiff=fd_gap,training=trainings,synthetic_training_steps=1600,real_ec_fit=0,real_ec_predict=0,real_score=0,raw_ec_reads=0,test_reads=0,EL1_rescore=0));print('PASS_SYNTHETIC_ONLY',fd_gap,trainings,flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--synthetic',action='store_true');args=parser.parse_args();assert not(args.prepare and args.synthetic)
    bootstrap(support=not args.synthetic)
    if args.synthetic:synthetic();return
    refs,prepared,cols=preflight()
    if args.prepare:savej(H/'preparation_v1.json',prepared);print('PREPARATION_PASS_MODEL_FIT0_PREDICT0_SCORE0',flush=True);return
    actual(refs,prepared,cols)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        failure=H/f'failure_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.json'
        savej(failure,dict(status='STOP_PRESERVE',error_type=type(error).__name__,error=str(error),source_sha256=sha(Path(__file__)),traceback=__import__('traceback').format_exc()));raise
