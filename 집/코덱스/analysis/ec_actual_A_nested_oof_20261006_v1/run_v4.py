from pathlib import Path
import uuid
import sys,os,json,hashlib,math,gc,time,argparse,platform,importlib.util,zipfile,subprocess
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env;import env_extra
for n in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[n]='1'
sys.path.insert(0,str(H.parent/'statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd,torch,sklearn,lightgbm
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
CKPT=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
SEEDS=[7,101,2024];PSEEDS=[1,2,3,4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p)
    if p.exists():assert load(p)==x;return
    stage=p.with_name(p.name+'.partial_'+uuid.uuid4().hex)
    with stage.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
    os.rename(stage,p)
def cp(name):return OUT/'components'/Path(name).stem/Path(name).name
def savecsv(p,df):
    if p.exists():pd.testing.assert_frame_equal(pd.read_csv(p,float_precision='round_trip'),df,check_dtype=False,check_exact=True);return
    stage=p.with_name(p.name+'.partial_'+uuid.uuid4().hex)
    with stage.open('x',encoding='utf-8',newline='') as f:df.to_csv(f,index=False)
    os.rename(stage,p)
def ids(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def ar(x):
    x=np.asarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ordered(q):
    assert q.row_id.is_unique
    for _,g in q.groupby(['farm','day'],sort=True):assert np.array_equal(g.hour,np.arange(24))
def splits(tr):
    assignment={}
    for f,g in tr.groupby('farm'):
        for i,d in enumerate(sorted(g.day.unique())):assignment[f,int(d)]=(i//5)%4
    keys=list(zip(tr.farm,tr.day));result=[]
    for j in range(4):
        chosen={key for key,z in assignment.items() if z==j};ban={(f,d+o) for f,d in chosen for o in [-1,0,1]}
        ti=[i for i,(f,d) in enumerate(keys) if (f,int(d)) not in ban];vi=[i for i,(f,d) in enumerate(keys) if (f,int(d)) in chosen]
        assert ti and vi and not set(ti)&set(vi);result.append((ti,vi))
    assert sorted(i for _,vi in result for i in vi)==list(range(len(tr)));return result
def runtime():
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,torch=str(torch.__version__))
    assert versions==dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0',torch='2.14.0+cpu'),versions
    return dict(versions=versions,module_init_sha={n:sha(m.__file__) for n,m in [('numpy',np),('pandas',pd),('sklearn',sklearn),('lightgbm',lightgbm),('torch',torch)]},tabpfn_sha=sha(sys.modules[TabPFNRegressor.__module__].__file__),checkpoint=sha(CKPT))
def dependencies(core):return {str(Path(p).relative_to(ROOT)):sha(p) for p in [Path(__file__),H/'plan_v1.md',H/'verify_v4.py',Path(S.__file__),Path(core.__file__),Path(env.__file__),Path(env_extra.__file__),Path(sys.modules[S.mapping.__module__].__file__)]}
def prep():
    assert sha(CKPT)=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    lab,core,wv,folds,outer=S.loadec();assert len(lab)==8640 and lab.row_id.is_unique;ordered(lab)
    original=load(H.parent/'ec_actual_A_cases_20261006_v1/result_v1.json');original_sigs={(m['validator'],m['fold']):m for m in original['audit_manifest']}
    full=[c for c in core.FULL if c!='day']+['season'];base=[c for c in core.BASE if c!='day']+['season'];records=[];jobs=[]
    for v,k,tm,vm in folds:
        if v not in ['DIAG10','A','B']:continue
        tr,q=lab[tm].reset_index(drop=True),lab[vm].reset_index(drop=True);sig=original_sigs[v,k]
        assert ids(tr.row_id)==sig['outer_train_ids'] and ids(q.row_id)==sig['outer_query_ids'];assert ar(tr.sub_ec.to_numpy())==sig['outer_train_targets']
        oids=set(q.row_id);outerban={(f,int(d)+o) for f,d in q[['farm','day']].itertuples(index=False,name=None) for o in [-1,0,1]}
        assert not set(tr[['farm','day']].itertuples(index=False,name=None))&outerban
        for j,(ti,vi) in enumerate(splits(tr)):
            a,b=tr.iloc[ti].reset_index(drop=True),tr.iloc[vi].reset_index(drop=True);ordered(a);ordered(b)
            assert not (set(a.row_id)|set(b.row_id))&oids
            ban={(f,int(d)+o) for f,d in b[['farm','day']].itertuples(index=False,name=None) for o in [-1,0,1]};assert not set(a[['farm','day']].itertuples(index=False,name=None))&ban
            aa,bb=S.seasonal(a,b,wv);allowed=set(a[['farm','day']].itertuples(index=False,name=None));sv={key:wv[key] for key in allowed}
            ts,qs,notes=S.mapping(a[['farm','day']].drop_duplicates(),b[['farm','day']].drop_duplicates(),sv)
            assert np.array_equal(bb.season.to_numpy(),S.seasonal(a,b,wv)[1].season.to_numpy())
            entry=dict(v=v,k=k,j=j,ti=ti,vi=vi,outer_train_ids=tr.row_id.tolist(),outer_query_ids=q.row_id.tolist(),train_ids=a.row_id.tolist(),query_ids=b.row_id.tolist(),train_target_sha=ar(a.sub_ec.to_numpy()),query_target_sha=ar(b.sub_ec.to_numpy()),full=full,base=base,train_full_sha=ar(aa[full].to_numpy()),query_full_sha=ar(bb[full].to_numpy()),train_base_sha=ar(aa[base].to_numpy()),query_base_sha=ar(bb[base].to_numpy()),bounds=[float(a.sub_ec.min()),float(a.sub_ec.max())],season_notes=notes)
            records.append(entry);jobs.append((entry,tr,a,b,aa,bb))
    assert len(records)==80
    p=dict(status='PREPARED_80_CONTEXTS_FIT0',runtime=runtime(),dependencies=dependencies(core),inputs=original['audit_inputs'],full=full,base=base,records=records,core_params=dict(et=core.et(7).steps[-1][1].get_params(),lgb=core.lg(7,'tweedie').get_params()),seeds=SEEDS,pfn_seeds=PSEEDS,pfn_config=dict(version='V2',device='cpu',precision='float32',context=2000,n_estimators=4,n_preprocessing_jobs=1),threads=dict(R3=2,PFN=4))
    assert sha(Path(env.DATA)/'train_X.csv')==p['inputs']['train_X'];assert sha(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')==p['inputs']['public_oof']
    return core,jobs,p
def frozen():
    for name,h in load(H/'registration_v4.json')['hashes'].items():assert sha(H/name)==h,name
def cache(path,signature,row_ids):
    path=cp(path.name)
    if not path.exists():assert not path.with_suffix('.json').exists();return None
    m=load(path.with_suffix('.json'));assert m['signature']==signature and m['sha']==sha(path)
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    assert np.array_equal(d['row_id'],row_ids);assert np.isfinite(d['raw']).all();return d,m
def writecache(path,d,signature,details):
    dest=cp(path.name);dest.parent.parent.mkdir(parents=True,exist_ok=True)
    stage=OUT/'.staging'/(dest.parent.name+'_'+uuid.uuid4().hex);stage.mkdir(parents=True,exist_ok=False);temp=stage/dest.name
    with temp.open('xb') as f:np.savez(f,**d)
    m=dict(signature=signature,sha=sha(temp),details=details,path=str(dest.relative_to(OUT)))
    save(temp.with_suffix('.json'),m);assert sha(temp)==m['sha']
    os.rename(stage,dest.parent);return d,m
def signature(e,p,kind,seed):return dict(record_sha=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest(),prepared_sha=sha(H/'preparation_v4.json'),dependencies=p['dependencies'],runtime=p['runtime'],kind=kind,seed=seed)
def models(core,s):return [(core.et(s).set_params(extratreesregressor__n_jobs=2),'full'),(core.lg(s,'tweedie').set_params(n_jobs=2),'base'),(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=s)),'base')]
def r3(core,aa,bb,e,p,s,first):
    path=OUT/f"{e['v']}_{e['k']}_{e['j']}_r3_{s}.npz";sig=signature(e,p,'R3 .6ET+.3LGB+.1MLP',s);old=cache(path,sig,bb.row_id)
    if old:return old
    frozen();t=time.monotonic();pred=[];trainpred=[];repeat_gap=None;info=[]
    for model,kind in models(core,s):
        cols=e[kind]
        with threadpool_limits(limits=2):model.fit(aa[cols],aa.sub_ec);a=np.asarray(model.predict(bb[cols]),float);b=np.asarray(model.predict(aa[cols]),float)
        if first:
            with threadpool_limits(limits=2):fresh=clone(model);fresh.fit(aa[cols],aa.sub_ec);gap=float(np.max(abs(a-fresh.predict(bb[cols]))))
            assert gap<1e-9;repeat_gap=max(repeat_gap or 0.,gap);del fresh
        pred.append(a);trainpred.append(b);info.append(dict(kind=kind,n_iter=int(getattr(model.steps[-1][1] if hasattr(model,'steps') else model,'n_iter_',0))));del model;gc.collect()
    raw=.6*pred[0]+.3*pred[1]+.1*pred[2];traw=.6*trainpred[0]+.3*trainpred[1]+.1*trainpred[2]
    return writecache(path,dict(row_id=bb.row_id.to_numpy(str),train_row_id=aa.row_id.to_numpy(str),raw=raw,et=pred[0],lgb=pred[1],mlp=pred[2],train_raw=traw),sig,dict(seconds=time.monotonic()-t,repeat_maxdiff=repeat_gap,components=info,train_raw_rmse=math.sqrt(math.fsum((traw-aa.sub_ec.to_numpy())**2)/len(aa))))
def pfn(aa,bb,e,p,s,first):
    path=OUT/f"{e['v']}_{e['k']}_{e['j']}_pfn_{s}.npz";sig=signature(e,p,'TabPFN V2 CPU float32 context2000 estimator4',s);old=cache(path,sig,bb.row_id)
    if old:return old
    frozen();t=time.monotonic();ix=np.random.default_rng(s).choice(len(aa),min(2000,len(aa)),replace=False)
    def fit():
        m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=str(CKPT),n_estimators=4,random_state=s,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            m.fit(aa[e['full']].to_numpy(np.float32)[ix],aa.sub_ec.to_numpy(float)[ix]);raw=np.asarray(m.predict(bb[e['full']].to_numpy(np.float32)),float)
            small=np.asarray(m.predict(bb.iloc[:8][e['full']].to_numpy(np.float32)),float)
        assert np.array_equal(raw[:8],small);del m;gc.collect();return raw
    raw=fit();gap=None
    if first:gap=float(np.max(abs(raw-fit())));assert gap<1e-5
    return writecache(path,dict(row_id=bb.row_id.to_numpy(str),train_row_id=aa.row_id.to_numpy(str),context_row_id=aa.row_id.iloc[ix].to_numpy(str),raw=raw),sig,dict(seconds=time.monotonic()-t,repeat_maxdiff=gap,first8_batch_invariance=True))
def actual():
    frozen();core,jobs,p=prep();assert p==load(H/'preparation_v4.json');OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'worker.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),source=sha(Path(__file__))),f)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assemblies={};files=[];cases=[]
    for e,tr,a,b,aa,bb in jobs:
        assert dependencies(core)==p['dependencies'];v,k,j=e['v'],e['k'],e['j'];first=(v=='DIAG10' and k==0 and j==0)
        rs={}
        for s in SEEDS:
            d,m=r3(core,aa,bb,e,p,s,first and s==7);rs[s]=d;files.append(dict(path=m['path'],sha=m['sha']));print('NESTED_R3',v,k,j,s,m['details']['seconds'],flush=True)
        ps=[]
        for s in PSEEDS:
            d,m=pfn(aa,bb,e,p,s,first and s==1);ps.append(d['raw']);files.append(dict(path=m['path'],sha=m['sha']));print('NESTED_PFN',v,k,j,s,m['details']['seconds'],flush=True)
        bag=np.mean(ps,axis=0)
        for s in SEEDS:
            raw=.8*rs[s]['raw']+.2*bag;pred=np.clip(core.shrink(raw,b),*e['bounds']);df=b[['row_id','farm','day','hour']].copy();df['y']=b.sub_ec;df['A']=pred;df['raw_r3']=rs[s]['raw'];df['raw_pfn']=bag;df['raw_A']=raw;df['clip_lo'],df['clip_hi']=e['bounds'];df['v'],df['k'],df['j'],df['s']=v,k,j,s
            assemblies.setdefault((v,k,s),[]).append(df)
        print('NESTED_CONTEXT_COMPLETE',v,k,j,flush=True)
        if j==3:
            for s in SEEDS:
                df=pd.concat(assemblies.pop((v,k,s)),ignore_index=True).set_index('row_id').loc[tr.row_id].reset_index();assert df.row_id.is_unique and len(df)==len(tr)
                prefix=df.groupby(['farm','day']).A.transform(lambda z:z.expanding().mean());df['prefix_A']=prefix;yd=df.groupby(['farm','day']).y.transform('mean');df['y_day']=yd;df['high']=(yd>=1).astype(int);df['hard_high']=((yd>=1)&(prefix<1.2)).astype(int);df['missed_high']=((yd>=1)&(prefix<.9)).astype(int);df['hard_low']=((yd<1)&(prefix>=.9)).astype(int)
                dest=OUT/f'OOF_{v}_{k}_{s}.csv'
                savecsv(dest,df)
                files.append(dict(path=dest.name,sha=sha(dest)));cases.append(df[df.hour.isin([0,6,12,23])].copy())
            print('NESTED_OUTER_COMPLETE',v,k,flush=True)
    assert not assemblies
    case=pd.concat(cases,ignore_index=True);dest=OUT/'nested_cases_v1.csv';savecsv(dest,case);files.append(dict(path=dest.name,sha=sha(dest)))
    consensus=[]
    for (v,k,farm,day,h),g in case.groupby(['v','k','farm','day','hour']):
        assert set(g.s)==set(SEEDS) and g.high.nunique()==1
        consensus.append(dict(v=v,k=int(k),farm=farm,day=int(day),hour=int(h),high=int(g.high.iloc[0]),y_day=float(g.y_day.iloc[0]),A_prefix_min=float(g.prefix_A.min()),A_prefix_max=float(g.prefix_A.max()),hard_high_votes=int(g.hard_high.sum()),missed_high_votes=int(g.missed_high.sum()),hard_low_votes=int(g.hard_low.sum())))
    dest=OUT/'nested_seed_consensus_v1.csv';savecsv(dest,pd.DataFrame(consensus));files.append(dict(path=dest.name,sha=sha(dest)))
    save(H/'receipt_v4.json',dict(status='COMPLETE_80_CONTEXTS_ACTUAL_A_OOF',component_files=560,outer_oof_files=60,files=files,source_sha=sha(Path(__file__)),new_classifier=False,submission=False))
    subprocess.run([sys.executable,str(H/'verify_v4.py')],check=True)
    print('NESTED_ALL_COMPLETE_VERIFIED',flush=True)
    lock.unlink()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:_,_,p=prep();save(H/'preparation_v4.json',p);print('PREPARED_80_CONTEXTS_FIT0',flush=True)
    else:actual()
if __name__=='__main__':main()
