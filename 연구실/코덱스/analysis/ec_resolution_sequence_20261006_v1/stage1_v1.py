from pathlib import Path
import sys,os,json,hashlib,importlib.util,time,gc,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
import numpy as np,pandas as pd,torch,sklearn,lightgbm,tabpfn
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
L=ROOT/'연구실/코덱스/local'/H.name;O=L/'stage1_v1';O.mkdir(parents=True,exist_ok=True)
C=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'
SOURCE=ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1/model.py'
spec=importlib.util.spec_from_file_location('readonly_frozen_season_model',SOURCE);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
CKPT=L/'tabpfn-v2-regressor.ckpt';SEEDS=[7,101,2024];PSEEDS=[1,2,3,4]
TARGETS=[('F13',98),('F13',112),('F47',160),('F47',161)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)
    if p.exists():assert p.read_text(encoding='utf-8')==text,p
    else:p.write_text(text,encoding='utf-8')
def fhash(frame,cols):return hashlib.sha256(pd.util.hash_pandas_object(frame[cols],index=False).values.tobytes()).hexdigest()
def identities(frame):return frame.row_id.to_numpy(str)
def ordered(x,ids):return x.set_index('row_id').loc[list(ids)].reset_index()
def prepare():
    assert sha(CKPT)=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    versions={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__,'tabpfn':tabpfn.__version__}
    assert versions==dict(numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0',torch='2.14.0+cpu',tabpfn='9.0.0')
    op=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'
    z=pd.read_csv(op,float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates();assert len(y)==8640
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    assert sha(Path(env.DATA)/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    f=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',how='inner',validate='one_to_one')
    assert len(f)==8640;vec=M.vectors(raw);original={};proof=[]
    for k in [0,1,8]:
        with np.load(C/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as a:ti=a['train_row_id'].astype(str);qi=a['row_id'].astype(str)
        tr=ordered(f,ti);q=ordered(f,qi);tr,q,notes=season(tr,q,vec)
        meta=json.loads((C/f'DIAG10_{k}_r3_7.json').read_text(encoding='utf-8'))['provenance']
        th=fhash(tr,['row_id','sub_ec']+M.FULL);qh=fhash(q,['row_id','sub_ec']+M.FULL)
        assert th==meta['train_hash'] and qh==meta['validation_hash'],(k,th,qh)
        original[k]=(tr,q);proof.append(dict(fold=k,train_hash=th,query_hash=qh,train_rows=len(tr),query_rows=len(q)))
    q= f[[(r.farm,int(r.day)) in TARGETS for r in f.itertuples()]].sort_values(['farm','day','hour']).reset_index(drop=True);assert len(q)==96
    bans={(f,d+j) for f,d in TARGETS for j in [-1,0,1]};jobs={}
    for k,(tr,v) in original.items():
        ids=tr.row_id[[(r.farm,int(r.day)) not in bans for r in tr.itertuples()]];t=ordered(f,ids)
        tt,qq,notes=season(t,q,vec);jobs[f'trainfold{k}_purged']=(tt,qq)
    common=set.intersection(*[set(t.row_id) for t,q in jobs.values()]);ids=jobs['trainfold0_purged'][0].row_id[jobs['trainfold0_purged'][0].row_id.isin(common)]
    jobs['common_intersection']=season(ordered(f,ids),q,vec)[:2]
    for name,(t,q) in jobs.items():
        assert not set(t.row_id)&set(q.row_id)
        assert len(t)>=2000
        for farm,day in TARGETS:assert min(abs(t.loc[t.farm==farm,'day']-day))>=2
    deps=[Path(__file__),H/'PLAN.md',SOURCE,SOURCE.with_name('season.py'),op]
    manifest=dict(status='PREPARED_SAME_MODEL_DIAGNOSTIC',versions=versions,sha={str(p):sha(p) for p in deps},checkpoint=sha(CKPT),original_reproduction=proof,targets=TARGETS,jobs={n:dict(rows=len(t),days=len(t)//24,train_hash=fhash(t,['row_id','sub_ec']+M.FULL),query_hash=fhash(q,['row_id','sub_ec']+M.FULL)) for n,(t,q) in jobs.items()},fit=0,test_reads=0,raw_y_reads=0)
    save(H/'preparation_v1.json',manifest);return original,jobs,manifest
def season(t,q,vec):
    td=t[['farm','day']].drop_duplicates();qd=q[['farm','day']].drop_duplicates().reset_index(drop=True)
    ts,qs,notes=M.mapping(td,qd,vec);qm=dict(zip(qd.itertuples(index=False,name=None),qs))
    t=t.copy();q=q.copy();t['season']=[ts[(r.farm,int(r.day))] for r in t.itertuples()];q['season']=[qm[(r.farm,int(r.day))] for r in q.itertuples()]
    return t,q,notes
def cache(name,fn,t,q,p):
    path=O/(name+'.npz');j=path.with_suffix('.json');key=dict(preparation_sha=sha(H/'preparation_v1.json'),source_sha=sha(__file__),train=fhash(t,['row_id','sub_ec']+M.FULL),query=fhash(q,['row_id','sub_ec']+M.FULL))
    if path.exists() or j.exists():
        assert path.exists() and j.exists();m=json.loads(j.read_text(encoding='utf-8'));assert m['key']==key and m['sha']==sha(path)
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    start=time.monotonic();a=fn();a.update(row_id=identities(q),train_row_id=identities(t))
    assert all(np.isfinite(v).all() for v in a.values() if v.dtype.kind in 'fc')
    tmp=path.with_suffix('.staging.npz');np.savez_compressed(tmp,**a);os.replace(tmp,path);save(j,dict(key=key,sha=sha(path),seconds=time.monotonic()-start))
    print('SAVED',name,round(time.monotonic()-start,1),flush=True);return a
def tree_trace(model,t,q,name):
    im,forest=model.steps[0][1],model.steps[-1][1];xt=im.transform(t[M.FULL]);xq=im.transform(q[M.FULL]);leaves=forest.apply(xt);paths=[]
    indices={(r.farm,int(r.day),int(r.hour)):i for i,r in enumerate(q.itertuples())}
    for farm,good,bad in [('F47',160,161),('F13',98,112)]:
        if (farm,good,0) not in indices or (farm,bad,0) not in indices:continue
        a,b=indices[farm,good,0],indices[farm,bad,0]
        for n,est in enumerate(forest.estimators_):
            tr=est.tree_;node=0;split=None
            while tr.children_left[node]!=tr.children_right[node]:
                c=tr.feature[node];th=tr.threshold[node];la=xq[a,c]<=th;lb=xq[b,c]<=th
                if la!=lb:split=dict(feature=M.FULL[c],threshold=float(th),good_value=float(xq[a,c]),bad_value=float(xq[b,c]));break
                node=int(tr.children_left[node] if la else tr.children_right[node])
            pa=float(est.predict(xq[[a]])[0]);pb=float(est.predict(xq[[b]])[0]);paths.append(dict(context=name,farm=farm,good=good,bad=bad,tree=n,good_prediction=pa,bad_prediction=pb,difference=pb-pa,first_divergence=split))
    save(O/(name+'_et_paths.json'),paths)
def r3(t,q,seed,name,trace=False):
    def run():
        out={}
        for kind,model,cols in [('et',M.et(seed),M.FULL),('lgb',M.lg(seed),M.BASE),('mlp',M.mlp(seed),M.BASE)]:
            with threadpool_limits(limits=2):
                model.fit(t[cols],t.sub_ec.to_numpy(float));out['raw_'+kind]=np.asarray(model.predict(q[cols]),float)
                small=np.asarray(model.predict(q.iloc[:8][cols]),float);assert np.max(abs(out['raw_'+kind][:8]-small))<1e-8
                if kind=='et' and trace:tree_trace(model,t,q,name)
            del model;gc.collect()
        out['raw_r3']=.6*out['raw_et']+.3*out['raw_lgb']+.1*out['raw_mlp'];return out
    return cache(name+'_r3_'+str(seed),run,t,q,None)
def pfn(t,q,seed,name):
    def run():
        ix=np.random.default_rng(seed).choice(len(t),min(2000,len(t)),replace=False)
        m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=str(CKPT),n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            m.fit(t[M.FULL].to_numpy(np.float32)[ix],t.sub_ec.to_numpy(float)[ix]);p=np.asarray(m.predict(q[M.FULL].to_numpy(np.float32)),float)
            small=np.asarray(m.predict(q.iloc[:8][M.FULL].to_numpy(np.float32)),float);assert np.max(abs(p[:8]-small))<1e-5
        del m;gc.collect();return dict(raw_pfn=p,context_index=ix,context_row_id=t.row_id.iloc[ix].to_numpy(str))
    return cache(name+'_pfn_'+str(seed),run,t,q,None)
def main():
    original,jobs,prep=prepare()
    if '--prepare' in sys.argv:print('PREPARED',prep['jobs'],flush=True);return
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    # Before any modified-training fit, reproduce the original fold0 recipe.
    t,q=original[0];q=q[[(r.farm,int(r.day)) in TARGETS for r in q.itertuples()]].reset_index(drop=True)
    gaps=[]
    for s in SEEDS:
        a=r3(t,q,s,'replay_fold0');
        with np.load(C/f'DIAG10_0_r3_{s}.npz',allow_pickle=False) as z:
            ix=pd.Index(z['row_id']).get_indexer(q.row_id);assert min(ix)>=0
            gap=max(float(np.max(abs(a[k]-z[k][ix]))) for k in ['raw_et','raw_lgb','raw_mlp','raw_r3']);assert gap<=1e-8,(s,gap)
            gaps.append(dict(model='R3',seed=s,maxdiff=gap))
    a=pfn(t,q,1,'replay_fold0')
    with np.load(C/'DIAG10_0_pfn_1.npz',allow_pickle=False) as z:
        ix=pd.Index(z['row_id']).get_indexer(q.row_id);gap=float(np.max(abs(a['raw_pfn']-z['raw_pfn'][ix])));assert gap<=1e-5,gap
        assert np.array_equal(a['context_index'],z['context_index']);gaps.append(dict(model='PFN',seed=1,maxdiff=gap))
    save(H/'replay_verification_v1.json',dict(status='PASS',gaps=gaps,query_rows=len(q)))
    frames=[]
    for n,(t,q) in jobs.items():
        rs={s:r3(t,q,s,n,trace=s==7) for s in SEEDS};bag=np.mean([pfn(t,q,s,n)['raw_pfn'] for s in PSEEDS],axis=0)
        for s in SEEDS:
            raw=.8*rs[s]['raw_r3']+.2*bag;pred=np.clip(M.shrink(raw,q),t.sub_ec.min(),t.sub_ec.max());f=q[['row_id','farm','day','hour','sub_ec','season']].copy()
            for k in ['raw_et','raw_lgb','raw_mlp']:f[k]=rs[s][k]
            f['raw_pfn']=bag;f['raw_mix']=raw;f['prediction']=pred;f['context']=n;f['seed']=s;f['clip_lo']=float(t.sub_ec.min());f['clip_hi']=float(t.sub_ec.max());frames.append(f)
        print('CONTEXT_COMPLETE',n,flush=True)
    allrows=pd.concat(frames,ignore_index=True);dest=H/'stage1_rows_v1.csv'
    if not dest.exists():allrows.to_csv(dest,index=False)
    else:pd.testing.assert_frame_equal(pd.read_csv(dest,float_precision='round_trip'),allrows.reset_index(drop=True),check_dtype=False)
    save(H/'stage1_receipt_v1.json',dict(status='COMPLETE_DIAGNOSTIC',rows=len(allrows),contexts=4,R3_fits=12,PFN_contexts=16,output_sha=sha(dest),model_adoption=False,fit_targets_excluded=True))
    print('STAGE1_COMPLETE',flush=True)
if __name__=='__main__':main()
