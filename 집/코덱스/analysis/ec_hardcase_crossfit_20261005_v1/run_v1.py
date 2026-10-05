from pathlib import Path
import sys,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sp=importlib.util.spec_from_file_location('parent_classifier',H.parent/'ec_high_classifier_20261005_v1/run_v1.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
np,pd=P.np,P.pd;R=P.R;save,sha,load,close=P.save,P.sha,P.load,P.close
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
OUT=ROOT/'집/코덱스/local'/H.name;SEEDS=P.SEEDS;MODES=['UNIFORM','HARD'];CORE=None;WV=None;COLS=None

def splits(q):
    assignments={}
    for f,g in q.groupby('farm'):
        for i,d in enumerate(sorted(g.day.unique())):assignments[f,int(d)]=(i//5)%4
    keys=list(zip(q.farm,q.day));out=[]
    for j in range(4):
        chosen={key for key,v in assignments.items() if v==j};ban={(f,d+o) for f,d in chosen for o in [-1,0,1]};ti=np.array([i for i,(f,d) in enumerate(keys) if (f,int(d)) not in ban],int);vi=np.array([i for i,(f,d) in enumerate(keys) if (f,int(d)) in chosen],int)
        y=P.target(q);days=q.iloc[ti][['farm','day']].copy();days['high']=y[ti];days=days.drop_duplicates();assert days.high.sum()>=3 and (days.high==0).sum()>=10
        out.append(dict(ti=ti.tolist(),vi=vi.tolist(),train_high=int(days.high.sum()),train_low=int((days.high==0).sum())))
    assert sorted(i for z in out for i in z['vi'])==list(range(len(q)));return out

def prep():
    global CORE,WV,COLS
    refs,sig,prior=P.D.prepare();P.SIG=sig;lab,CORE,WV,folds,outer=R.S.loadec();COLS=[c for c in CORE.BASE if c!='day']+['season'];records=[]
    for (v,k),r in refs.items():
        if v not in ['DIAG10','A','B']:continue
        z=splits(r['tr']);records.append(dict(v=v,k=k,tr=R.ids(r['tr'].row_id),q=R.ids(r['q'].row_id),split=z,y_sha=R.ar(r['tr'].sub_ec.to_numpy())))
        assert not set(r['tr'].row_id)&set(r['q'].row_id)
    return refs,dict(status='PREPARED_FIT0',records=records,prior=prior,cols=COLS,proxy_params=CORE.lg(7,'tweedie').get_params(),lr=P.CFG)

def proxy_fit(tr,q,s):
    a,b=R.S.seasonal(tr,q,WV);model=CORE.lg(s,'tweedie');model.set_params(n_jobs=2)
    with P.M.threadpool_limits(limits=2):model.fit(a[COLS],a.sub_ec.to_numpy())
    raw=model.predict(b[COLS]);tree=model.booster_.dump_model();bounds=[float(tr.sub_ec.min()),float(tr.sub_ec.max())];assert tree['objective'].startswith('tweedie');raw2=np.exp(P.M.tree_logits(b[COLS].to_numpy(),tree));close(raw,raw2)
    pred=np.clip(.5*raw+.5*R.prefix(q,raw),*bounds)
    meta=dict(ids=R.ids(tr.row_id),query_ids=R.ids(q.row_id),x_sha=R.ar(a[COLS].to_numpy()),query_x_sha=R.ar(b[COLS].to_numpy()),y_sha=R.ar(tr.sub_ec.to_numpy()),tree=tree,bounds=bounds,seed=s)
    return pred,meta

def proxy_replay(tr,q,m):
    a,b=R.S.seasonal(tr,q,WV);assert m['ids']==R.ids(tr.row_id) and m['query_ids']==R.ids(q.row_id);assert m['x_sha']==R.ar(a[COLS].to_numpy()) and m['query_x_sha']==R.ar(b[COLS].to_numpy()) and m['y_sha']==R.ar(tr.sub_ec.to_numpy());assert m['tree']['objective'].startswith('tweedie');raw=np.exp(P.M.tree_logits(b[COLS].to_numpy(),m['tree']));close(m['bounds'],[tr.sub_ec.min(),tr.sub_ec.max()]);return np.clip(.5*raw+.5*R.prefix(q,raw),*m['bounds'])

def design(ref,q,a):
    nx,sources=P.reference(ref,q,'ALL');prefix=R.prefix(q,a);sig=P.SIG.reindex(q.row_id).to_numpy();sig=np.nan_to_num(sig);x=np.column_stack([a,prefix,sig,q.hour.to_numpy()/23,nx]);assert x.shape[1]==22 and np.isfinite(x).all();return x,sources

def weights(q,x,mode):
    y=P.target(q);n=len(y);w=np.where(y==1,n/(2*sum(y)),n/(2*(n-sum(y))));hard=((y==1)&(x[:,1]<1.2))|((y==0)&(x[:,1]>=.9))
    if mode=='HARD':w=w*np.where(hard,4.,1.)
    return w/w.mean(),hard

def classifier(q,x,s,mode):
    y=P.target(q);w,hard=weights(q,x,mode);scale=StandardScaler().fit(x);model=LogisticRegression(C=1,class_weight=None,max_iter=3000,tol=1e-8,solver='lbfgs',random_state=s)
    with P.M.threadpool_limits(limits=2):model.fit(scale.transform(x),y,sample_weight=w)
    m=dict(ids=R.ids(q.row_id),y_sha=R.ar(y),x_sha=R.ar(x),w_sha=R.ar(w),constant=None,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_.ravel().tolist(),intercept=float(model.intercept_[0]),mode=mode,seed=s)
    close(P.forward(x,m),model.predict_proba(scale.transform(x))[:,1]);return m

def frozen():
    for n,h in load(H/'registration_v1.json')['hashes'].items():assert sha(H/n)==h,n

def actual():
    frozen();refs,pr=prep();assert pr==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);files=[];models=[]
    for (v,k),r in refs.items():
        if v not in ['DIAG10','A','B']:continue
        tr,q=r['tr'],r['q'];ss=splits(tr)
        for s in SEEDS:
            xx=np.full((len(tr),22),np.nan);px=np.full(len(tr),np.nan);pm=[];sources=[]
            for j,z in enumerate(ss):
                ti,vi=np.array(z['ti']),np.array(z['vi']);a,b=tr.iloc[ti].reset_index(drop=True),tr.iloc[vi].reset_index(drop=True);p,m=proxy_fit(a,b,s);x,src=design(a,b,p);xx[vi]=x;px[vi]=p;pm.append(m);sources.append(src);print('HARD_PROXY',v,k,s,j,flush=True)
            assert np.isfinite(xx).all();op,full=proxy_fit(tr,q,s);ox,osrc=design(tr,q,op)
            shared=OUT/f'proxy_{v}_{k}_{s}.json';save(shared,dict(split=ss,proxy=pm,proxy_full=full,inner_sources=sources,outer_sources=osrc))
            for mode in MODES:
                m=classifier(tr,xx,s,mode);prob=P.forward(ox,m);ip=P.forward(xx,m);bundle=dict(v=v,k=k,s=s,mode=mode,shared=shared.name,shared_sha=sha(shared),classifier=m)
                mp=OUT/f'model_{mode}_{v}_{k}_{s}.json';save(mp,bundle);models.append(dict(path=mp.name,sha=sha(mp)))
                for context,qq,proxy,x,p,base in [('outer',q,op,ox,prob,r['baseline'][s]),('train',tr,px,xx,ip,px)]:
                    df=qq[['row_id','farm','day','hour']].copy();df['high']=P.target(qq);df['proxy']=proxy;df['prefix_proxy']=x[:,1];df['p']=p;df['A']=base;df['prefix_A']=R.prefix(qq,base);df['DIRECT']=(p>=.5).astype(int);df['GUARD']=((p>=.5)&(df.prefix_A.to_numpy()>=.9)).astype(int);df['hard_high']=((df.high==1)&(df.prefix_proxy<1.2)).astype(int);df['hard_low']=((df.high==0)&(df.prefix_proxy>=.9)).astype(int);fp=OUT/f'{context}_{mode}_{v}_{k}_{s}.csv';df.to_csv(fp,index=False);files.append(dict(path=fp.name,sha=sha(fp),model=mp.name,model_sha=sha(mp),v=v,k=k,s=s,mode=mode,context=context))
            frozen();print('HARD_FIT',v,k,s,flush=True)
    save(H/'receipt_v1.json',dict(status='COMPLETE_PROXY300_CLASSIFIER120',files=files,models=models))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:refs,p=prep();save(H/'preparation_v1.json',p)
    else:actual()
