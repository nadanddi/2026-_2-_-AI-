from pathlib import Path
import sys,importlib.util,argparse,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sp=importlib.util.spec_from_file_location('prior_B',H.parent/'ec_state_B_probe_20261005_v1/run_v2.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T)
D=T.D;np,pd=T.np,T.pd;R=T.R;M=T.M
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
sha,save,load,close=T.sha,T.save,T.load,T.close
OUT=ROOT/'집/코덱스/local'/H.name
MODES=['PAST','ALL'];SEEDS=[7,101,2024];CFG=dict(C=1,class_weight='balanced',max_iter=3000,tol=1e-8,solver='lbfgs');SIG=None
def reference(ref,q,mode):
    daily=ref.groupby(['farm','day']).sub_ec.mean();av=SIG.reindex(ref.row_id).to_numpy();qv=SIG.reindex(q.row_id).to_numpy();out=np.zeros((len(q),5));sources=[]
    assert not set(ref.row_id)&set(q.row_id)
    for h in range(24):
        rr=ref[ref.hour==h];x=av[ref.hour.to_numpy()==h];med=np.nan_to_num(np.nanmedian(x,axis=0));x=np.where(np.isnan(x),med,x);sd=x.std(0);sd[sd==0]=1;qi=np.where(q.hour.to_numpy()==h)[0];z=qv[qi];z=np.where(np.isnan(z),med,z)
        for f in ['F13','F47']:
            for phase in [False,True]:
                ai=np.where(((rr.farm==f)&((rr.day>=179)==phase)).to_numpy())[0];jj=np.where(((q.iloc[qi].farm==f)&((q.iloc[qi].day>=179)==phase)).to_numpy())[0]
                if not len(ai) or not len(jj):continue
                ai=ai[np.argsort(rr.day.to_numpy()[ai],kind='stable')];days=rr.day.to_numpy()[ai];labels=np.array([daily[f,int(day)] for day in days]);dist=np.sqrt(np.mean(((x[ai][None,:,:]-z[jj][:,None,:])/sd)**2,axis=2));allow=np.ones_like(dist,bool) if mode=='ALL' else days[None,:]<q.day.to_numpy()[qi[jj],None];dist[~allow]=np.inf;order=np.argsort(dist,axis=1,kind='stable')
                for u,j in enumerate(jj):
                    ii=int(qi[j]);n=int(allow[u].sum())
                    if not n:continue
                    ix=order[u,:min(5,n)];y=labels[ix];out[ii]=[y.mean(),(y>=1).mean(),y.std(),dist[u,ix[0]],np.log1p(n)];sources.append(dict(row_id=q.row_id.iloc[ii],days=days[ix].astype(int).tolist()))
    return out,sources
def target(q):return (q.groupby(['farm','day']).sub_ec.transform('mean')>=1).to_numpy(int)
def prep():
    global SIG
    refs,sig,p=D.prepare();SIG=sig;bank={};records=[]
    for (v,k),ref in refs.items():
        if v!='DIAG10':continue
        for mode in MODES:
            for inner in [True,False]:
                q=ref['b'] if inner else ref['q'];rr=ref['a'] if inner else ref['tr'];n,sources=reference(rr,q,mode)
                for s in SEEDS:
                    qq,a,x=T.pair(ref,s,inner);xx=np.column_stack([x,n]);assert xx.shape[1]==22 and np.isfinite(xx).all();bank[k,s,mode,inner]=(qq,a,xx,sources);records.append(dict(k=k,s=s,mode=mode,inner=inner,ids=R.ids(q.row_id),x_sha=R.ar(xx),y_sha=R.ar(target(q)),reference_ids=R.ids(rr.row_id),future_source_rows=sum(any(day>int(rid[4:7]) for day in z['days']) for z in sources for rid in [z['row_id']])))
        print('CLASS_PREP',k,flush=True)
    return refs,bank,dict(status='PREPARED_FIT0',records=records,params=CFG)
def fit(q,x,indices,s):
    y=target(q)[indices];m=dict(indices=np.asarray(indices).tolist(),ids=R.ids(q.row_id.iloc[indices]),labels_sha=R.ar(y),constant=float(y.mean()),n=len(y))
    if len(np.unique(y))<2:return m
    scale=StandardScaler().fit(x[indices]);model=LogisticRegression(**CFG,random_state=s).fit(scale.transform(x[indices]),y);m.update(constant=None,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_.reshape(-1).tolist(),intercept=float(model.intercept_[0]));close(model.predict_proba(scale.transform(x))[:,1],forward(x,m));return m
def forward(x,m):return np.full(len(x),m['constant']) if m['constant'] is not None else expit(((x-np.array(m['mean']))/m['scale'])@np.array(m['coef'])+m['intercept'])
def threshold(q,p):
    y=target(q);z=p[(q.hour==23).to_numpy()&(y==1)];return float(np.quantile(z,.05,method='lower')) if len(z) else .5
def frozen():
    for n,h in load(H/'registration_v1.json')['hashes'].items():assert sha(H/n)==h,n
def actual():
    frozen();refs,bank,p=prep();assert p==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);manifest=[]
    for (v,k),ref in refs.items():
        if v!='DIAG10':continue
        for s in SEEDS:
            for mode in MODES:
                q,a,x,sources=bank[k,s,mode,True];oq,oa,ox,osources=bank[k,s,mode,False];meta=np.full(len(q),np.nan);seen=np.zeros(len(q),int);mm=[]
                for j in range(3):
                    ti,vi=D.meta_split(ref,j)
                    with M.threadpool_limits(limits=2):model=fit(q,x,ti,s)
                    meta[vi]=forward(x[vi],model);seen[vi]+=1;mm.append(model)
                assert (seen==1).all() and np.isfinite(meta).all();th=threshold(q,meta)
                with M.threadpool_limits(limits=2):model=fit(q,x,np.arange(len(q)),s)
                pr=forward(ox,model);close(pr,forward(ox[::-1],model)[::-1]);bundle=dict(mode=mode,k=k,s=s,full=model,meta=mm,threshold=th,inner_sources=sources,outer_sources=osources);mp=OUT/f'model_{mode}_{k}_{s}.json';save(mp,bundle)
                for context,qq,aa,prob in [('outer',oq,oa,pr),('meta',q,a,meta)]:
                    df=qq[['row_id','farm','day','hour']].copy();df['high']=target(qq);df['A']=aa;df['prefix_A']=R.prefix(qq,aa);df['p']=prob;df['threshold']=th;df['mode']=mode;df['k']=k;df['s']=s;df['context']=context;fp=OUT/f'{context}_{mode}_{k}_{s}.csv';df.to_csv(fp,index=False);manifest.append(dict(path=fp.name,sha=sha(fp),model=mp.name,model_sha=sha(mp),mode=mode,k=k,s=s,context=context))
            frozen();print('CLASS_FIT',k,s,flush=True)
    assert len(manifest)==120;save(H/'receipt_v1.json',dict(status='COMPLETE240_FITS_NO_RMSE_BLEND',manifest=manifest,registration_sha=sha(H/'registration_v1.json')))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:refs,bank,p=prep();save(H/'preparation_v1.json',p);print('CLASS_PREP_PASS',flush=True)
    else:actual()
