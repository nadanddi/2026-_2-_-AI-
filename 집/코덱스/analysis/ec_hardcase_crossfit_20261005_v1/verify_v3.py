import run_v1 as T
import math,importlib.util
sp=importlib.util.spec_from_file_location('scalar_verifier',T.H.parent/'ec_high_classifier_20261005_v1/verify_v1.py');V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V)
np,pd=T.np,T.pd

def scalar_proxy(tr,q,m):
    a,b=T.R.S.seasonal(tr,q,T.WV);x=b[T.COLS].to_numpy();tree=m['tree'];pred=[]
    for row in x:
        raw=math.fsum(T.P.M.tree_value(t['tree_structure'],row) for t in tree['tree_info']);pred.append(math.exp(raw))
    raw=np.array(pred);return np.clip(.5*raw+.5*T.R.prefix(q,raw),*m['bounds'])

def learning(q,x,m):
    assert m['ids']==T.R.ids(q.row_id) and m['x_sha']==T.R.ar(x) and m['y_sha']==T.R.ar(T.P.target(q));y=T.P.target(q);w,hard=T.weights(q,x,m['mode']);assert T.R.ar(w)==m['w_sha']
    mu=np.array([math.fsum(float(v) for v in x[:,j])/len(x) for j in range(x.shape[1])]);sd=np.sqrt(np.array([math.fsum((float(v)-float(u))**2 for v in x[:,j])/len(x) for j,u in enumerate(mu)]));sd[sd==0]=1;T.close(mu,m['mean']);T.close(sd,m['scale']);p=V.scalar(x,m);z=(x-mu)/sd;g=((w*(p-y))@z+np.array(m['coef']))/sum(w);gi=math.fsum(float(a)*(float(b)-int(c)) for a,b,c in zip(w,p,y))/sum(w);norm=max(float(abs(g).max()),abs(gi));assert norm<1e-5,norm;return norm

def sources(ref,q,src):
    rk=set(ref[['farm','day']].itertuples(index=False,name=None));qk=set(q[['farm','day']].itertuples(index=False,name=None));assert not rk&qk
    for z in src:
        f=z['row_id'][:3];assert all((f,d) in rk and (f,d) not in qk for d in z['days'])

def metrics(q,name):
    y=q.high.to_numpy(int);score=q.p.to_numpy();picked=q[name].to_numpy(bool);got=V.metrics(y,score,picked);base=V.metrics(y,q.prefix_A.to_numpy(),q.prefix_A.to_numpy()>=.9)
    high=score[y==1];low=score[y==0]
    if len(high) and len(low):
        pair=math.fsum(1 if a>b else .5 if a==b else 0 for a in high for b in low)/(len(high)*len(low));assert abs(pair-got['auc'])<1e-12
    return dict(model=got,baseline=base)

def main():
    T.frozen();refs,pr=T.prep();assert pr==T.load(T.H/'preparation_v1.json');rc=T.load(T.H/'receipt_v1.json');frames=[];grad=0.;error=0.;proxyerror=0.;repeats=0;sharedcache={};checked=set()
    for rec in rc['files']:
        v,k,s,mode,context=[rec[n] for n in ['v','k','s','mode','context']];fp=T.OUT/rec['path'];mp=T.OUT/rec['model'];assert T.sha(fp)==rec['sha'] and T.sha(mp)==rec['model_sha'];m=T.load(mp);shared=T.OUT/m['shared'];assert T.sha(shared)==m['shared_sha'];r=refs[v,k];tr,q=r['tr'],r['q'];key=v,k,s
        if key not in sharedcache:
            b=T.load(shared);ss=T.splits(tr);assert b['split']==ss;xx=np.full((len(tr),22),np.nan);px=np.full(len(tr),np.nan)
            for j,(z,proxy) in enumerate(zip(ss,b['proxy'])):
                ti,vi=np.array(z['ti']),np.array(z['vi']);a,c=tr.iloc[ti].reset_index(drop=True),tr.iloc[vi].reset_index(drop=True);ban={(f,int(d)+o) for f,d in c[['farm','day']].itertuples(index=False,name=None) for o in [-1,0,1]};assert not set(a[['farm','day']].itertuples(index=False,name=None))&ban;assert not set(a.row_id)&set(q.row_id);p=T.proxy_replay(a,c,proxy);x,src=T.design(a,c,p);assert src==b['inner_sources'][j];sources(a,c,src);xx[vi]=x;px[vi]=p
                if (v,k,s,j)==('DIAG10',0,7,0):
                    small=c.iloc[:12].reset_index(drop=True);sm=dict(proxy);sm['query_ids']=T.R.ids(small.row_id);aa,bb=T.R.S.seasonal(a,small,T.WV);sm['query_x_sha']=T.R.ar(bb[T.COLS].to_numpy());manual=scalar_proxy(a,small,sm);proxyerror=max(proxyerror,T.close(p[:12],manual))
                    fresh,fm=T.proxy_fit(a,c,s);T.close(fresh,p);assert fm==proxy;repeats+=1
            op=T.proxy_replay(tr,q,b['proxy_full']);ox,src=T.design(tr,q,op);assert src==b['outer_sources'];sources(tr,q,src)
            if (v,k,s)==('DIAG10',0,7):
                fresh,fm=T.proxy_fit(tr,q,s);T.close(fresh,op);assert fm==b['proxy_full'];repeats+=1
            sharedcache[key]=(xx,px,ox,op)
        xx,px,ox,op=sharedcache[key];qq,a,x=(q,op,ox) if context=='outer' else (tr,px,xx);df=pd.read_csv(fp,float_precision='round_trip');assert np.array_equal(df.row_id,qq.row_id) and np.array_equal(df.high,T.P.target(qq));T.close(a,df.proxy);T.close(x[:,1],df.prefix_proxy);base=r['baseline'][s] if context=='outer' else a;T.close(base,df.A);T.close(T.R.prefix(qq,base),df.prefix_A);p=V.scalar(x,m['classifier']);error=max(error,T.close(p,df.p));assert np.array_equal(p>=.5,df.DIRECT.to_numpy(bool));assert np.array_equal((p>=.5)&(df.prefix_A.to_numpy()>=.9),df.GUARD.to_numpy(bool))
        assert np.array_equal(((df.high==1)&(df.prefix_proxy<1.2)).to_numpy(int),df.hard_high) and np.array_equal(((df.high==0)&(df.prefix_proxy>=.9)).to_numpy(int),df.hard_low)
        if (v,k,s,mode) not in checked:
            grad=max(grad,learning(tr,xx,m['classifier']));checked.add((v,k,s,mode))
            if (v,k,s)==('DIAG10',0,7):assert T.classifier(tr,xx,s,mode)==m['classifier'];repeats+=1
        df['v']=v;df['k']=k;df['seed']=s;df['mode']=mode;df['context']=context;frames.append(df);print('HARD_VERIFY',v,k,s,mode,context,flush=True)
    allrows=pd.concat(frames,ignore_index=True);results=[];segments=[];support=[];foldmetrics=[]
    for (v,s,mode,context),g in allrows.groupby(['v','seed','mode','context']):
        if v=='DIAG10' and context=='outer':assert len(g)==8640 and g.row_id.is_unique
        for h in [0,6,12,23]:
            z=g[g.hour==h]
            for name in ['DIRECT','GUARD']:results.append(dict(v=v,seed=int(s),mode=mode,context=context,hour=h,selection=name,**metrics(z,name)))
        if context=='outer':
            for (f,phase),z in g[g.hour==23].assign(phase=lambda d:(d.day>=179).astype(int)).groupby(['farm','phase']):segments.append(dict(v=v,seed=int(s),mode=mode,farm=f,phase=int(phase),**metrics(z,'GUARD')))
            for k,z in g[g.hour==23].groupby('k'):foldmetrics.append(dict(v=v,k=int(k),seed=int(s),mode=mode,**metrics(z,'GUARD')))
    for rec in rc['files']:
        if rec['context']!='train' or rec['mode']!='UNIFORM':continue
        z=allrows[(allrows.v==rec['v'])&(allrows.k==rec['k'])&(allrows.seed==rec['s'])&(allrows.context=='train')&(allrows['mode']=='UNIFORM')&(allrows.hour==23)]
        support.append(dict(v=rec['v'],k=rec['k'],seed=rec['s'],n=len(z),high=int(z.high.sum()),hard_high=int(z.hard_high.sum()),hard_low=int(z.hard_low.sum())))
    primary={}
    for mode in T.MODES:
        cells=[r for r in results if r['mode']==mode and r['context']=='outer' and r['hour']==23 and r['selection']=='GUARD'];nonworse=sum(z['model']['tp']>=z['baseline']['tp'] and z['model']['fp']<=z['baseline']['fp'] for z in cells);strict=sum(z['model']['fp']<z['baseline']['fp'] for z in cells if z['v']=='DIAG10');primary[mode]=dict(nonworse=nonworse,total=len(cells),diag_strict=strict,passed=nonworse==9 and strict==3)
    T.save(T.H/'verification_v3.json',dict(status='PASS_NESTED_OOF_LEARNING_REPLAY',max_error=error,proxy_scalar_error=proxyerror,max_gradient=grad,repeats=repeats,files=len(frames),classifiers=len(checked),proxy_models=len(sharedcache)*5,results=results,segments=segments,foldmetrics=foldmetrics,support=support,primary=primary,adoption=False));print('HARD_VERIFY_PASS',primary,flush=True)
if __name__=='__main__':main()


