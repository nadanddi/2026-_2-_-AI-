"""Family27 direct-MSE sigmoid mixture; immutable artifacts; no test predictions."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
spec=importlib.util.spec_from_file_location('anchor_source',H.parent/'ec_anchor_trust_20261005_v2/run_v6.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local'/H.name
CFG=dict(family=27,alpha=.025/27,seeds=R.SEEDS,keys=R.KEYS,penalty=.01,prior=.05,maxiter=3000,ftol=1e-13,gtol=1e-8,maxls=50,judged=['DIAG10','A','B'],all_passes=True,EL1_scored=False)
sha=R.sha;save=R.save;load=R.load;close=R.close
def pair(ref,seed,inner):
    q=ref['b'] if inner else ref['q'];a=ref['inner'][seed] if inner else ref['baseline'][seed]
    N=ref['NB'] if inner else ref['NQ'];Q=ref['QB'] if inner else ref['QQ'];bounds=ref['innerbounds'] if inner else ref['bounds']
    x,_,_=R.design(q,a,N,Q);ok=N[:,5]>0
    b=np.clip(a+np.where(ok,.5*np.clip(N[:,0]-R.prefix(q,a),-.6,.6),0),*bounds)
    assert np.isfinite(x).all() and np.array_equal(b[~ok],a[~ok]);return q,a,b,x,ok
def objective(t,z,a,d,y,t0):
    g=expit(z@t);r=a+g*d-y
    f=float(np.mean(r*r)+CFG['penalty']*np.sum((t-t0)**2))
    grad=2*z.T@(r*d*g*(1-g))/len(y)+2*CFG['penalty']*(t-t0)
    return f,grad
def toy():
    rng=np.random.default_rng(273);z=np.column_stack([np.ones(31),rng.normal(size=(31,4))]);a=rng.normal(size=31);d=rng.normal(size=31);y=rng.normal(size=31);t=rng.normal(size=5);t0=np.zeros(5)
    f,g=objective(t,z,a,d,y,t0);fd=[]
    for j in range(5):
        e=np.zeros(5);e[j]=1e-6;fd.append((objective(t+e,z,a,d,y,t0)[0]-objective(t-e,z,a,d,y,t0)[0])/2e-6)
    err=float(np.max(np.abs(g-fd)));assert err<1e-8
    return dict(gradient_error=err,synthetic_rows=31)
def prep():
    assert sha(H.parent/'ec_anchor_trust_20261005_v2/run_v6.py')==load(H.parent/'ec_anchor_trust_20261005_v2/registration_v1.json')['run_sha']
    refs,p=R.preparation();assert p==load(H.parent/'ec_anchor_trust_20261005_v2/preparation_v6.json')
    records=[]
    for (v,k),ref in refs.items():
        for s in R.SEEDS:
            b,a,sp,x,ok=pair(ref,s,True);q,aq,sq,xq,okq=pair(ref,s,False)
            assert not set(b.row_id)&set(q.row_id)
            records.append(dict(validator=v,fold=k,seed=s,inner_ids=R.ids(b.row_id),outer_ids=R.ids(q.row_id),inner_a=R.ar(a),inner_b=R.ar(sp),inner_x=R.ar(x),outer_a=R.ar(aq),outer_b=R.ar(sq),outer_x=R.ar(xq),inner_eligible=int(ok.sum()),outer_eligible=int(okq.sum())))
    receipt=dict(status='PREPARED_FIT0_SCORE0',config=CFG,source=sha(Path(__file__)),prior_preparation_sha=sha(H.parent/'ec_anchor_trust_20261005_v2/preparation_v6.json'),prior=p,records=records,synthetic=toy())
    return refs,receipt
def fit(ref,s):
    b,a,sp,x,ok=pair(ref,s,True);q,aq,sq,xq,okq=pair(ref,s,False)
    if not ok.any() or np.all(sp[ok]==a[ok]):
        m=dict(empty=True,n=int(ok.sum()),reason='no eligible or zero direction',theta=None);return aq.copy(),np.zeros(len(q)),m
    scale=StandardScaler().fit(x[ok]);z=np.column_stack([np.ones(ok.sum()),scale.transform(x[ok])]);t0=np.zeros(z.shape[1]);t0[0]=math.log(CFG['prior']/(1-CFG['prior']))
    y=b.sub_ec.to_numpy()[ok];d=sp[ok]-a[ok]
    opt=minimize(objective,t0,args=(z,a[ok],d,y,t0),jac=True,method='L-BFGS-B',options={k:CFG[k] for k in ['maxiter','ftol','gtol','maxls']})
    f,grad=objective(opt.x,z,a[ok],d,y,t0);assert opt.success and np.max(abs(grad))<=1e-6,(opt.message,np.max(abs(grad)))
    m=dict(empty=False,n=int(ok.sum()),ids=R.ids(b.row_id[ok]),mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),theta=opt.x.tolist(),prior=t0.tolist(),objective=f,gradient_max=float(np.max(abs(grad))),iterations=int(opt.nit),message=str(opt.message),train_mse=float(np.mean((a[ok]+expit(z@opt.x)*d-y)**2)),train_a_mse=float(np.mean((a[ok]-y)**2)),train_b_mse=float(np.mean((sp[ok]-y)**2)))
    g=replay(xq,okq,m);p=aq+g*(sq-aq);assert np.array_equal(p[~okq],aq[~okq]);return p,g,m
def replay(x,ok,m):
    if m['empty']:return np.zeros(len(x))
    z=np.column_stack([np.ones(len(x)),(x-np.asarray(m['mean']))/m['scale']]);return np.where(ok,expit(z@np.asarray(m['theta'])),0)
def frozen():
    reg=load(H/'registration_v1.json')
    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name
    return reg
def actual():
    frozen();refs,receipt=prep();assert receipt==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);manifest=[]
    for (v,k),ref in refs.items():
        for s in R.SEEDS:
            frozen()
            with threadpool_limits(limits=2):p,g,m=fit(ref,s)
            q,a,b,x,ok=pair(ref,s,False);close(g,replay(x,ok,m));close(g,replay(x[::-1],ok[::-1],m)[::-1]);close(g,np.array([replay(x[i:i+1],ok[i:i+1],m)[0] for i in range(len(x))]));assert np.all((g>=0)&(g<=1))
            if (v,k,s)==('DIAG10',0,7):
                with threadpool_limits(limits=2):p2,g2,m2=fit(ref,s)
                close(p,p2);close(g,g2);assert m==m2
                altered=x[:8].copy();altered[1:]+=1e4;close(g[:1],replay(altered,ok[:8],m)[:1]);save(H/'first_audit_v1.json',dict(status='PASS',repeat_error=close(p,p2),query_independent=True,synthetic=toy()))
            df=q[['row_id','farm','day','hour']].copy();df['y']=q.sub_ec;df['A']=a;df['B']=b;df['g']=g;df['candidate']=p;df['eligible']=ok;df['validator']=v;df['fold']=k;df['seed']=s
            path=OUT/f'{v}_{k}_{s}.csv';df.to_csv(path,index=False);save(OUT/f'{v}_{k}_{s}_fit.json',m);manifest.append(dict(validator=v,fold=k,seed=s,csv_sha=sha(path),fit_sha=sha(OUT/f'{v}_{k}_{s}_fit.json')))
        print('SOFT',v,k,'COMPLETE',flush=True)
    assert len(manifest)==66
    agg=pd.concat([pd.read_csv(OUT/f'{v}_{k}_{s}.csv',float_precision='round_trip') for v,k in R.KEYS for s in R.SEEDS],ignore_index=True);agg.to_csv(OUT/'oof.csv',index=False)
    save(H/'fit_receipt_v1.json',dict(status='COMPLETE66_SCORE0',manifest=manifest,rows=len(agg),aggregate_sha=sha(OUT/'oof.csv'),registration_sha=sha(H/'registration_v1.json')));print('ALL66_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:
        refs,receipt=prep();save(H/'preparation_v1.json',receipt);print('PREPARATION_PASS',flush=True)
    else:actual()
