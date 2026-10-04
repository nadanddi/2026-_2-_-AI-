"""Independent arithmetic/checkpoint replay. No torch, runner or model fitting.
Loss, causal shrinkage and bootstrap use independent scalar/NumPy methods.
"""
from pathlib import Path
import sys, math, hashlib, json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
ALPHA=.025/24
CHECKS=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def near(a,b,tol=1e-12):
    global CHECKS
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    gap=float(np.max(abs(a-b)));assert gap<=tol,(gap,tol)
    CHECKS+=a.size;return gap

def scalar_final(frame,raw,lo,hi):
    assert len(frame)==len(raw) and np.isfinite(raw).all() and math.isfinite(lo) and math.isfinite(hi) and lo<=hi
    result=np.empty(len(raw));hist={}
    ix=sorted(range(len(raw)),key=lambda i:(frame.farm.iloc[i],int(frame.day.iloc[i]),int(frame.hour.iloc[i])))
    seen=set()
    for i in ix:
        key=(frame.farm.iloc[i],int(frame.day.iloc[i]));keyh=(*key,int(frame.hour.iloc[i]))
        assert keyh not in seen;seen.add(keyh)
        old=hist.setdefault(key,[]);old.append(float(raw[i]))
        result[i]=min(hi,max(lo,math.fsum((.5*float(raw[i]),.5*math.fsum(old)/len(old)))))
    return result

def matrix_final(frame,raw,lo,hi):
    result=np.empty(len(raw))
    for _,ix in frame.reset_index(drop=True).groupby(['farm','day']).groups.items():
        ix=list(sorted(ix,key=lambda i:frame.hour.iloc[i]));n=len(ix)
        s=.5*np.eye(n)+.5*np.tril(np.ones((n,n)))/np.arange(1,n+1)[:,None]
        result[ix]=np.clip(s@np.asarray(raw)[ix],lo,hi)
    return result

def replay_checkpoint(z,training_x,query_x,features):
    assert list(z['feature_names'])==features and len(features)==14 and features[-1]=='season' and 'day' not in features
    a=np.asarray(training_x,float);q=np.asarray(query_x,float)
    assert a.shape[1]==q.shape[1]==14 and not np.isinf(a).any() and not np.isinf(q).any()
    med=np.nanmedian(a,axis=0);med=np.where(np.isnan(med),0.,med);assert np.isfinite(med).all()
    med=np.where(np.isnan(med),0.,med); near(med,z['imputer_statistics'])
    a=np.where(np.isnan(a),med,a);q=np.where(np.isnan(q),med,q)
    mean=np.mean(a,axis=0);var=np.mean((a-mean)**2,axis=0)
    near(mean/np.maximum(1.,abs(mean)),z['scaler_mean']/np.maximum(1.,abs(mean)));near(var/np.maximum(1.,abs(var)),z['scaler_var']/np.maximum(1.,abs(var)))
    scale=np.sqrt(var)
    # StandardScaler's constant-feature guard; float64 roundoff bounded by .eps.
    eps=np.finfo(float).eps;n=len(a)
    constant=var<=n*eps*var+(n*mean*eps)**2
    scale[constant]=1.;near(scale/np.maximum(1.,abs(scale)),z['scaler_scale']/np.maximum(1.,abs(scale)))
    assert int(np.asarray(z['scaler_n_samples_seen']).item())==len(a)
    out=(q-z['scaler_mean'])/z['scaler_scale']
    shapes={0:((64,14),(64,)),2:((32,64),(32,)),4:((1,32),(1,))}
    for k,(ws,bs) in shapes.items():
        w=z[f'model__{k}.weight'];b=z[f'model__{k}.bias']
        assert w.shape==ws and b.shape==bs and w.dtype==b.dtype==np.dtype('float64')
        assert np.isfinite(w).all() and np.isfinite(b).all()
        out=out@w.T+b
        if k!=4:out=np.maximum(out,0.)
    assert out.shape==(len(q),1) and np.isfinite(out).all()
    return out[:,0]

def rmse(y,p):
    a=np.asarray(y,float);b=np.asarray(p,float)
    scalar=math.sqrt(math.fsum((float(x)-float(z))**2 for x,z in zip(a,b))/len(a))
    vector=float(np.sqrt(np.mean((a-b)**2)));near([scalar],[vector]);return scalar

def qmanual(x,p):
    x=sorted(map(float,x));a=(len(x)-1)*p;i=math.floor(a);f=a-i
    return x[i] if i==len(x)-1 else (1-f)*x[i]+f*x[i+1]

def bootstrap(frame,seed,comparison='baseline'):
    assert len(frame)==8640 and seed in (7,101,2024) and set(frame.farm)=={'F13','F47'}
    rec={}
    for r in frame.itertuples():
        val=(float(r.candidate)-float(r.y))**2-(float(getattr(r,comparison))-float(r.y))**2
        rec.setdefault((r.farm,int(r.day)),[]).append((int(r.hour),val))
    assert len(rec)==360 and all(len(g)==24 and {h for h,_ in g}==set(range(24)) for g in rec.values())
    rng=np.random.default_rng(20261003+seed);total=np.zeros(20000);count=np.zeros(20000);draws={};blocks={}
    for farm in ('F13','F47'):
        days=sorted(d for f,d in rec if f==farm)
        blocks[farm]=[(math.fsum(v for d in days[i:i+5] for _,v in rec[farm,d]),sum(len(rec[farm,d]) for d in days[i:i+5])) for i in range(0,len(days),5)]
        g=blocks[farm];draws[farm]=rng.integers(len(g),size=(20000,len(g)))
        sums=np.array([v for v,n in g]);n=np.array([n for v,n in g])
        total+=sums[draws[farm]].sum(axis=1);count+=n[draws[farm]].sum(axis=1)
    sample=total/count
    # Independent fsum per draw and manually interpolated quantiles.
    scalar=[]
    for i in range(20000):
        vals=[];n=0
        for farm in ('F13','F47'):
            for j in draws[farm][i]:v,num=blocks[farm][j];vals.append(v);n+=num
        scalar.append(math.fsum(vals)/n)
    near(sample,scalar)
    qs=[ALPHA,1-ALPHA,.025,.975]
    a=np.quantile(sample,qs);b=[qmanual(scalar,q) for q in qs];near(a,b);assert (a[1]<0)==(b[1]<0)
    pw=float(np.mean(sample>=0));assert pw==sum(s>=0 for s in scalar)/20000
    return dict(p_worse=pw,ci_adjusted=a[:2].tolist(),ci95=a[2:].tolist(),draws=20000,
      rng_seed=20261003+seed,blocks={f:len(g) for f,g in blocks.items()},alpha=ALPHA,
      pass_gate=bool(pw<ALPHA and a[1]<0),block_unit='5 ordered observed days per farm')

def synthetic():
    rng=np.random.default_rng(20261004)
    f=pd.DataFrame(dict(farm=['F13']*24+['F47']*24,day=[1]*24+[3]*24,hour=list(range(24))*2))
    raw=rng.uniform(-2,4,48);gap=near(scalar_final(f,raw,.1,2.8),matrix_final(f,raw,.1,2.8))
    perm=rng.permutation(48);near(scalar_final(f.iloc[perm],raw[perm],.1,2.8),scalar_final(f,raw,.1,2.8)[perm])
    causal=[]
    for farm in ('F13','F47'):
        for h in (0,6,12):
            m=(f.farm==farm)&(f.hour<=h);changed=raw.copy();changed[~m]+=1000
            causal.append(near(scalar_final(f,changed,.1,2.8)[m],scalar_final(f,raw,.1,2.8)[m]))
    # Fixture contains 360 days/8640 rows with independently chosen mixed loss signs.
    rs=[]
    for farm in ('F13','F47'):
        for d in range(180):
            for h in range(24):
                y=.5+.001*d;b=y+.1+.003*math.sin(d);c=b+(.01 if d%3==0 else -.007)
                rs.append((farm,d,h,y,b,c))
    b=bootstrap(pd.DataFrame(rs,columns=['farm','day','hour','y','baseline','candidate']),7)
    z=dict(status='PASS_SYNTHETIC',checks=CHECKS,scalar_matrix_maxdiff=gap,prefix=causal,
      bootstrap=b,source_sha256=sha(Path(__file__)),fit=0,EC_reads=0)
    with (H/'verification_math_synthetic_v4.json').open('x',encoding='utf-8') as out:json.dump(z,out,indent=2)
    print('PASS_SYNTHETIC',CHECKS,gap)
if __name__=='__main__':synthetic()
