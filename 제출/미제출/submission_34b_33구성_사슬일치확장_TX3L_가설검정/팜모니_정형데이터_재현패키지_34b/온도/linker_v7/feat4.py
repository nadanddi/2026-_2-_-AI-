# package copy of research linker_v5/feat4.py (paths via LINKER_WORK / LINKER5_OUT)
# linker v4 (2026-10-11 집 클로드): joint two-greenhouse date calendar from TRAINING records + same-date exclusion.
# Rule (user decision 10-11): other greenhouse TRAINING inputs/labels allowed; other greenhouse EVALUATION inputs and
# same-greenhouse later-numbered inputs forbidden.
import sys, os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, pickle
HERE=os.path.dirname(os.path.abspath(__file__))
S3=os.environ.get("LINKER_WORK", os.path.join(HERE, "..", "linker", "work"))   # base.pkl, truth.pkl from 온도/linker (base, t1, t3)
S4=os.environ.get("LINKER5_OUT", os.path.join(HERE, "work")); os.makedirs(S4, exist_ok=True)
R=pickle.load(open(S3+r"\base.pkl","rb")); r=R["recs"].reset_index(drop=True)
X=R["OUT"].copy(); IN_=R["IN"].copy(); AC=R["ACT"].copy(); EC=R["EC"]; ST=R["ST"]
FARM=r.farm.values; NUM=r.day.values; TRAIN=(r.split=="train").values; N=len(r)
sdO=np.nanstd(X[TRAIN].reshape(-1,4),0)
VARS=np.concatenate([IN_,AC],2)
sdV=np.nanstd(np.diff(VARS[TRAIN],axis=1).reshape(-1,8),0)+1e-6
def refresh():   # call after perturbing X/IN_/AC in tests
    global VARS; VARS=np.concatenate([IN_,AC],2)
def label_successor(labeled):
    L=np.where(labeled)[0]; succ={}
    pe=lambda A,i:A[i,23]+0.5*(A[i,23]-A[i,22]); pb=lambda A,j:A[j,0]-0.5*(A[j,1]-A[j,0])
    CE=np.abs(pe(EC,L)[:,None]-pb(EC,L)[None,:])/0.01+np.abs(pe(ST,L)[:,None]-pb(ST,L)[None,:])/0.3
    np.fill_diagonal(CE,np.inf)
    for jb in range(len(L)):
        o=np.argsort(CE[:,jb]); c1,c2=CE[o[0],jb],CE[o[1],jb]
        if c1<=1.0 and c2-c1>=1.0:
            a=L[o[0]]
            succ[a]=-1 if a in succ else L[jb]
    return {a:b for a,b in succ.items() if b>=0}
_succ_cache={}
def succ_for(labeled):
    k=labeled.tobytes()
    if k not in _succ_cache: _succ_cache[k]=label_successor(labeled)
    return _succ_cache[k]
def visible(b,removed):
    """records whose INPUTS may be used for target b: same farm & NUM<NUM[b] (any split), other farm TRAINING only."""
    f=FARM[b]; v=((FARM==f)&(NUM<NUM[b])) | ((FARM!=f)&TRAIN)
    if removed: v[list(removed)]=False
    v[b]=False
    return v
def components(vis):
    """date components among visible records: union of pairs with >=70% equal outdoor cells (computed on the fly)."""
    ids=np.where(vis)[0]; par={i:i for i in ids}
    def fd(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    Xv=X[ids].reshape(len(ids),-1)
    for k,i in enumerate(ids):
        eq=(np.abs(Xv[k+1:]-Xv[k])<1e-6).mean(1)
        for j in ids[k+1:][eq>0.7]: par[fd(j)]=fd(i)
    return {i:fd(i) for i in ids}
def calendar(vis,comp):
    """joint position on F13-number scale from seg1 (<179) visible records of both farms."""
    s1=[i for i in np.where(vis)[0] if NUM[i]<179]
    by={}
    for i in s1: by.setdefault(comp[i],[]).append(i)
    pairs=[(np.mean([NUM[i] for i in v if FARM[i]=="F47"]),np.mean([NUM[i] for i in v if FARM[i]=="F13"])) for v in by.values()
           if any(FARM[i]=="F13" for i in v) and any(FARM[i]=="F47" for i in v)]
    pairs=sorted(pairs); px=np.array([p[0] for p in pairs]); py=np.array([p[1] for p in pairs])
    # robust monotone map F47 -> F13 scale (drop pairs with |offset - local median| > 6)
    off=py-px; med=pd.Series(off).rolling(7,center=True,min_periods=1).median().values; ok=np.abs(off-med)<=6
    px,py=px[ok],py[ok]
    m47=lambda n: float(np.interp(n,px,py)) if len(px) else float(n)
    def mp(i): return float(NUM[i]) if FARM[i]=="F13" else m47(NUM[i])
    def cpos(v):
        a=[NUM[i] for i in v if FARM[i]=="F13"]
        return float(np.median(a)) if a else float(np.median([mp(i) for i in v]))   # prefer F13 seg1 numbers
    gpos={c:cpos(v) for c,v in by.items()}
    return gpos,mp
_cal_cache={}
def features(b,h,labeled,claimed=set(),removed=set()):
    f=FARM[b]; nb=NUM[b]
    vis=visible(b,removed)
    key=(vis.tobytes())
    if key not in _cal_cache:
        comp=components(vis); gpos,mp=calendar(vis,comp); _cal_cache[key]=(comp,gpos,mp)
    comp,gpos,mp=_cal_cache[key]
    vi=np.where(vis)[0]
    # date of b from its own 0..h outdoor vs visible records
    eq=(np.abs(X[vi,:h+1]-X[b,:h+1])<1e-6).reshape(len(vi),-1).mean(1)
    thr=1.0 if h<6 else 0.9
    mC={comp[i] for i in vi[eq>=thr]}
    ps=[gpos[c] for c in mC if c in gpos]
    if nb<179: ps=ps+[mp(b)]
    bpos=np.mean(ps) if ps else np.nan
    allp=sorted(set(list(gpos.values())+([bpos] if np.isfinite(bpos) else [])))
    rk={p:k for k,p in enumerate(allp)}
    cand=np.array([i for i in vi if FARM[i]==f and comp[i] not in mC],dtype=int)   # same farm, NOT same date (same-date exclusion)
    if len(cand)<2: return pd.DataFrame(),{}
    excl=set()
    if labeled is not None:
        excl={a for a,bb in succ_for(labeled).items() if bb!=b}
    pa=X[cand,23]+0.5*(X[cand,23]-X[cand,22]); pbv=X[b,0]
    oj=np.sum(np.array([1,1,.3,.5])*np.abs(pa-pbv)/sdO,1)
    dv=np.nan_to_num(np.abs(VARS[cand,23]-VARS[b,0])/sdV,nan=2.0)
    rows=[]
    for k,a in enumerate(cand):
        pa_=gpos.get(comp[a],np.nan)
        if np.isfinite(bpos) and np.isfinite(pa_): ds=rk[bpos]-rk[pa_]; dsna=0
        else: ds=np.nan; dsna=1
        dn=nb-NUM[a]
        rows.append(dict(a=a,oj=oj[k],**{f"v{j}":min(dv[k,j],6) for j in range(8)},
            isprev=float(ds==1),dsdev=0 if dsna else min(abs(ds-1),6)/6,dsna=dsna,
            d1=float(dn==1),d3=float(dn==3),ldn=np.log(dn),excl=float(a in excl),
            d2w=float(dn==2)*dsna,d2c=float(dn==2)*(1-dsna),datecomp=comp[a]))
    return pd.DataFrame(rows), dict(nmatch=len(mC),bpos=bpos)
FEATS=["oj"]+[f"v{j}" for j in range(8)]+["isprev","dsdev","dsna","d1","d2w","d2c","d3","ldn","excl"]   # sameG removed (hard same-date exclusion)
def fit_clogit(groups,l2=1.0,iters=400,lr=0.1):
    k=groups[0][0].shape[1]; w=np.zeros(k); m=np.zeros(k); v=np.zeros(k)
    for t in range(1,iters+1):
        g=l2*w.copy()
        for Xc,y in groups:
            s=Xc@w; s-=s.max(); p=np.exp(s); p/=p.sum(); g-=(Xc[y]-p@Xc)
        g/=len(groups); m=.9*m+.1*g; v=.999*v+.001*g*g
        w-=lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return w
def score(df,w,claimed):
    s=df[FEATS].values.astype(float)@w; s=np.where(np.isin(df.a.values,list(claimed)),-np.inf,s)
    p=np.exp(s-s.max()); p/=p.sum(); return p
def pick(b,h,lab,removed,claimed,w):
    df,_=features(b,h,lab,removed=removed)
    if len(df)==0: return None,0.0
    p=score(df,w,claimed); k=int(np.argmax(p)); return int(df.a.values[k]),float(p[k])
rid=lambda i:f"{FARM[i]}_{NUM[i]:03d}"
TL=pd.read_pickle(S3+r"\truth.pkl"); TL=TL[TL.cert&(TL.fa==TL.fb)][["b","a1"]].rename(columns={"a1":"a"})
TEST_IDX=set(np.where(~TRAIN)[0].tolist())
def train_groups():
    TRG=[]
    for b,a in zip(TL.b,TL.a):
        lab=TRAIN.copy(); lab[(FARM==FARM[b])&(np.abs(NUM-NUM[b])<=1)]=False
        df,_=features(b,0,lab,removed=TEST_IDX)
        if len(df)==0: continue
        y=np.where(df.a.values==a)[0]
        if len(y): TRG.append((b,a,df[FEATS].values.astype(float),y[0]))
    return TRG
