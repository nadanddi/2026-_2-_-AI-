# package copy of research linker_v7/feat7.py
# linker v7 (2026-10-12 집 클로드): v5 (independent per-record decisions, no claimed / chains) + OTHER-GREENHOUSE TRAINING records
# as previous-date candidates.  Motivation: 25 % of label-certain true predecessors are in the other greenhouse ID (houses move
# between the F13 / F47 IDs), and inside evaluation chunks the same-ID previous date is usually an unlabelled evaluation record.
# Rule: other greenhouse TRAINING inputs / labels allowed (user ruling 2026-10-11); other greenhouse EVALUATION inputs and
# same-greenhouse later-numbered inputs never used (visible() unchanged).  Truth now keeps cross-ID links.
from feat4 import *
import feat4 as F4
S7=S4
TL7=pd.read_pickle(S3+r"\truth.pkl"); TL7=TL7[TL7.cert][["b","a1"]].rename(columns={"a1":"a"})
FEATS7=FEATS+["xf"]
def features7(b,h,labeled,removed=set()):
    f=FARM[b]; nb=NUM[b]
    vis=visible(b,removed); key=vis.tobytes()
    if key not in F4._cal_cache:
        comp=components(vis); gpos,mp=calendar(vis,comp); F4._cal_cache[key]=(comp,gpos,mp)
    comp,gpos,mp=F4._cal_cache[key]
    vi=np.where(vis)[0]
    eq=(np.abs(X[vi,:h+1]-X[b,:h+1])<1e-6).reshape(len(vi),-1).mean(1)
    thr=1.0 if h<6 else 0.9
    mC={comp[i] for i in vi[eq>=thr]}
    ps=[gpos[c] for c in mC if c in gpos]
    if nb<179: ps=ps+[mp(b)]
    bpos=np.mean(ps) if ps else np.nan
    allp=sorted(set(list(gpos.values())+([bpos] if np.isfinite(bpos) else []))); rk={p:k for k,p in enumerate(allp)}
    cand=np.array([i for i in vi if comp[i] not in mC],dtype=int)   # same farm (earlier) + other farm (training), never same date
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
        xf=float(FARM[a]!=f); dn=(nb-NUM[a]) if not xf else 0
        rows.append(dict(a=a,oj=oj[k],**{f"v{j}":min(dv[k,j],6) for j in range(8)},
            isprev=float(ds==1),dsdev=0 if dsna else min(abs(ds-1),6)/6,dsna=dsna,
            d1=float(dn==1),d3=float(dn==3),ldn=np.log(dn) if dn>0 else 0.0,excl=float(a in excl),
            d2w=float(dn==2)*dsna,d2c=float(dn==2)*(1-dsna),xf=xf))
    # drop same-farm candidates with number >= b (never visible anyway) and other-farm candidates that are clearly not previous date
    D=pd.DataFrame(rows)
    D=D[(D.xf==0)|(D.oj<6)]          # other farm: keep only outdoor-midnight-plausible previous dates (cost cap fixed a priori)
    return D.reset_index(drop=True), dict(nmatch=len(mC),bpos=bpos)
def train_groups7():
    TRG=[]
    for b,a in zip(TL7.b,TL7.a):
        lab=TRAIN.copy(); lab[(FARM==FARM[b])&(np.abs(NUM-NUM[b])<=1)]=False
        df,_=features7(b,0,lab,removed=TEST_IDX)
        if len(df)==0: continue
        y=np.where(df.a.values==a)[0]
        if len(y): TRG.append((b,a,df[FEATS7].values.astype(float),y[0]))
    return TRG
def pick7(b,h,lab,removed,w):
    df,_=features7(b,h,lab,removed=removed)
    if len(df)==0: return None,0.0
    s=df[FEATS7].values.astype(float)@w; p=np.exp(s-s.max()); p/=p.sum(); k=int(np.argmax(p)); return int(df.a.values[k]),float(p[k])
