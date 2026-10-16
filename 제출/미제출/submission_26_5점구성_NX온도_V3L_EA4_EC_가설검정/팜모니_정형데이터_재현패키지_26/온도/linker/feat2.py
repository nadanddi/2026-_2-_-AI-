from base import *
R=pickle.load(open(S+r"\base.pkl","rb")); r=R["recs"].reset_index(drop=True); X=R["OUT"]; IN_=R["IN"]; AC=R["ACT"]; EC=R["EC"]; ST=R["ST"]
FARM=r.farm.values; NUM=r.day.values; G=r.G.values; TRAIN=(r.split=="train").values
sdO=np.nanstd(X[TRAIN].reshape(-1,4),0)   # rule fix 10-11: training records only (no evaluation-input statistics)
VARS=np.concatenate([IN_,AC],2)  # 8 vars
sdV=np.nanstd(np.diff(VARS[TRAIN],axis=1).reshape(-1,8),0)+1e-6   # rule fix: training records only
idx={(f,d):i for i,(f,d) in enumerate(zip(FARM,NUM))}
def lab_cost(a_idx,b):
    pe=lambda A,i:A[i,23]+0.5*(A[i,23]-A[i,22]); pb=lambda A,j:A[j,0]-0.5*(A[j,1]-A[j,0])
    return np.abs(pe(EC,a_idx)-pb(EC,b))/0.01+np.abs(pe(ST,a_idx)-pb(ST,b))/0.3
def label_successor(labeled):
    """for each labeled record a: confident successor among labeled records (labels only). returns dict a->b"""
    L=np.where(labeled)[0]; succ={}
    pe=lambda A,i:A[i,23]+0.5*(A[i,23]-A[i,22]); pb=lambda A,j:A[j,0]-0.5*(A[j,1]-A[j,0])
    CE=np.abs(pe(EC,L)[:,None]-pb(EC,L)[None,:])/0.01+np.abs(pe(ST,L)[:,None]-pb(ST,L)[None,:])/0.3
    np.fill_diagonal(CE,np.inf)
    # predecessor view: for each b', best a; confident if margin; then a->b'
    for jb in range(len(L)):
        o=np.argsort(CE[:,jb]); c1,c2=CE[o[0],jb],CE[o[1],jb]
        if c1<=1.0 and c2-c1>=1.0:
            a=L[o[0]]
            if a in succ: succ[a]=-1  # conflict
            else: succ[a]=L[jb]
    return {a:b for a,b in succ.items() if b>=0}
def features(b,h,labeled,claimed=set(),removed=set()):
    """candidates: same farm, NUM<NUM[b]. labeled: bool mask of records whose labels are visible."""
    f=FARM[b]; nb=NUM[b]
    allowed=np.array([i for i in np.where((FARM==f)&(NUM<nb))[0] if i not in removed],dtype=int)
    if len(allowed)<2: return pd.DataFrame(),{}
    # date match of b using 0..h outdoor among allowed
    xb=X[b,:h+1]
    eq=(np.abs(X[allowed,:h+1]-xb)<1e-6).reshape(len(allowed),-1).mean(1)
    thr=1.0 if h<6 else 0.9
    mG=set(G[allowed[eq>=thr]])
    # seg1 calendar positions (same farm, allowed seg1 records + b if b seg1)
    s1=[i for i in allowed if NUM[i]<179]
    pos={}
    for i in s1: pos.setdefault(G[i],[]).append(NUM[i])
    if nb<179:
        gb_pos=[nb]+[p for g in mG for p in pos.get(g,[])]
        bpos=np.mean(gb_pos)
    else:
        ps=[p for g in mG for p in pos.get(g,[])]
        bpos=np.mean(ps) if ps else np.nan
    gpos={g:np.mean(v) for g,v in pos.items()}
    allp=sorted(set(list(gpos.values())+([bpos] if np.isfinite(bpos) else [])))
    rk={p:k for k,p in enumerate(allp)}
    excl=set()
    if labeled is not None:
        succ=label_successor(labeled)
        excl={a for a,bb in succ.items() if bb!=b}
    rows=[]
    pa=X[allowed,23]+0.5*(X[allowed,23]-X[allowed,22]); pbv=X[b,0]  # v2: only hour-0 input of b
    oj=np.sum(np.array([1,1,.3,.5])*np.abs(pa-pbv)/sdO,1)
    dv=np.abs(VARS[allowed,23]-VARS[b,0])/sdV  # 8
    dv=np.nan_to_num(dv,nan=2.0)
    for k,a in enumerate(allowed):
        ga=G[a]; pa_=gpos.get(ga,np.nan)
        if np.isfinite(bpos) and np.isfinite(pa_): ds=rk[bpos]-rk[pa_]; dsna=0
        else: ds=np.nan; dsna=1
        dn=nb-NUM[a]
        rows.append(dict(a=a,oj=oj[k],**{f"v{j}":min(dv[k,j],6) for j in range(8)},
            isprev=float(ds==1),dsdev=0 if dsna else min(abs(ds-1),6)/6,dsna=dsna,
            sameG=float(ga in mG),d1=float(dn==1),d2=float(dn==2),d3=float(dn==3),ldn=np.log(dn),
            excl=float(a in excl),d2w=float(dn==2)*dsna,d2c=float(dn==2)*(1-dsna),claimed=float(a in claimed),lab=float(labeled[a]) if labeled is not None else float(TRAIN[a])))
    return pd.DataFrame(rows), dict(nmatch=len(mG),bpos=bpos)
FEATS=["oj"]+[f"v{j}" for j in range(8)]+["isprev","dsdev","dsna","sameG","d1","d2w","d2c","d3","ldn","excl"]
def fit_clogit(groups,l2=1.0,iters=400,lr=0.1):
    """groups: list of (Xc [n,k], y index). returns weights (minimize -loglik). Adam."""
    k=groups[0][0].shape[1]; w=np.zeros(k); m=np.zeros(k); v=np.zeros(k)
    for t in range(1,iters+1):
        g=l2*w.copy()
        for Xc,y in groups:
            s=Xc@w; s-=s.max(); p=np.exp(s); p/=p.sum()
            g-= (Xc[y]-p@Xc)
        g/=len(groups)
        m=.9*m+.1*g; v=.999*v+.001*g*g
        w-=lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return w
