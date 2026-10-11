# -*- coding: utf-8 -*-
"""PA1 (2026-10-12 집 클로드; critic path A pre-check).  Step 1 DATE: previous-date component D' = argmin over date components c
(full-day outdoor >= 90% equal) that contain >= 1 LABELLED training record, of midnight outdoor cost from c's members' 23 h to the
target's 0 h (v7 oj formula), excluding t's own date.  Step 2 HOUSE: among labelled members of D', argmin fp_dist(t, a, h) (linker v6
fingerprint, t's hours 0..h).  Truth: certain links.  Training targets only (target label hidden = it is just the query).  Reports
date accuracy, house accuracy given correct date, overall, at h = 0 / 6 / 12, pass split; compared with linker v7m (mirror CV) accuracy.
Then counts evaluation records that would get a NEW anchor (no current confident anchor; top-1 margin rule fixed: fp gap to 2nd >= .3)."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
Xf=X.reshape(N,-1); par=list(range(N))
def fd(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i in range(N):
    eq=(np.abs(Xf[i+1:]-Xf[i])<1e-6).mean(1)
    for j in np.where(eq>=.9)[0]+i+1: par[fd(j)]=fd(i)
comp=np.array([fd(i) for i in range(N)])
rid=lambda i:f"{FARM[i]}_{NUM[i]:03d}"
T0=pd.read_pickle(S3+r"\truth.pkl"); T0=T0[T0.cert]; prev={int(b):int(a) for a,b in zip(T0.a1,T0.b)}
W_=np.array([1,1,.3,.5])
comps={}
for j in np.where(TRAIN)[0]: comps.setdefault(comp[j],[]).append(j)
def choose(t,h,lab_ok=lambda j: True):
    best=None;bc=np.inf
    for c,mem in comps.items():
        if c==comp[t]: continue
        mem2=[j for j in mem if j!=t and lab_ok(j)]
        if not mem2: continue
        a=mem2[0]; pa=X[a,23]+0.5*(X[a,23]-X[a,22]); cost=float(np.sum(W_*np.abs(pa-X[t,0])/sdO))
        if cost<bc: bc=cost;best=c
    mem=[j for j in comps[best] if j!=t and lab_ok(j)]
    sc=np.nan_to_num(np.array([F6.fp_dist(t,j,h) for j in mem]),nan=99); o=np.argsort(sc)
    marg=(sc[o[1]]-sc[o[0]]) if len(mem)>1 else 9
    return best,mem[o[0]],marg,bc
rows=[]
for t,a in prev.items():
    if not TRAIN[t] or not TRAIN[a]: continue
    for h in (0,6,12):
        c,j,marg,cost=choose(t,h)
        rows.append(dict(t=rid(t),h=h,p2=NUM[t]>=179,date_ok=c==comp[a],house_ok=j==a,marg=marg,cost=cost))
R=pd.DataFrame(rows)
for p2,G in R.groupby("p2"):
    print(f"pass2={p2}:"); print(G.groupby("h").agg(n=("t","size"),date_acc=("date_ok","mean"),overall=("house_ok","mean"),
        house_given_date=("house_ok",lambda s:(s&G.loc[s.index,"date_ok"]).sum()/max(G.loc[s.index,"date_ok"].sum(),1)),
        acc_marg03=("house_ok",lambda s:s[G.loc[s.index,"marg"]>=.3].mean()),cover_marg03=("marg",lambda s:(s>=.3).mean())).round(2).to_string())
L=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","cv_all400_choice_v7m.csv")).set_index("rec")
tr={rid(b):rid(a) for b,a in prev.items()}
Lk=L[L.index.isin(tr.keys())]; okk=Lk.prev==Lk.index.map(tr)
print("linker v7m baseline on same truth targets: acc %.2f (pass2 %.2f) | p>=.8 coverage %.2f acc %.2f"%(okk.mean(),okk[Lk.index.str[4:7].astype(int)>=179].mean(),(Lk.p>=.8).mean(),okk[Lk.p>=.8].mean()))

# --- PA1b (added): DATE from a training-label date graph (critic path A core) ---
# edges comp(a) -> comp(b) from certain links among TRAINING records, excluding any link that involves the target t (no leakage).
TLa=pd.read_pickle(S3+r"\truth.pkl"); TLa=TLa[TLa.cert]; links=[(int(a),int(b)) for a,b in zip(TLa.a1,TLa.b) if TRAIN[int(a)] and TRAIN[int(b)]]
from collections import Counter
rows=[]
for t,a in prev.items():
    if not TRAIN[t] or not TRAIN[a]: continue
    E=Counter(comp[x] for x,y in links if comp[y]==comp[t] and x!=t and y!=t and comp[x]!=comp[t])
    if not E: rows.append(dict(t=rid(t),p2=NUM[t]>=179,has=False)); continue
    D1=E.most_common(1)[0][0]
    mem=[j for j in comps.get(D1,[]) if j!=t]
    r={"t":rid(t),"p2":NUM[t]>=179,"has":True,"date_ok":D1==comp[a]}
    for h in (0,6,12):
        sc=np.nan_to_num(np.array([F6.fp_dist(t,j,h) for j in mem]),nan=99); r[f"house{h}"]=mem[int(np.argmin(sc))]==a if len(mem) else False
    rows.append(r)
Q=pd.DataFrame(rows)
for p2,G in Q.groupby("p2"):
    H=G[G.has]
    print(f"PA1b pass2={p2}: targets {len(G)}, date graph gives a predecessor date for {len(H)} ({100*len(H)/len(G):.0f}%) | date acc {H.date_ok.mean():.2f} | overall house acc h0 {H.house0.mean():.2f} h6 {H.house6.mean():.2f} h12 {H.house12.mean():.2f}")
