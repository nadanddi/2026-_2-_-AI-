# -*- coding: utf-8 -*-
"""KD1 (2026-10-12 집 클로드; "EC 0시 지식 총동원"): can a same-house record k days back (k = 1..5) be identified WITHOUT midnight
continuity, by operation fingerprint only?  Oracle date: candidates = labelled training records on the TRUE ancestor's date
(full-day outdoor >= 90% equal), excluding the target itself.  Fingerprint = linker v6 residual distance fp_dist(t, a, h) at h = 0, 6,
12, 23 (t's own hours 0..h vs candidate's same hours).  Truth = certain links chained k steps.  Reports top-1 accuracy, chance level,
and the EC 0 h residual gain if the identified ancestor's 23 h EC were used with the measured k-day coefficient (truth-chain oracle)."""
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
TLc=TL if "cert" not in TL.columns else TL
T0=pd.read_pickle(S3+r"\truth.pkl"); T0=T0[T0.cert]; prev={int(b):int(a) for a,b in zip(T0.a1,T0.b)}
rid=lambda i:f"{FARM[i]}_{NUM[i]:03d}"
res=[]
for t in range(N):
    if not TRAIN[t]: continue
    a=t
    for k in range(1,6):
        a=prev.get(a)
        if a is None: break
        cands=[j for j in np.where(comp==comp[a])[0] if TRAIN[j] and j!=t]
        if a not in cands or len(cands)<2: continue
        for h in (0,6,12,23):
            sc=np.array([F6.fp_dist(t,j,h) for j in cands]); sc=np.nan_to_num(sc,nan=99)
            res.append(dict(t=rid(t),k=k,h=h,p2=NUM[t]>=179,ncand=len(cands),ok=cands[int(np.argmin(sc))]==a))
R=pd.DataFrame(res)
print(R.groupby(["k","h"]).agg(n=("ok","size"),acc=("ok","mean"),chance=("ncand",lambda s:(1/s).mean())).round(2).unstack("h").to_string())
print("\npass-2 targets:"); print(R[R.p2].groupby(["k","h"]).agg(n=("ok","size"),acc=("ok","mean")).round(2).unstack("h").to_string())

# --- prize sizing (added): EC 0 h / day residual vs k-day ancestor 23 h EC (truth chains), coefficient by k, pass split ---
O=pd.read_csv(os.path.join(env.ROOT,"연구실","클로드","results","ec_rv1_preds_v1.csv")); O=O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
Q=O.groupby(["rec","hour"]).sg.mean().unstack(); Yc=O.groupby(["rec","hour"]).sub_ec.first().unstack()
rr=[]
for t in range(N):
    if not TRAIN[t] or rid(t) not in Q.index: continue
    a=t
    for k in range(1,6):
        a=prev.get(a)
        if a is None or rid(a) not in Yc.index: break
        T_=rid(t); A_=rid(a); rr.append(dict(k=k,p2=NUM[t]>=179,g=Yc.at[A_,23]-Q.at[T_,0],r0=Yc.at[T_,0]-Q.at[T_,0],rd=(Yc.loc[T_]-Q.loc[T_]).mean(),
                                      e=(Yc.loc[T_]-Q.loc[T_]).values))
RR=pd.DataFrame(rr); rm_=lambda v: float(np.sqrt(np.mean(np.square(v))))
print("\nprize (truth chains): k, pass, n, coef(0h), 0h RMSE model -> with anchor, day RMSE model -> with anchor (coef on day mean)")
for (k,p2),G in RR.groupby(["k","p2"]):
    if len(G)<4: continue
    b0=(G.g*G.r0).sum()/(G.g**2).sum(); bd=(G.g*G.rd).sum()/(G.g**2).sum(); E_=np.stack(G.e.values)
    print(f"  k={k} p2={p2} n {len(G):3d} | coef0 {b0:.2f} | 0h {rm_(G.r0):.3f}->{rm_(G.r0-b0*G.g):.3f} | day {rm_(E_):.3f}->{rm_(E_-bd*G.g.values[:,None]):.3f} (coef {bd:.2f})")
