# -*- coding: utf-8 -*-
"""PA2 (2026-10-12 집 클로드; critic path A pre-check, corrected).  DATE from a training-label date graph: edges comp(a) -> comp(b)
from certain links between TRAINING records (target's own links excluded); predecessor date of comp(t) = most frequent source.
HOUSE: among labelled training members of that date, argmin linker-v6 fingerprint fp_dist(t, a, h).  Reports overall accuracy at
h = 0 / 6 / 12 (training targets), margin-gated accuracy/coverage, vs linker v7m; and EVALUATION coverage: evaluation records without
a confident anchor whose own date (h0 exact outdoor match to a training / same-ID earlier record) has a graph predecessor date with
labelled members — and whether that differs from the current v7 choice."""
import sys, io; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
src=io.open(r"C:\work\farmai\집\클로드\research\ec0h\pa1_date_then_house_precheck_v1.py",encoding="utf-8").read(); pre=src[:src.index("rows=[]")]
g={"__name__":"x"}; exec(compile(pre,"pa","exec"),g)
comp,prev,rid,TRAIN,NUM,F6,X,N,FARM=g["comp"],g["prev"],g["rid"],g["TRAIN"],g["NUM"],g["F6"],g["X"],g["N"],g["FARM"]
import os, numpy as np, pandas as pd
from collections import Counter
TLa=pd.read_pickle(g["S3"]+r"\truth.pkl"); TLa=TLa[TLa.cert]; links=[(int(a),int(b)) for a,b in zip(TLa.a1,TLa.b) if TRAIN[int(a)] and TRAIN[int(b)]]
lab_by_comp={}
for j in np.where(TRAIN)[0]: lab_by_comp.setdefault(comp[j],[]).append(j)
def pred_date(c,excl):
    E=Counter(comp[x] for x,y in links if comp[y]==c and x not in excl and y not in excl and comp[x]!=c)
    return E.most_common(1)[0][0] if E else None
rows=[]
for t,a in prev.items():
    if not TRAIN[t] or not TRAIN[a]: continue
    D1=pred_date(comp[t],{t})
    if D1 is None: continue
    mem=[j for j in lab_by_comp.get(D1,[]) if j!=t]
    r=dict(t=rid(t),p2=NUM[t]>=179,date_ok=D1==comp[a],ncand=len(mem))
    for h in (0,6,12):
        sc=np.nan_to_num(np.array([F6.fp_dist(t,j,h) for j in mem]),nan=99); o=np.argsort(sc)
        r[f"ok{h}"]=bool(len(mem)) and mem[o[0]]==a; r[f"m{h}"]=(sc[o[1]]-sc[o[0]]) if len(mem)>1 else 9
    rows.append(r)
R=pd.DataFrame(rows)
for p2,G in R.groupby("p2"):
    s=f"pass2={p2} n {len(G)} | date acc {G.date_ok.mean():.2f} | cands median {G.ncand.median():.0f}"
    for h in (0,6,12): s+=f" | h{h}: acc {G[f'ok{h}'].mean():.2f}, margin>=.3 cover {(G[f'm{h}']>=.3).mean():.2f} acc {G[G[f'm{h}']>=.3][f'ok{h}'].mean():.2f}"
    print(s)
# --- evaluation coverage ---
ev=[i for i in range(N) if not TRAIN[i]]
A=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","eval60_confident_1step_anchor_v7.csv")); anch=set(A.row)
P=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","eval60_predecessor_v7.csv")).set_index("row")
cov=[]
for t in ev:
    r_=rid(t)
    if r_ in anch: continue
    # own date at h0: exact hour-0 outdoor match with a training record or same-ID earlier record
    m=[j for j in range(N) if j!=t and (TRAIN[j] or (FARM[j]==FARM[t] and NUM[j]<NUM[t])) and np.all(np.abs(X[j,0]-X[t,0])<1e-6)]
    c=comp[m[0]] if m else None
    D1=pred_date(c,set()) if c is not None else None
    mem=lab_by_comp.get(D1,[]) if D1 is not None else []
    pick=None
    if mem:
        sc=np.nan_to_num(np.array([F6.fp_dist(t,j,0) for j in mem]),nan=99); o=np.argsort(sc); pick=rid(mem[o[0]]); marg=(sc[o[1]]-sc[o[0]]) if len(mem)>1 else 9
    cov.append(dict(rec=r_,date_found=c is not None,pred_date=D1 is not None,ncand=len(mem),pick=pick,margin=marg if mem else np.nan,v7=P.at[r_,"prev_h0"] if r_ in P.index else None,v7p=P.at[r_,"p_h0"] if r_ in P.index else None))
C=pd.DataFrame(cov); print("\nEVAL records without confident anchor:",len(C),"| own date found at 0h:",int(C.date_found.sum()),"| graph predecessor date with labelled members:",int((C.ncand>0).sum()),"| margin>=.3:",int((C.margin>=.3).sum()))
print(C[C.ncand>0].round(2).to_string(index=False))
