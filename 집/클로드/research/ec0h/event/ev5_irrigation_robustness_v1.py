# -*- coding: utf-8 -*-
"""EV5: robustness of EV4 (prev-day input-only irrigation score -> next-day 0 h EC residual).
 (a) control NEXT record's own q0 and its own same-day scores; (b) detector folds = chunk blocks (record number // 10 per farm,
 so prev/next records sit in the same fold -> next record's labels never train the detector that scores prev); (c) prev chosen by
 truth cert links / linker v7m (p>=.8, same ID) / literal previous number."""
import sys, io; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
src=io.open(r"ec0h/event/ev3_irrigation_detector_v1.py",encoding="utf-8").read(); src=src[:src.index("for kind,hours,sel in")]
g={"__name__":"x"}; exec(compile(src,"ev3","exec"),g); D,F=g["D"],g["F"]
D["blk"]=D.rec.str[:3]+(D.rec.str[4:7].astype(int)//10).astype(str); ub=D.blk.unique(); rng=np.random.default_rng(7)
bf=dict(zip(ub,rng.permutation(len(ub))%5)); D["fold"]=D.blk.map(bf)
S=pd.DataFrame(index=D.rec.unique())
for kind,hours,sel in (("drop",range(8,14),D.dEC<=-.05),("rise",range(13,19),D.dEC>=.05)):
    Z=D[D.h.isin(hours)].copy(); Z["y"]=sel.reindex(Z.index).fillna(False).astype(int); pr=np.zeros(len(Z))
    for k in range(5):
        te=(Z.fold==k).values; m=ExtraTreesClassifier(400,min_samples_leaf=10,max_features=.4,random_state=1,n_jobs=2).fit(Z.loc[~te,F+["h"]].fillna(-99),Z.y[~te]); pr[te]=m.predict_proba(Z.loc[te,F+["h"]].fillna(-99))[:,1]
    Z["pr"]=pr; S[kind]=Z.groupby("rec").pr.sum()
O=pd.read_csv(os.path.join(env.ROOT,"연구실","클로드","results","ec_rv1_preds_v1.csv")); O=O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
A=O.groupby(["rec","hour"]).agg(q=("sg","mean"),y=("sub_ec","first")).reset_index(); A0=A[A.hour==0].set_index("rec")
S3=os.path.join(env.LOCAL,"linker_v3"); B=pickle.load(open(os.path.join(S3,"base.pkl"),"rb")); r=B["recs"].reset_index(drop=True)
TL=pd.read_pickle(os.path.join(S3,"truth.pkl")); TL=TL[TL.cert]; rid=lambda i:f"{r.farm[i]}_{r.day[i]:03d}"
L=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","cv_all400_choice_v7m.csv")).set_index("rec")
maps={"truth":{rid(b):rid(a) for a,b in zip(TL.a1,TL.b)},
      "linker v7m p>=.8 same ID":{t:L.at[t,"prev"] for t in L.index if L.at[t,"p"]>=.8 and not L.at[t,"xf"]},
      "previous number":{t:f"{t[:3]}_{int(t[4:7])-1:03d}" for t in A0.index}}
rng2=np.random.default_rng(3)
for nm,mp in maps.items():
    rows=[(t,p) for t,p in mp.items() if t in A0.index and p in S.index and t in S.index]
    J=pd.DataFrame(rows,columns=["t","p"]); J["r0"]=[A0.at[t,"y"]-A0.at[t,"q"] for t in J.t]; J["q0"]=[A0.at[t,"q"] for t in J.t]
    J["pd"]=S.loc[J.p,"drop"].values; J["pr_"]=S.loc[J.p,"rise"].values; J["td"]=S.loc[J.t,"drop"].values; J["tr"]=S.loc[J.t,"rise"].values
    J["p2"]=J.t.str[4:7].astype(int)>=179
    for part,M in (("all",J.t==J.t),("p1",~J.p2),("p2",J.p2)):
        G=J[M]; Z=np.c_[np.ones(len(G)),G.q0,G.td,G.tr]
        res=lambda v: v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]
        rx,ry=res(G.pd.values),res(G.r0.values); c=np.corrcoef(rx,ry)[0,1]
        pp=np.mean([abs(np.corrcoef(rng2.permutation(rx),ry)[0,1])>=abs(c) for _ in range(2000)])
        print(f"{nm:26s} {part}: n {len(G):3d} | partial corr(prev-day DROP score, next 0h residual | next q0 + next own scores) {c:+.3f} (p {pp:.3f})")
