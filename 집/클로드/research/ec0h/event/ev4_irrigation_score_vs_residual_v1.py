# -*- coding: utf-8 -*-
"""EV4: daily irrigation scores from the input-only event detector (EV3, CV out-of-fold probabilities) vs EC model residuals.
Scores per record: DROP (morning 8-13) max/mean prob, RISE (13-18) max/mean prob, expected event counts.  Targets: same-day mean
residual (y - q, q = DIAG10 CUR sg seed mean) and NEXT same-house day 0 h residual (truth cert links).  Partial correlation after
controlling q day mean (model level).  Pass-1 / pass-2 split; permutation p."""
import sys, io, contextlib; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
src=io.open(r"ec0h/event/ev3_irrigation_detector_v1.py",encoding="utf-8").read(); src=src[:src.index("for kind,hours,sel in")]
g={"__name__":"x"}; exec(compile(src,"ev3","exec"),g); D,F=g["D"],g["F"]
S=pd.DataFrame(index=D.rec.unique())
for kind,hours,sel in (("drop",range(8,14),D.dEC<=-.05),("rise",range(13,19),D.dEC>=.05)):
    Z=D[D.h.isin(hours)].copy(); Z["y"]=sel.reindex(Z.index).fillna(False).astype(int); pr=np.zeros(len(Z))
    for k in range(5):
        te=(Z.fold==k).values; m=ExtraTreesClassifier(400,min_samples_leaf=10,max_features=.4,random_state=1,n_jobs=2).fit(Z.loc[~te,F+["h"]].fillna(-99),Z.y[~te]); pr[te]=m.predict_proba(Z.loc[te,F+["h"]].fillna(-99))[:,1]
    Z["pr"]=pr; S[kind+"_sum"]=Z.groupby("rec").pr.sum(); S[kind+"_max"]=Z.groupby("rec").pr.max()
O=pd.read_csv(os.path.join(env.ROOT,"연구실","클로드","results","ec_rv1_preds_v1.csv")); O=O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
A=O.groupby(["rec","hour"]).agg(q=("sg","mean"),y=("sub_ec","first")).reset_index()
Rd=A.assign(e=A.y-A.q).groupby("rec").agg(rd=("e","mean"),qm=("q","mean")); R0=A[A.hour==0].set_index("rec").assign(r0=lambda d:d.y-d.q).r0
S3=os.path.join(env.LOCAL,"linker_v3"); B=pickle.load(open(os.path.join(S3,"base.pkl"),"rb")); r=B["recs"].reset_index(drop=True)
TL=pd.read_pickle(os.path.join(S3,"truth.pkl")); TL=TL[TL.cert]; rid=lambda i:f"{r.farm[i]}_{r.day[i]:03d}"
nxt={rid(a):rid(b) for a,b in zip(TL.a1,TL.b)}
J=S.join(Rd,how="inner"); J["r0_next"]=[R0.get(nxt.get(x),np.nan) for x in J.index]; J["p2"]=J.index.str[4:7].astype(int)>=179
rng=np.random.default_rng(1)
def pcor(x,y,z):
    m=np.isfinite(x)&np.isfinite(y)&np.isfinite(z); x,y,z=x[m],y[m],z[m]
    rx=x-np.polyval(np.polyfit(z,x,1),z); ry=y-np.polyval(np.polyfit(z,y,1),z); c=np.corrcoef(rx,ry)[0,1]
    perm=[np.corrcoef(rng.permutation(rx),ry)[0,1] for _ in range(2000)]; return c,float(np.mean(np.abs(perm)>=abs(c))),m.sum()
for tgt in ("rd","r0_next"):
    for s in ["drop_sum","drop_max","rise_sum","rise_max"]:
        out=[]
        for nm,M in (("all",J.index==J.index),("p1",~J.p2),("p2",J.p2)):
            c,p,n=pcor(J[s].values[M],J[tgt].values[M],J.qm.values[M]); out.append(f"{nm} {c:+.2f}(p{p:.2f},n{n})")
        print(f"{tgt:8s} {s:9s}: "+" | ".join(out))
