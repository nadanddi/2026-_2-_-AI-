import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import pandas as pd, numpy as np
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(os.path.join(S3,"truth.pkl"))
print("a1 train", TRAIN[TL.a1.values.astype(int)].mean(), "cert & a1 train", (TL.cert.astype(bool) & TRAIN[TL.a1.values.astype(int)]).sum())
c=TL[TL.cert.astype(bool)]; print("dup a1 among cert", c.a1.duplicated().sum())
O = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv"); O=O[(O.tag=="CUR")]
O["rec"]=O.row_id.str[:7]
for v,G in O.groupby("validator"): print(v, G.rec.nunique(), G.seed.unique(), G.fold.nunique())
D=O[O.validator=="DIAG10"]; F=D.groupby("rec").fold.first()
# same-bay neighbor fold sharing
pre={rid(int(b)):rid(int(a)) for a,b,cc in zip(TL.a1,TL.b,TL.cert) if cc and TRAIN[int(a)]}
same=[F[r]==F[p] for r,p in pre.items() if r in F and p in F]; print("cert-linked neighbours in same DIAG10 fold:", np.mean(same), len(same))
print(D.groupby("fold").rec.nunique().to_dict())
print(F.groupby(F).apply(lambda s: sorted(s.index)[:12]).head(3).to_dict())
# 13th vs 9th
S13=pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); 
import glob; f9=glob.glob(r"C:\work\farmai\제출\09회차*\*.csv"); print(f9)
S9=pd.read_csv(f9[0]); 
m=S13.merge(S9,on="row_id",suffixes=("13","9")); m["rec"]=m.row_id.str[:7]
for r in ["F13_221","F13_223","F13_225","F13_237","F47_221","F47_223","F47_225","F47_227"]:
    g=m[m.rec==r]; print(r, round(g.sub_ec13.mean(),3), round(g.sub_ec9.mean(),3), round((g.sub_ec13-g.sub_ec9).abs().max(),4))
