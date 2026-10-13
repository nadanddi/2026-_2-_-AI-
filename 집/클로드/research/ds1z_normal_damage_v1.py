import os, sys; R=r"C:\work\farmai\집\클로드\research"; sys.path.insert(0,R); os.chdir(R)
import env, pandas as pd, numpy as np
D = pd.read_csv("ds1y_zone_rows_v1.csv"); N = D[~D.trueH]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
base = O.assign(e=(O.sg-O.sub_ec)**2).groupby(["seed","rec"]).e.sum().groupby("rec").mean()
for p2 in (False, True):
    n = N[N.p2==p2]; b = base.reindex(n.rec).sum()
    print(f"pass2={p2}: normal recs {len(n)} | base SSE {b:.2f} | HMM dSSE {n.dsse.sum():+.2f} ({100*n.dsse.sum()/b:+.1f}% of SSE)")
    n = n.assign(cls=np.select([n.pred>=1.0, n["shift"]>.1, n["shift"]<-.02], ["predicted>=1", "raised>.1 (pred<1)", "lowered"], "tiny"))
    print(n.groupby("cls").agg(k=("rec","size"), dsse=("dsse","sum"), shift=("shift","mean"), pH=("pH23","mean"), chain_prevpred=("prev_pred","mean")).round(3).to_string())
    print("  top:", n.nlargest(5,"dsse")[["rec","pred","prev_pred","pH23","shift","dsse"]].round(3).to_dict("records"))
