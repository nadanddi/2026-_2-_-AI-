# SC2b (사후, 판정 변경 아님): SC2 EV 이득에서 F13_243 / F13_243+F13_241 을 뺐을 때 남는 변화와 덩어리 P
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
O = pd.read_csv(r"sc1\sc2_cv_oof.csv"); FV = pd.read_csv(r"ev_validator\ev_folds_v1.csv"); ch = dict(zip(FV[FV.role == "val"].rec, FV[FV.role == "val"].chunk))
O["rec"] = O.row_id.str[:7]; r = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(7)
for v in ("EV", "DIAG10"):
    for drop in ([], ["F13_243"], ["F13_243", "F13_241"]):
        out = []
        for sd, G in O[(O.validator == v) & (O.day >= 179) & ~O.rec.isin(drop)].groupby("seed"):
            B = G[G.arm == "BASE"].set_index("row_id"); S = G[G.arm == "P2DATE"].set_index("row_id").reindex(B.index)
            per = ((S.p - S.y) ** 2 - (B.p - B.y) ** 2).groupby(B.rec).sum(); grp = per.groupby([ch.get(x, x) for x in per.index]).sum() if v == "EV" else per
            ii = rng.integers(0, len(grp), (4000, len(grp))); pw = (grp.values[ii].sum(1) >= 0).mean()
            out.append(f"s{sd} {100*(r(S.p-S.y)/r(B.p-B.y)-1):+.1f}% P {pw:.3f} improved {int((per<0).sum())}/{int((per!=0).sum())}")
        print(f"{v:6s} drop {drop}: " + " | ".join(out))
