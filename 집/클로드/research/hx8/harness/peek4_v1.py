# -*- coding: utf-8 -*-
# peek4 (HX3 준비, 2026-10-13 집 클로드): 학습 전용 정보로 같은 동 사슬 만들기 점검. 서술.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); print("truth cols", TL.columns.tolist(), len(TL))
a = TL.a1.values.astype(int); b = TL.b.values.astype(int); c = TL.cert.astype(bool).values
print("a1 train %.2f  b train %.2f  cert %d  cert&both train %d" % (TRAIN[a].mean(), TRAIN[b].mean(), c.sum(), (c & TRAIN[a] & TRAIN[b]).sum()))
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); print("cal", CAL.shape, CAL.columns.tolist(), "segs", sorted(CAL.seg.unique())[:20])
SD = {r: (int(s), int(d)) for r, s, d in zip(CAL.rec, CAL.seg, CAL.day)}
# cert train-train links: farm/num relation & calendar day diff
rows = []
for x, y, cc in zip(a, b, c):
    if not (TRAIN[x] and TRAIN[y]): continue
    rx, ry = rid(x), rid(y)
    rows.append(dict(cert=cc, same_farm=FARM[x] == FARM[y], dnum=NUM[y] - NUM[x], seg_eq=SD.get(rx, (-1,))[0] == SD.get(ry, (-2,))[0],
                     dday=(SD[ry][1] - SD[rx][1]) if (rx in SD and ry in SD) else np.nan, p2=NUM[y] >= 179))
L = pd.DataFrame(rows)
for cc, G in L.groupby("cert"):
    print("cert", cc, "n", len(G), "same_farm %.2f" % G.same_farm.mean(), "dnum==2 %.2f" % ((G.dnum == 2) & G.same_farm).mean(), "seg_eq %.2f" % G.seg_eq.mean(), "dday", G.dday.value_counts().head(4).to_dict())
    for p2, H in G.groupby("p2"): print("   p2", p2, "n", len(H), "same_farm&dnum2 %.2f" % ((H.dnum == 2) & H.same_farm).mean())
# number structure among training records: same farm, NUM-2k, both training, same seg -> day diff == k ?
res = []
for r, (s, d) in SD.items():
    i = idx[r]
    for k in range(1, 11):
        r2 = f"{FARM[i]}_{NUM[i]-2*k:03d}"
        if r2 in SD:
            s2, d2 = SD[r2]; res.append(dict(k=k, seg_eq=s2 == s, dd_ok=(s2 == s) and (d - d2 == k), p2=NUM[i] >= 179))
R = pd.DataFrame(res)
print("NUM-2k structure (training pairs):"); print(R.groupby(["p2", "k"]).agg(n=("dd_ok", "size"), seg_eq=("seg_eq", "mean"), dd_ok=("dd_ok", "mean")).round(2).unstack(0).to_string())
# DIAG10 fold sharing of cert neighbours
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"), usecols=["validator", "tag", "seed", "fold", "row_id"])
O = O[O.tag == "CUR"]; O["rec"] = O.row_id.str[:7]
for v, G in O.groupby("validator"):
    F = G.groupby("rec").fold.first(); same = [F[rid(x)] == F[rid(y)] for x, y, cc in zip(a, b, c) if cc and TRAIN[x] and TRAIN[y] and rid(x) in F.index and rid(y) in F.index]
    print(v, "recs", G.rec.nunique(), "folds", G.fold.nunique(), "seeds", sorted(G.seed.unique()), "cert-neighbours same fold %.2f (n %d)" % (np.mean(same), len(same)))
