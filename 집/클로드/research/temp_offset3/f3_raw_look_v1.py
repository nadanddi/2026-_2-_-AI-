# -*- coding: utf-8 -*-
"""temp_offset3 F3: 최악 날 원자료 나란히 보기(서술). 큰 날 + 같은 날짜 형제 + 진실 연결 전날."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
RES = r"C:\work\farmai\집\클로드\research"; OUTD = os.path.join(RES, "temp_offset3")
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Yd = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); X = X.merge(Yd, on="row_id")
T = pd.read_pickle(os.path.join(RES, r"ev_validator\temp\local_ckpt\ev_all.pkl")); T["q"] = T[[c for c in T.columns if c.startswith("v22_")]].mean(axis=1)
X = X.merge(T[["row_id", "q"]], on="row_id", how="left"); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[-2:].astype(int)
groups = [("F47_178", ["F47_008", "F13_009", "F13_180"]), ("F47_191", ["F47_190", "F13_056", "F13_194"]), ("F47_189", ["F47_074", "F13_049", "F13_050"]),
          ("F47_158", ["F13_155", "F13_156"]), ("F47_241", ["F47_166", "F13_164"]), ("F47_240", ["F13_159", "F13_160"])]
TT = pd.read_pickle(os.path.join(RES, r"local\linker_v3\truth.pkl"))
import pickle; R = pickle.load(open(os.path.join(RES, r"local\linker_v3\base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
ids = [f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]
pv = {ids[int(t.b)]: (ids[int(t.a1)], bool(t.cert)) for _, t in TT.iterrows()}
cols = ["sub_temp", "q", "in_temp", "out_temp", "act_heating", "act_circfan", "act_thermal", "act_vent", "out_rad", "sub_ec"]
for b, sibs in groups:
    p, cert = pv.get(b, (None, False))
    print(f"\n########## {b}  (진실 연결 전날 {p}, cert {cert})  형제 {sibs}")
    for rec in ([p] if p else []) + [b] + sibs:
        g = X[X.rec == rec].sort_values("h")
        if len(g) == 0: print("  ", rec, "없음(평가 기록?)"); continue
        print(f"  -- {rec}")
        for c in cols:
            v = g[c].values; print(f"   {c:11s} " + " ".join(f"{x:5.1f}" if np.isfinite(x) else "  nan" for x in v[::2]))
