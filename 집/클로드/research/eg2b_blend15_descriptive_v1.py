# -*- coding: utf-8 -*-
"""EG2B (2026-10-14 집 클로드; 사용자: '15% 는 왜 안 함').  서술(사후 — EV 는 EG2 에서 이미 봄, 판정 아님).  저장된 GRU 예측으로 비율 10~30% 곡선."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
GE = pd.read_csv(os.path.join(HERE, "eg2_gru_ev_preds_v1.csv")).set_index("rec"); G1 = pd.read_csv(os.path.join(HERE, "eg1_gru_preds_v1.csv")).set_index(["val", "rec"])
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
W = (.1, .15, .2, .3); rows = []
def add(v, sd, G, base, gr, rng):
    y = G.y.values; p2 = G.day.values >= 179; bk = (G.farm + "_" + (G.day // 5).astype(str)).values
    for w in W:
        new = (1 - w) * base + w * gr; d = (new - y) ** 2 - (base - y) ** 2
        s = pd.Series(d[p2]).groupby(bk[p2]).mean().values
        rows.append(dict(val=v, seed=sd, w=w, p2_pct=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1), all_pct=100 * (rm(new - y) / rm(base - y) - 1), blocks_better=float((s < 0).mean())))
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); gr = np.array([GE.at[r, f"h{h}"] for r, h in zip(G.rec, G.hour)]); add("EV", sd, G, G.SG2.values, gr, None)
for (v, sd), G in O[O.validator.isin(["P2LOO", "EL1", "DIAG10"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"sub_ec": "y"}); gr = np.array([G1.at[(v, r), f"h{h}"] for r, h in zip(G.rec, G.hour)]); add(v, str(sd), G, G.sg.values, gr, None)
S = pd.DataFrame(rows); pd.set_option("display.width", 200)
print("pass-2 rows % change (seed mean):"); print(S.groupby(["val", "w"]).p2_pct.mean().unstack().round(3).to_string())
print("\nall rows % change (seed mean):"); print(S.groupby(["val", "w"]).all_pct.mean().unstack().round(3).to_string())
print("\nshare of pass-2 farm x 5-day blocks improved (seed mean):"); print(S.groupby(["val", "w"]).blocks_better.mean().unstack().round(3).to_string())
