# -*- coding: utf-8 -*-
"""EG1N (2026-10-14 집 클로드; 사용자: 'GRU 가 보통 날에서 제출 모델보다 얼마나 좋아?').  서술.  EG1·EG2 저장 GRU 예측 vs 검증기 기준 예측(sg / SG2 — 앵커·B5 전 기본 모델).
보통 날 = (a) 실제 하루 평균 < 1.0 (사후, 정답 기준), (b) 관문: 그 시각까지 기본 예측 누적평균 < 1.0 (과거 정보만, 실제 쓸 수 있는 기준).
RMSE: 기준 / GRU 단독 / 10% 섞기 / 20% 섞기, 2차 행과 전체 행."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
G1 = pd.read_csv(os.path.join(HERE, "eg1_gru_preds_v1.csv")).set_index(["val", "rec"]); GE = pd.read_csv(os.path.join(HERE, "eg2_gru_ev_preds_v1.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
frames = []
for (v, sd), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"sub_ec": "y"}); G["gru"] = [G1.at[(v, r), f"h{h}"] for r, h in zip(G.rec, G.hour)]; G["val"] = v; frames.append(G[["val", "seed", "rec", "hour", "day", "y", "sg", "gru"]])
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"SG2": "sg"}); G["gru"] = [GE.at[r, f"h{h}"] for r, h in zip(G.rec, G.hour)]; G["val"] = "EV"; frames.append(G[["val", "seed", "rec", "hour", "day", "y", "sg", "gru"]])
A = pd.concat(frames, ignore_index=True); A["cm"] = A.groupby(["val", "seed", "rec"]).sg.transform(lambda s: s.expanding().mean())
A["trueN"] = [DM[idx[r]] < 1.0 for r in A.rec]; A["gateN"] = A.cm < 1.0; A["p2"] = A.day >= 179
rows = []
for (v, sd), G in A.groupby(["val", "seed"]):
    for nm, m in (("trueN p2", G.trueN & G.p2), ("gateN p2", G.gateN & G.p2), ("trueN all", G.trueN), ("gateN all", G.gateN), ("gate-HIGH p2", ~G.gateN & G.p2)):
        g = G[m]
        if len(g) == 0: continue
        b = rm(g.sg - g.y); rows.append(dict(val=v, seed=sd, set=nm, rows=len(g), base=b, gru_alone=100 * (rm(g.gru - g.y) / b - 1), blend10=100 * (rm(.9 * g.sg + .1 * g.gru - g.y) / b - 1), blend20=100 * (rm(.8 * g.sg + .2 * g.gru - g.y) / b - 1)))
S = pd.DataFrame(rows).groupby(["set", "val"])[["rows", "base", "gru_alone", "blend10", "blend20"]].mean(); pd.set_option("display.width", 200)
print("seed-mean; base = RMSE of validator base model, others = % change vs base (neg = better)"); print(S.round(3).to_string())
