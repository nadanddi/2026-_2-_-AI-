# -*- coding: utf-8 -*-
"""ND1 (2026-10-14 집 클로드; 사용자: '기본 모델 오차가 왜 이렇게 커? 보통 날은 잘 나와야 하지 않나').  서술.  EC, 실제 보통(하루 평균 < 1.0) 기록, 검증기 기본 예측.
(1) 오차 분해: 하루 수준(기록 평균 잔차) 몫 vs 하루 안 모양 몫, 0시 오차.  (2) 상수·하루 평균 오라클과 비교.  (3) 보통 날 실제 하루 평균의 퍼짐(동·날 사이 차이).
(4) 같은 날 다른 보통 기록과의 차이(같은 날짜인데 동마다 수준이 다른가) — 학습 달력 DMT3e."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].rename(columns={"SG2": "sg"})
A = pd.concat([O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].rename(columns={"sub_ec": "y"})[["validator", "seed", "rec", "hour", "day", "y", "sg"]],
               R.assign(validator="EV")[["validator", "seed", "rec", "hour", "day", "y", "sg"]]], ignore_index=True)
A = A[[DM[idx[r]] < 1.0 for r in A.rec]].copy(); A["e"] = A.sg - A.y
rows = []
for (v, p2), G in A.groupby(["validator", A.day >= 179]):
    rec_bias = G.groupby(["seed", "rec"]).e.transform("mean"); e0 = G[G.hour == 0].e
    ym = G.groupby(["seed", "rec"]).y.transform("mean")
    rows.append(dict(val=v, pass2=p2, recs=G.rec.nunique(), y_mean=G.y.mean(), rmse=rm(G.e), day_level_share=float(np.sum(rec_bias ** 2) / np.sum(G.e ** 2)),
                     rmse_0h=rm(e0), rmse_if_daymean_known=rm(G.sg - rec_bias - G.y), const_baseline=rm(G.y - G.y.mean()), true_daymean_sd=float(G.groupby("rec").y.mean().std()),
                     shape_only=rm(G.y - ym)))
pd.set_option("display.width", 220); print(pd.DataFrame(rows).round(3).to_string(index=False))
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s, d) for r, s, d in zip(C.rec, C.seg, C.day)}
L = pd.DataFrame([dict(rec=rid(i), key=DAY[rid(i)], y=DM[i], p2=NUM[i] >= 179) for i in range(N) if TRAIN[i] and np.isfinite(DM[i]) and DM[i] < 1.0 and rid(i) in DAY])
g = L.groupby("key").y.agg(["size", "std", "min", "max"]); g = g[g["size"] >= 2]
print(f"\nnormal labelled records on the same training-calendar date (dates with >=2): {len(g)} dates | mean within-date SD of day-mean EC {g['std'].mean():.3f} | mean range {(g['max'] - g['min']).mean():.3f}")
print(f"overall SD of normal day-mean EC: {L.y.std():.3f} (pass-1 {L[~L.p2].y.std():.3f}, pass-2 {L[L.p2].y.std():.3f}); mean {L.y.mean():.3f}")
# ---- ND1b (사용자: '고EC 문제를 다 뺀다고 치면 보통 날 RMSE?') ----
B = pd.concat([O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].rename(columns={"sub_ec": "y"})[["validator", "seed", "rec", "hour", "day", "y", "sg"]],
               R.assign(validator="EV")[["validator", "seed", "rec", "hour", "day", "y", "sg"]]], ignore_index=True)
B["hi"] = [DM[idx[r]] >= 1.0 for r in B.rec]; B["e"] = B.sg - B.y; out = []
for (v, p2), G in B.groupby(["validator", B.day >= 179]):
    out.append(dict(val=v, pass2=p2, high_rec_share=float(G.drop_duplicates(["seed", "rec"]).hi.mean()), rmse_all=rm(G.e), rmse_normal_only=rm(G.e[~G.hi]),
                    rmse_all_high_perfect=rm(np.where(G.hi, 0, G.e)), high_share_of_SSE=float(np.sum(G.e[G.hi] ** 2) / np.sum(G.e ** 2))))
print("\n[ND1b] if high-EC records were removed / perfect:"); print(pd.DataFrame(out).round(3).to_string(index=False))
