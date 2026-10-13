# -*- coding: utf-8 -*-
"""PR1 (2026-10-14 집 클로드; 사용자: '보통 날 말고 짝이 있는 날/없는 날을 기준으로 큰 틀 — 짝 있는 날 EC 오차율은?').  서술.  EV(앵커까지 들어 있는 검증기), 시드 3 평균.
짝 정의: (A) 실제 = 진단용 진짜 전날(truth a1 & DM11 차 1)이 정답 기록이고 그 폴드에서 숨김 아님;  (B) 찾음 = 현재 EA2 앵커가 붙은 기록(anchored).
오차: RMSE(EA2 = 현재 구성, 앵커 적용), RMSE(SG2 = 앵커 전), 상대 오차 = RMSE / 실제 평균, 하루 평균 절대 오차.  1차/2차, 보통/고EC 로도 나눔."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
R["true_pair"] = [(r in CLEAN) and TRAIN[idx[CLEAN[r]]] and (CLEAN[r] not in hid[int(f)]) for r, f in zip(R.rec, R.fold)]
R["found_pair"] = R.anchored.astype(bool); R["p2"] = R.day >= 179; R["hi"] = [DM[idx[r]] >= 1.0 for r in R.rec]
rows = []
def add(name, m):
    for (sd), G in R[m].groupby(R.seed.astype(str)):
        dm = G.groupby("rec").agg(y=("y", "mean"), ea=("EA2", "mean"), sg=("SG2", "mean"))
        rows.append(dict(set=name, seed=sd, recs=G.rec.nunique(), y_mean=G.y.mean(), rmse_now=rm(G.EA2 - G.y), rmse_preanchor=rm(G.SG2 - G.y),
                         rel_now=rm(G.EA2 - G.y) / G.y.mean(), daymean_abs_now=float((dm.ea - dm.y).abs().mean())))
for nm, m in [("ALL", R.y.notna()), ("true pair", R.true_pair), ("no true pair", ~R.true_pair), ("found pair (anchored)", R.found_pair), ("not found", ~R.found_pair),
              ("true pair & found", R.true_pair & R.found_pair), ("true pair & NOT found", R.true_pair & ~R.found_pair), ("no true pair & found (wrong/other)", ~R.true_pair & R.found_pair),
              ("p2 ALL", R.p2), ("p2 true pair", R.p2 & R.true_pair), ("p2 no true pair", R.p2 & ~R.true_pair), ("p2 found", R.p2 & R.found_pair), ("p2 not found", R.p2 & ~R.found_pair),
              ("p2 true pair & high", R.p2 & R.true_pair & R.hi), ("p2 no true pair & high", R.p2 & ~R.true_pair & R.hi)]:
    if m.any(): add(nm, m)
S = pd.DataFrame(rows).groupby("set", sort=False)[["recs", "y_mean", "rmse_now", "rmse_preanchor", "rel_now", "daymean_abs_now"]].mean(); pd.set_option("display.width", 200)
print("EV, seed mean (rmse_now = current config with EA2 anchors; rel = RMSE / mean EC)"); print(S.round(3).to_string())
# ---- PR1b: 1차도 같은 표로 ----
rows = []
for nm, m in [("p1 ALL", ~R.p2), ("p1 found", ~R.p2 & R.found_pair), ("p1 not found", ~R.p2 & ~R.found_pair), ("p1 true pair", ~R.p2 & R.true_pair), ("p1 no true pair", ~R.p2 & ~R.true_pair)]:
    add(nm, m)
print("\n[PR1b] pass-1"); print(pd.DataFrame(rows).groupby("set", sort=False)[["recs", "y_mean", "rmse_now", "rmse_preanchor", "rel_now", "daymean_abs_now"]].mean().round(3).to_string())
