# -*- coding: utf-8 -*-
"""NA2 (2026-10-12 집 클로드; NA1 엄격판).  사전 고정 — 실행 전 커밋.  NA1 선택(na1_picks_v1.csv) 중 '소거 뒤 다음날 후보가 1개'(ncand == 1)인 행만 사용.
β_h(시각 블록 4개)는 다른 폴드 1차 채택 행 최소제곱, [0,1] 자르기.  pred_h = EA2 + β (A − EA2).
판정: 대상 기록 3시드 모두 개선 AND 덩어리 P(worse) < .025 AND 최대 기록 몫 ≤ .5.  선택 정확도(진실 있는 경우), 1차/2차 따로.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research"); here = os.path.dirname(os.path.abspath(__file__))
Pk = pd.read_csv(os.path.join(here, "na1_picks_v1.csv")); Pk = Pk[Pk.ncand == 1]
K = Pk[Pk.ok >= 0]; print("rows", len(Pk), "records", Pk.t.nunique(), "| accuracy (truth known) rows %.3f records %.3f" % (K.ok.mean(), K.groupby("t").ok.mean().mean()))
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Y = R.groupby(["rec", "hour"]).y.first().unstack(); out, base = {}, {}
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
    for k in Pk.fold.unique():
        oth = Pk[(Pk.fold != k) & (Pk.t.str[4:7].astype(int) < 179)]; beta = {}
        for b in range(4):
            o = oth[oth.h // 6 == b]; x = np.array([r.A - P.at[r.t, r.h] for r in o.itertuples()]); y = np.array([Y.at[r.t, r.h] - P.at[r.t, r.h] for r in o.itertuples()])
            beta[b] = float(np.clip((x * y).sum() / (x * x).sum(), 0, 1)) if len(x) > 5 else 0.0
        for r in Pk[Pk.fold == k].itertuples(): new.at[r.t, r.h] = P.at[r.t, r.h] + beta[r.h // 6] * (r.A - P.at[r.t, r.h])
    out[sd] = new; print("seed", sd, "betas (last fold):", {b: round(v, 2) for b, v in beta.items()})
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261241)
for part, cond in (("all", lambda s: True), ("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = sorted(r for r in Pk.t.unique() if cond(r))
    if not recs: print(part, "none"); continue
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
    cl = per.groupby(V.chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    if part == "all": print("VERDICT NA2:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
