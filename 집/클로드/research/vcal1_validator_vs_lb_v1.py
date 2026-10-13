# -*- coding: utf-8 -*-
"""VCAL1 (2026-10-13 집 클로드; 사용자: '실험 오류 재조사'). 서술: 검증기가 리더보드 변화를 얼마나 맞히나 — EC 12→13회차(v7 EA2 앵커 추가)를 검증기마다 재현.
LB: 12회차 .1271 → 13회차 .1279 (+0.63%).  검증기: rv1 DIAG10·P2LOO·EL1 × 3시드(V7 vs NONE), EV × 3시드(EA2 vs SG2).  2차 행·전체 행 변화 %.
또 '앵커 받은 기록'의 기준 오차(NONE 하루 수준 RMSE)를 검증기와 평가(12회차 예측 vs 전날 정답 차, 6.669 류)로 비교."""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.p and L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5} if False else {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
out = []; anc = []
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    new = sg.copy()
    for r, g in Gv.groupby("rec"):
        if r in V7 and V7[r] not in hidden[fold_of[r]]:
            ix = g.index.values; new[ix] = np.clip(sg[ix] + (EC[idx[V7[r]], 23] - cm[ix]), lo_, hi_)
            if NUM[idx[r]] >= 179: anc.append(dict(val=v, seed=sd, rec=r, base_err=abs(cm[ix][-1] - y[ix].mean()), gap=abs(EC[idx[V7[r]], 23] - cm[ix][-1])))
    out.append(dict(val=v, seed=sd, p2=100 * (rm((new - y)[p2]) / rm((sg - y)[p2]) - 1), all=100 * (rm(new - y) / rm(sg - y) - 1)))
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    p2 = Gv.day.values >= 179; y = Gv.y.values
    out.append(dict(val="EV", seed=sd, p2=100 * (rm((Gv.EA2.values - y)[p2]) / rm((Gv.SG2.values - y)[p2]) - 1), all=100 * (rm(Gv.EA2.values - y) / rm(Gv.SG2.values - y) - 1)))
    g2 = Gv[p2 & Gv.anchored.astype(bool)].groupby("rec").apply(lambda g: pd.Series(dict(base_err=abs(g.SG2.mean() - g.y.mean()), gap=abs(g.EA2.mean() - g.SG2.mean()))))
    for r, rr in g2.iterrows(): anc.append(dict(val="EV", seed=sd, rec=r, base_err=rr.base_err, gap=rr.gap))
S = pd.DataFrame(out); A = pd.DataFrame(anc); pd.set_option("display.width", 200)
print("local predicted change of adding v7 EC anchors (LB actual: +0.63% all rows, eval is all pass-2):")
print(S.groupby("val")[["p2", "all"]].mean().round(2).to_string())
print("\nanchored pass-2 records: model day-level error before anchoring and anchor-vs-model gap (record means):")
print(A.groupby("val")[["base_err", "gap"]].agg(["mean", "median"]).round(3).to_string())
# eval: anchored records' gap between 12th prediction and the anchor value
P12 = pd.read_csv(os.path.join(env.ROOT, "제출", "12회차_2026-10-11(나)", "submission_19.csv")); P12["rec"] = P12.row_id.str[:7]
P13 = pd.read_csv(os.path.join(env.ROOT, "제출", "13회차_2026-10-12(나)", "submission_22.csv")); P13["rec"] = P13.row_id.str[:7]
g = (P13.groupby("rec").sub_ec.mean() - P12.groupby("rec").sub_ec.mean()).abs(); g = g[g > 1e-6]
print(f"\neval: records changed by v7 anchors {len(g)}, |change| mean {g.mean():.3f} median {g.median():.3f}")
