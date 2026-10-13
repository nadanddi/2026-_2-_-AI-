# -*- coding: utf-8 -*-
"""LBC1 (2026-10-13 집 클로드; 사용자: '리더보드로 확인하고 싶은 거 각각 1개씩 골라봐'). 서술: 후보별 평가 예측 변화량(13회차 온도 / 12회차 EC 기준)과 점수 산수.
온도: TNA8(v15R·HP2 일치 + TN8 앵커, 13회차 공식), TX2 COLD(+.25·w(EC 예측)·추운 게이트).  EC: B5(δ −.29, 6.791), W2C(6.784).
"""
import os, sys, glob, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py")
exec(src[:src.index("PE15 = ")])  # hp2, v15_new, ST, idx, TRAIN, A7(BETA)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
P13 = pd.read_csv(os.path.join(env.ROOT, "제출", "13회차_2026-10-12(나)", "submission_22.csv")); P12 = pd.read_csv(os.path.join(env.ROOT, "제출", "12회차_2026-10-11(나)", "submission_19.csv"))
for P in (P13, P12): P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
P12 = P12.set_index("row_id").reindex(P13.row_id).reset_index()
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv"))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
allv = lambda r: TRAIN.copy()
TNA8 = {**v15_new(E15, allv, v7e, 0.0, True), **v15_new(E15, allv, v7e, .8, False)}
t0 = P13.sub_temp.values.copy(); tn = t0.copy()
for r, p in TNA8.items():
    m = (P13.rec == r).values; hh = P13.h.values[m]; q0 = t0[m][hh == 0][0]
    tn[m] = t0[m] + A7.BETA[hh] * np.clip(ST[idx[p], 23] - .36 - q0, -2, 2)
Xt = pd.read_csv(os.path.join(env.DATA, "test_X.csv")).set_index("row_id").reindex(P13.row_id)
cmEC = P12.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()).values
w = np.clip((cmEC - .8) / .4, 0, 1); g = np.nan_to_num(np.clip((10 - Xt.in_temp.values) / 2, 0, 1), nan=0.0); tx = t0 + .25 * w * g
def report(name, old, new, score):
    d = new - old; r_ = rm(d); n = int((np.abs(d) > 1e-9).sum())
    recs = sorted(set(P13.rec[np.abs(d) > 1e-9]))
    print(f"\n{name}: rows changed {n}, records {len(recs)}, RMS change {r_:.4f}, mean {d.mean():+.4f}")
    print("   records:", recs[:20])
    for rho in (-.3, -.2, -.1, 0):
        print(f"   corr(change, current error) {rho:+.2f} -> score {np.sqrt(score ** 2 + r_ ** 2 + 2 * rho * r_ * score):.4f}")
    print(f"   breakeven corr = {-r_ / (2 * score):+.3f}")
report("TEMP TNA8 (vs 13th .4883)", t0, tn, .4883)
report("TEMP TX2 COLD (vs 13th .4883)", t0, tx, .4883)
cold = (Xt.in_temp.values <= 8); print(f"   eval cold rows (in_temp<=8) {cold.sum()}; TX2 rows on cold {(np.abs(tx - t0) > 1e-9)[cold].sum()}")
both = tn.copy() + .25 * w * g; report("TEMP TNA8 + TX2 COLD", t0, both, .4883)
# EC
e0 = P12.sub_ec.values
s_ = importlib.util.spec_from_file_location("b5", os.path.join(HERE, "exc1_same_date_high_exclusion_v1.py"))
b5src = open(os.path.join(HERE, "b5_exclusion_learned_size_v1.py"), encoding="utf-8").read()
ns = {}; exec(b5src[:b5src.index("out = []")].replace('__file__ = os.path.join(HERE, "exc1_same_date_high_exclusion_v1.py")', '__file__ = os.path.join(HERE, "exc1_same_date_high_exclusion_v1.py")'), ns)
eb = e0.copy()
for r, gg in P12.groupby("rec"):
    ix = gg.index.values; m = ns["fired_rows"](r, gg.h.values, cmEC[ix], ns["TRAIN"].copy())
    if m.any(): eb[ix[m]] = np.clip(e0[ix[m]] - .290, ns["lo_"], ns["hi_"])
report("EC B5 (vs 12th .1271)", e0, eb, .1271)
ew = e0.copy(); m = (P12.rec == "F13_237").values; cm237 = cmEC[m]; ew[m] = np.clip(e0[m] + (.642 - cm237), ns["lo_"], ns["hi_"])
m2 = (P12.rec == "F13_222").values; ew[m2] = np.clip(e0[m2] + (.399 - cmEC[m2]), ns["lo_"], ns["hi_"])
report("EC W2C (vs 12th .1271)", e0, ew, .1271)
