# -*- coding: utf-8 -*-
"""verify_ops v3: chg_inp_co2h 분해 — 자기 CO2 시간 vs 링커 전날 CO2 시간 vs 링커 전날 EC(학습 정답, 사용 가능) / 같은 날짜 다른 동."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rng = np.random.default_rng(9)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "ops", "vars_v1.csv")).set_index("rec")
LK = pd.read_csv(os.path.join(HX9, "ops", "link_input_v1.csv")).set_index("rec")
def load(fn):
    D = pd.read_csv(os.path.join(env.DATA, fn)); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["hour"] = D.row_id.str[8:].astype(int); return D
A = pd.concat([load("train_X.csv"), load("test_X.csv")])
CO = A.pivot(index="rec", columns="hour", values="act_co2")
coh = (CO > 0).sum(1)
R = R.join(F[["chg_inp_co2h", "chg_inp_heath", "co2_hours"]]).join(LK[["inp_prev", "inp_prev_train", "inp_margin"]])
R["prev_co2h"] = [coh.get(p, np.nan) for p in R.inp_prev]
R["prev_ec"] = [R.ec_mean.get(p, np.nan) for p in R.inp_prev]
R["prev_hi"] = [R.hi10.get(p, np.nan) for p in R.inp_prev]
S = R[(R.hi10 == 1) & (R.has_sg == 1)].copy(); y = S.resid_mean.values; g = S.ep_id.values.astype(int)
def r_(a, b):
    m = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 4 and np.std(a[m]) > 0 else np.nan
print("chg_inp_co2h value counts:", S.chg_inp_co2h.value_counts().sort_index().to_dict())
print("own co2_hours counts:", S.co2_hours.value_counts().sort_index().to_dict())
for c in ["chg_inp_co2h", "co2_hours", "prev_co2h", "prev_ec", "prev_hi", "inp_margin", "sg_mean"]:
    print(f"{c:14s} r_resid {r_(S[c].values.astype(float), y):.3f}  r_chg {r_(S[c].values.astype(float), S.chg_inp_co2h.values):.3f}")
print(S[["co2_hours", "prev_co2h", "chg_inp_co2h", "prev_ec", "inp_prev", "prev_rec", "resid_mean", "ep_id", "pass2"]].sort_values("chg_inp_co2h").round(2).to_string())
# drop-one-episode on co2
for u in np.unique(g):
    k = g != u; print("drop ep", u, round(r_(S.chg_inp_co2h.values[k], y[k]), 3))
# placebo: same date-shift? random visible same-farm earlier record as prev (repeat)
