# -*- coding: utf-8 -*-
"""HX9-OPS v3 (2026-10-13 집 클로드): v2 생존 후보 chg_inp_heath(난방 시간 − 입력 링커 전날)가 '관리 지문'인지 '날씨 변화 대리'인지.
사후 점검(결과 본 뒤). 사전 기준: 전날 대비 외기 변화(밤·낮 외기온, 일사)와 실내온 변화를 통제한 편상관이 |r| < .20 으로 떨어지면
'날씨 변화 대리'(난방은 온도조절기 반응; 일반 지식) → 관리 기제 근거 아님. 유지되면 '관리/운영 변화 단서'로 남김.
또 같은 통제를 sg 와 계절(하루 외기 평균)에도 함께.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
L = pd.read_csv(os.path.join(HERE, "link_input_v1.csv")).set_index("rec")
V = ["out_temp", "out_rad", "in_temp", "act_heating"]
def load(fn):
    D = pd.read_csv(os.path.join(env.DATA, fn)); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["hour"] = D.row_id.str[8:].astype(int); return D[["rec", "hour"] + V]
A = pd.concat([load("train_X.csv"), load("test_X.csv")])
P = {v: A.pivot(index="rec", columns="hour", values=v) for v in V}
night = lambda v, r: float(np.r_[P[v].loc[r].values[:6], P[v].loc[r].values[20:]].mean())
day = lambda v, r: float(P[v].loc[r].values[9:17].mean())
S = R[(R.hi10 == 1) & (R.has_sg == 1)].join(F[["chg_inp_heath", "chg_clean_heath", "heat_hours"]]).join(L[["inp_prev"]])
for nm, fn in [("d_out_night", lambda r, p: night("out_temp", r) - night("out_temp", p)), ("d_out_day", lambda r, p: day("out_temp", r) - day("out_temp", p)),
               ("d_rad_day", lambda r, p: day("out_rad", r) - day("out_rad", p)), ("d_in_night", lambda r, p: night("in_temp", r) - night("in_temp", p)),
               ("out_mean", lambda r, p: float(P["out_temp"].loc[r].values.mean()))]:
    S[nm] = [fn(r, p) for r, p in zip(S.index, S.inp_prev)]
y = S.resid_mean.values; x = S.chg_inp_heath.values
def pr(x, y, Z):
    Zm = np.column_stack([np.ones(len(y))] + [S[z].values for z in Z])
    rx = x - Zm @ np.linalg.lstsq(Zm, x, rcond=None)[0]; ry = y - Zm @ np.linalg.lstsq(Zm, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]
print("corr of chg_inp_heath with weather changes:", {z: round(stats.pearsonr(x, S[z].values)[0], 3) for z in ["d_out_night", "d_out_day", "d_rad_day", "d_in_night", "out_mean"]})
print("r raw %.3f" % stats.pearsonr(x, y)[0])
for Z in [["d_out_night"], ["d_out_night", "d_out_day"], ["d_out_night", "d_out_day", "d_rad_day"], ["d_out_night", "d_out_day", "d_rad_day", "d_in_night"],
          ["d_out_night", "d_out_day", "d_rad_day", "d_in_night", "out_mean", "sg_mean"]]:
    print("partial | " + "+".join(Z) + ": %.3f" % pr(x, y, Z))
k = S.index != "F47_132"
print("weather-change r with Y2 itself:", {z: round(stats.pearsonr(S[z].values, y)[0], 3) for z in ["d_out_night", "d_out_day", "d_rad_day", "d_in_night"]})
print(S[["inp_prev", "chg_inp_heath", "d_out_night", "d_out_day", "d_in_night", "resid_mean"]].sort_values("chg_inp_heath").round(2).to_string())
