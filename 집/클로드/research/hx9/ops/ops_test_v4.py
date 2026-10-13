# -*- coding: utf-8 -*-
"""HX9-OPS v4 (2026-10-13 집 클로드): Y3(에피소드 안 하루 EC 변화 = ec_mean(b) − ec_mean(clean 전날), 27쌍·7사슬)에 대한
전 변수 사슬 부트스트랩 CI + 기록 순열 max-|r| (v1 은 Y3 상관만 냄). 상위 변수는 clean 전날 대비 날씨 변화(낮 일사·낮/밤 외기온)
통제 편상관. 사전 기준: Y3 '단서' = |r| >= .35 AND 사슬 CI 0 제외 AND 날씨 변화 통제 후 |r| >= .25. Y3 는 정답 사건이라
변수의 usable 여부와 별개로 이 분석은 서술용(diagnostic).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(5)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
M = pd.read_csv(os.path.join(HERE, "vars_v1_meta.csv")).set_index("variable")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
VARS = [v for v in M.index if v != "op_cluster"]
S = R[(R.hi10 == 1) & (R.prev_hi10 == 1)].join(F[VARS])
S["Y3"] = S.ec_mean - S.prev_ec_mean
P = {v: HO.pivot(index="rec", columns="hour", values=v) for v in ["out_temp", "out_rad"]}
day = lambda v, r: float(P[v].loc[r].values[9:17].mean()); night = lambda v, r: float(np.r_[P[v].loc[r].values[:6], P[v].loc[r].values[20:]].mean())
S["d_rad_day"] = [day("out_rad", r) - day("out_rad", p) for r, p in zip(S.index, S.prev_rec)]
S["d_out_day"] = [day("out_temp", r) - day("out_temp", p) for r, p in zip(S.index, S.prev_rec)]
S["d_out_night"] = [night("out_temp", r) - night("out_temp", p) for r, p in zip(S.index, S.prev_rec)]
y = S.Y3.values; g = S.ep_id.values.astype(int)
print("Y3 n", len(S), "chains", len(np.unique(g)), "| weather-change r with Y3:", {z: round(stats.pearsonr(S[z], y)[0], 3) for z in ["d_rad_day", "d_out_day", "d_out_night"]})
X = S[VARS].values.astype(float)
def rv(yv):
    o = []
    for j in range(X.shape[1]):
        x = X[:, j]; m = np.isfinite(x)
        o.append(np.corrcoef(x[m], yv[m])[0, 1] if m.sum() > 4 and np.std(x[m]) > 1e-12 else np.nan)
    return np.array(o)
obs = rv(y); valid = np.isfinite(obs)
mx = np.array([np.nanmax(np.abs(rv(rng.permutation(y)))) for _ in range(3000)])
rows = []
for j, v in enumerate(VARS):
    if not valid[j]: continue
    x = X[:, j]; ug = np.unique(g); bs = []
    for _ in range(2000):
        ii = np.concatenate([np.where(g == u)[0] for u in rng.choice(ug, len(ug))]); xx, yy = x[ii], y[ii]; m = np.isfinite(xx)
        if m.sum() > 4 and np.std(xx[m]) > 1e-12 and np.std(yy[m]) > 1e-12: bs.append(np.corrcoef(xx[m], yy[m])[0, 1])
    Zm = np.column_stack([np.ones(len(y)), S.d_rad_day, S.d_out_day, S.d_out_night]); m = np.isfinite(x)
    rx = x[m] - Zm[m] @ np.linalg.lstsq(Zm[m], x[m], rcond=None)[0]; ry = y[m] - Zm[m] @ np.linalg.lstsq(Zm[m], y[m], rcond=None)[0]
    prw = stats.pearsonr(rx, ry)[0] if np.std(rx) > 1e-12 else np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(variable=v, usable=M.at[v, "usable"], r_Y3=obs[j], ci_lo=lo, ci_hi=hi, pr_weather=prw,
                     p_fwer=(np.sum(mx >= abs(obs[j])) + 1) / 3001,
                     clue=bool(abs(obs[j]) >= .35 and (lo > 0 or hi < 0) and abs(prw) >= .25)))
T = pd.DataFrame(rows).set_index("variable").sort_values("r_Y3", key=np.abs, ascending=False)
pd.set_option("display.width", 200)
print("valid tests", valid.sum()); print(T.head(12).round(3).to_string())
T.to_csv(os.path.join(HERE, "test_v4_Y3.csv"), encoding="utf-8")
