# -*- coding: utf-8 -*-
"""verify_ops v2: v1 의 within-episode 순열 버그(groupby.transform 이 index 재정렬로 섞이지 않음) 수정 + 상위 3기록 제거 후 within/Spearman."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
from scipy import stats
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rng = np.random.default_rng(8)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "ops", "vars_v1.csv")).set_index("rec")
R = R.join(F, rsuffix="_f")
S = R[(R.hi10 == 1) & (R.has_sg == 1)].copy(); y = S.resid_mean.values; g = S.ep_id.values.astype(int)
def r_(x, yy):
    m = np.isfinite(x) & np.isfinite(yy)
    return np.corrcoef(x[m], yy[m])[0, 1] if m.sum() > 4 and np.std(x[m]) > 0 and np.std(yy[m]) > 0 else np.nan
def dm(v, gg):
    out = v.astype(float).copy()
    for u in np.unique(gg): k = gg == u; out[k] = v[k] - np.nanmean(v[k])
    return out
def within_perm(x, yy, gg, B=3000):
    obs = r_(dm(x, gg), dm(yy, gg)); null = []
    for _ in range(B):
        yp = yy.copy()
        for u in np.unique(gg):
            k = np.where(gg == u)[0]; yp[k] = yy[rng.permutation(k)]
        null.append(r_(dm(x, gg), dm(yp, gg)))
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)
drop = ["F13_217", "F47_132", "F47_231"]
VV = ["chg_inp_heath", "chg_inp_co2h", "chg_inp_sealed", "chg_inp_fanh", "chg_inp_L1", "proto_sim_hi", "curtain_decouple", "sh_partial", "heat_hours", "th_trans", "fan_night", "te_clu_mag", "chg_clean_L1am", "chg_clean_L1", "chg_clean_heath"]
rows = []
for v in VV:
    x = S[v].values.astype(float); d = dict(var=v)
    d["r_within"], d["p_within"] = within_perm(x, y, g)
    k = ~S.index.isin(drop)
    d["r_drop3"] = r_(x[k], y[k]); d["rho_drop3"] = stats.spearmanr(x[k], y[k], nan_policy="omit")[0]
    d["r_within_drop3"], d["p_within_drop3"] = within_perm(x[k], y[k], g[k])
    kp = S.pass2.values == 0; d["r_pass1_n26"] = r_(x[kp], y[kp])
    d["r_drop_F47_132"] = r_(x[S.index != "F47_132"], y[S.index != "F47_132"])
    rows.append(d)
T = pd.DataFrame(rows).set_index("var"); pd.set_option("display.width", 250)
print(T.round(3).to_string())
# how many episodes contribute nonzero within variance in chg_inp_heath
for u in np.unique(g):
    k = g == u; print("ep", u, "n", k.sum(), "heath", S.chg_inp_heath.values[k].astype(int).tolist(), "resid", np.round(y[k], 2).tolist())
