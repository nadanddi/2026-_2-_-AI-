# -*- coding: utf-8 -*-
"""HX9 verify_ops v1 (2026-10-13 집 클로드, 반박 검증자). ops 가족 weak/signal 변수 반박 시도. 읽기 전용 입력, 산출은 로그만."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "ops", "vars_v1.csv")).set_index("rec")
LK = pd.read_csv(os.path.join(HX9, "ops", "link_input_v1.csv")).set_index("rec")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
# all-record hourly inputs incl eval
def load(fn):
    D = pd.read_csv(os.path.join(env.DATA, fn)); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["hour"] = D.row_id.str[8:].astype(int); return D
A = pd.concat([load("train_X.csv"), load("test_X.csv")])
P = {v: A.pivot(index="rec", columns="hour", values=v) for v in ["out_temp", "act_heating", "act_co2", "in_temp"]}
R = R.join(F, rsuffix="_f").join(LK[["inp_prev"]])
R["out_mean"] = P["out_temp"].mean(1).reindex(R.index)
R["d_out"] = [P["out_temp"].loc[r].mean() - P["out_temp"].loc[p].mean() if isinstance(p, str) else np.nan for r, p in R.inp_prev.items()]
# chain id (clean-link components) as in v6
par = {r: r for r in R.index}
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for r, p in R.prev_rec.items():
    if isinstance(p, str) and p in par: par[find(r)] = find(p)
R["chain"] = [find(r) for r in R.index]
S = R[(R.hi10 == 1) & (R.has_sg == 1)].copy(); y = S.resid_mean.values
print("n hi&oof", len(S), "ep chains", S.ep_id.nunique(), "clean chains", S.chain.nunique())
# same-date groups among S (outdoor identical >=70%)
O = P["out_temp"].loc[S.index].values
dg = -np.ones(len(S), int); k = 0
for i in range(len(S)):
    if dg[i] < 0:
        m = (np.abs(O - O[i]) < 1e-6).mean(1) >= .7; dg[m & (dg < 0)] = k; k += 1
S["dategrp"] = dg; print("unique dates among hi&oof:", k)

VV = ["chg_inp_heath", "chg_inp_co2h", "chg_inp_sealed", "chg_inp_fanh", "chg_inp_L1", "proto_sim_hi", "curtain_decouple", "sh_partial", "heat_hours", "th_trans", "fan_night", "te_clu_hirate", "te_clu_mag", "chg_clean_L1am", "chg_clean_L1", "chg_clean_heath", "d_out"]
def r_(x, yy):
    m = np.isfinite(x) & np.isfinite(yy)
    return (np.corrcoef(x[m], yy[m])[0, 1] if m.sum() > 4 and np.std(x[m]) > 0 and np.std(yy[m]) > 0 else np.nan)
def demean(v, g):
    s = pd.Series(v); return (s - s.groupby(g).transform("mean")).values
g = S.ep_id.values
rows = []
for v in VV:
    x = S[v].values.astype(float)
    d = dict(var=v, r=r_(x, y))
    # within-episode (chain-demeaned) correlation -> removes house/season level; cluster permutation within chain
    xw, yw = demean(x, g), demean(y, g); d["r_within_ep"] = r_(xw, yw)
    null = []
    for _ in range(2000):
        yp = pd.Series(y).groupby(g).transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9)))).values
        null.append(r_(xw, demean(yp, g)))
    d["p_within_perm"] = (np.sum(np.abs(null) >= abs(d["r_within_ep"])) + 1) / 2001 if np.isfinite(d["r_within_ep"]) else np.nan
    # between-episode (chain means), n=8
    cm = pd.DataFrame(dict(x=x, y=y, g=g)).groupby("g").mean(); d["r_between_ep_n8"] = r_(cm.x.values, cm.y.values)
    # whole-chain permutation (shuffle y-chain blocks' means across chains: chain-label permutation of x means)
    # partial on d_out + out_mean + sg
    Z = np.column_stack([S.d_out, S.out_mean, S.sg_mean]); m = np.isfinite(x) & np.all(np.isfinite(Z), 1)
    Zm = np.column_stack([np.ones(m.sum()), Z[m]])
    rx = x[m] - Zm @ np.linalg.lstsq(Zm, x[m], rcond=None)[0]; ry = y[m] - Zm @ np.linalg.lstsq(Zm, y[m], rcond=None)[0]
    d["pr_dout_out_sg"] = r_(rx, ry) if v != "d_out" else np.nan
    # drop the 3 largest-|x| records
    o = np.argsort(-np.abs(np.nan_to_num(x - np.nanmedian(x))))[:3]; k3 = np.ones(len(x), bool); k3[o] = False
    d["r_drop_top3x"] = r_(x[k3], y[k3]); d["top3x"] = ",".join(S.index[o])
    # drop 3 largest residuals
    o2 = np.argsort(-y)[:3]; k4 = np.ones(len(x), bool); k4[o2] = False; d["r_drop_top3y"] = r_(x[k4], y[k4])
    # date-cluster bootstrap
    ud = np.unique(dg); bs = []
    for _ in range(2000):
        pk = rng.choice(ud, len(ud)); ii = np.concatenate([np.where(dg == u)[0] for u in pk]); bs.append(r_(x[ii], y[ii]))
    d["ci_date_lo"], d["ci_date_hi"] = np.nanpercentile(bs, [2.5, 97.5])
    # all 360 OOF records: residual corr, and normal-only
    SA = R[R.has_sg == 1]; xa = SA[v].values.astype(float); ya = SA.resid_mean.values; nm = SA.hi10.values == 0
    d["r_all360"] = r_(xa, ya); d["r_normal329"] = r_(xa[nm], ya[nm])
    rows.append(d)
T = pd.DataFrame(rows).set_index("var")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).to_string())
print("top3 resid recs:", list(S.index[np.argsort(-y)[:3]]))

# ---- selection-on-outcome check: is chg_inp_heath associated with residual only because hi10 is selected on y? ----
SA = R[R.has_sg == 1].copy()
x = SA.chg_inp_heath.values; ya = SA.resid_mean.values
for thr in [0.6, 0.8, 1.0, 1.2, 1.5]:
    m = SA.ec_mean.values >= thr; print(f"y>={thr}: n {m.sum()} r {r_(x[m], ya[m]):.3f}")
for thr in [0.6, 0.8, 1.0, 1.2]:
    m = SA.sg_mean.values >= thr; print(f"sg>={thr} (usable selection): n {m.sum()} r {r_(x[m], ya[m]):.3f}")

# ---- hour-causal version: heat hours 0..h minus linker prev heat hours 0..h, vs hourly residual at h (hi records) ----
H = HO.set_index(["rec", "hour"])
for h in [5, 11, 17, 23]:
    xs, ys = [], []
    for r in S.index:
        p = R.at[r, "inp_prev"]
        xs.append((P["act_heating"].loc[r].values[:h + 1] > 0).sum() - (P["act_heating"].loc[p].values[:h + 1] > 0).sum())
        ys.append(H.at[(r, h), "sub_ec"] - H.at[(r, h), "sg"])
    xs, ys = np.array(xs, float), np.array(ys)
    print(f"hour {h}: causal chg_heat(0..h) vs hourly resid r {r_(xs, ys):.3f} | within-ep {r_(demean(xs, g), demean(ys, g)):.3f}")

# ---- linker sanity: how often linker prev is same date as the clean predecessor's date / pass ----
agree = (S.inp_prev == S.prev_rec).mean(); print("hi&oof linker==clean prev:", round(agree, 3))
print("pass2 hi:", list(S.index[S.pass2 == 1]))
print(S[["chg_inp_heath", "chg_inp_co2h", "d_out", "resid_mean", "ep_id", "pass2", "inp_prev", "prev_rec"]].sort_values("resid_mean").round(2).to_string())
