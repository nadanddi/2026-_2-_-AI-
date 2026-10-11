# -*- coding: utf-8 -*-
"""temp_offset3 F2: H1 편상관, H7 계단, H8 지속률, U1a/U1b 비인과 상한 (계획 00·01·02)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd, pickle, importlib.util, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata, binomtest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
RES = r"C:\work\farmai\집\클로드\research"; OUTD = os.path.join(RES, "temp_offset3")
rng = np.random.default_rng(20261213)
R = pickle.load(open(os.path.join(RES, r"local\linker_v3\base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
ids = np.array([f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]); N = len(r); IDX = {k: i for i, k in enumerate(ids)}
FARM = r.farm.values; NUM = r.day.values
VARS = ['out_temp', 'out_hum', 'out_rad', 'out_wspd', 'in_temp', 'in_hum', 'in_co2', 'act_vent', 'act_shade', 'act_thermal',
        'act_heating', 'act_circfan', 'act_co2', 'act_fog']
Xd = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Xd = Xd[Xd.row_id.str[:3].isin(["F13", "F47"])]
Yd = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yd = Yd[Yd.row_id.str[:3].isin(["F13", "F47"])]
def tens(df, col):
    a = np.full((N, 24), np.nan); rid = df.row_id.str[:7].values; h = df.row_id.str[-2:].astype(int).values
    ii = np.array([IDX.get(x, -1) for x in rid]); m = ii >= 0; a[ii[m], h[m]] = df[col].values[m]; return a
I = {v: tens(Xd, v) for v in VARS}; Y = tens(Yd, "sub_temp")
def qmat(df, col):
    Q = np.full((N, 24), np.nan)
    for rec, g in df.groupby(df.row_id.str[:7]): Q[IDX[rec], g.hour.values] = g[col].values
    return Q
T = pd.read_pickle(os.path.join(RES, r"ev_validator\temp\local_ckpt\ev_all.pkl"))
T["q"] = T[[c for c in T.columns if c.startswith("v22_")]].mean(axis=1); QE = qmat(T, "q")
sys.argv = ["x"]; s_ = importlib.util.spec_from_file_location("ta2", os.path.join(RES, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A2)
FR = A2.A1.frames(); M, cells = FR["DIAG10"]; M = M.copy(); M["q"] = np.mean(np.vstack(cells), 0); QD = qmat(M, "q")
P2 = NUM >= 179
# truth prev (diagnostic) and v7 prev (causal-legal identification)
TT = pd.read_pickle(os.path.join(RES, r"local\linker_v3\truth.pkl")); prevT = np.full(N, -1)
for _, t in TT.iterrows():
    if t.cert and t.same_farm: prevT[int(t.b)] = int(t.a1)
C7 = pd.read_csv(os.path.join(RES, r"local\linker_v7\cv_all400_choice_v7m.csv")); prev7 = np.full(N, -1)
for _, c in C7.iterrows():
    if c.p >= .8 and c.prev[:3] == c.rec[:3] and c.prev in IDX: prev7[IDX[c.rec]] = IDX[c.prev]
print("v7 prev p>=.8 same ID:", (prev7 >= 0).sum(), "| agree with truth where both:", np.mean(prev7[(prev7 >= 0) & (prevT >= 0)] == prevT[(prev7 >= 0) & (prevT >= 0)]).round(3))
def mh(A, h0, h1): return np.nanmean(A[:, h0:h1 + 1], 1)
def summ(i_idx):
    out = {}
    for v in VARS:
        A = I[v]
        for nm, (a, b) in {"d": (0, 23), "n": (0, 5), "m": (9, 16), "e": (18, 23)}.items(): out[f"{v}_{nm}"] = mh(A, a, b)
        out[f"{v}_mn"] = np.nanmin(A, 1); out[f"{v}_mx"] = np.nanmax(A, 1)
    out["tin_23m0"] = I["in_temp"][:, 23] - I["in_temp"][:, 0]
    return pd.DataFrame(out)
S = summ(None)
def run_target(tag, Q):
    E = Y - Q; has = np.isfinite(E).all(1); o = np.nanmean(E, 1); e0 = E[:, 0]
    base = S.copy(); base["q_d"] = np.nanmean(Q, 1); base["qg0"] = Q[:, 0] - I["in_temp"][:, 0]; base["qg23"] = Q[:, 23] - I["in_temp"][:, 23]
    def prevblock(prev, pre, cols_from):
        B = pd.DataFrame(index=range(N))
        for c in cols_from.columns:
            v = cols_from[c].values; B[pre + c] = np.where(prev >= 0, v[np.maximum(prev, 0)], np.nan)
        B[pre + "has"] = (prev >= 0).astype(float); return B
    pv_cols = base[[c for c in base.columns if c.endswith("_d") or c.endswith("_e") or c in ("qg23",)]]
    U1a = pd.concat([base, prevblock(prev7, "p7_", pv_cols)], axis=1)
    lab = pd.DataFrame({"oa": o, "ra23": E[:, 23], "g23": Y[:, 23] - I["in_temp"][:, 23]})
    U1b = pd.concat([U1a, prevblock(prevT, "pT_", lab)], axis=1)
    idx = np.where(has)[0]
    # groups: per farm consecutive-number blocks, 10 blocks
    grp = np.zeros(N, int)
    for f in ("F13", "F47"):
        ii = idx[FARM[idx] == f]; ii = ii[np.argsort(NUM[ii])]
        for k, ch in enumerate(np.array_split(ii, 10)): grp[ch] = k
    res = {}
    for fname, Fm in (("U1a", U1a), ("U1b", U1b)):
        Xf = Fm.iloc[idx].copy(); Xf = Xf.fillna(Xf.median()).fillna(0).values; y = o[idx]; g = grp[idx]
        for mname in ("ridge", "lgb"):
            pr = np.zeros(len(y))
            for k in range(10):
                tr, te = g != k, g == k
                if mname == "ridge":
                    sc = StandardScaler().fit(Xf[tr]); pr[te] = Ridge(10.0).fit(sc.transform(Xf[tr]), y[tr]).predict(sc.transform(Xf[te]))
                else:
                    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=7, learning_rate=.03, min_child_samples=10, subsample=.8, subsample_freq=1,
                                          colsample_bytree=.8, random_state=7, verbose=-1).fit(Xf[tr], y[tr]); pr[te] = m.predict(Xf[te])
            res[(fname, mname)] = pr
            line = []
            for nm, mm in (("p1", ~P2[idx]), ("p2", P2[idx]), ("p2 F47", P2[idx] & (FARM[idx] == "F47")), ("p1 F47", ~P2[idx] & (FARM[idx] == "F47"))):
                red = 1 - np.sum((y[mm] - pr[mm]) ** 2) / np.sum(y[mm] ** 2)
                Eh = E[idx][mm]; r0 = np.sqrt(np.mean(Eh ** 2)); r1 = np.sqrt(np.mean((Eh - pr[mm, None]) ** 2))
                line.append(f"{nm}: offSSE {100*red:+6.1f}% hourly RMSE {r0:.3f}->{r1:.3f} ({100*(r1/r0-1):+5.1f}%)")
            print(f"  [{tag}] {fname} {mname:5s} | " + " | ".join(line))
    # H8: carry k(x) with oracle e0
    y = o[idx]; x0 = e0[idx]; g = grp[idx]; p1 = ~P2[idx]
    kc = np.sum(x0[p1] * y[p1]) / np.sum(x0[p1] ** 2)
    Xf = U1a.iloc[idx].copy(); Xf = Xf.fillna(Xf.median()).fillna(0).values
    prk = np.zeros(len(y))
    for k in range(10):
        tr, te = g != k, g == k
        sc = StandardScaler().fit(Xf[tr]); Z = sc.transform(Xf[tr]) * x0[tr, None]; Zt = sc.transform(Xf[te]) * x0[te, None]
        Z = np.c_[x0[tr], Z]; Zt = np.c_[x0[te], Zt]
        prk[te] = Ridge(10.0, fit_intercept=False).fit(Z, y[tr]).predict(Zt)
    for nm, mm in (("p1", p1), ("p2", ~p1), ("p2 F47", ~p1 & (FARM[idx] == "F47"))):
        s0 = np.sum(y[mm] ** 2); sc_ = np.sum((y[mm] - kc * x0[mm]) ** 2); sk = np.sum((y[mm] - prk[mm]) ** 2)
        print(f"  [{tag}] H8 {nm:6s}: oracle e0 const k={kc:.2f} offSSE {100*(1-sc_/s0):+.1f}% | learned k(x) {100*(1-sk/s0):+.1f}% | extra {100*(sc_-sk)/s0:+.1f}pp")
    return o, has, E, res, idx
for tag, Q in (("EV", QE), ("DIAG10", QD)):
    print(f"\n######## {tag}")
    o, has, E, res, idx = run_target(tag, Q)
    if tag == "EV":
        oE, hasE, EE = o, has, E
# ---- H1 partial (EV) ----
print("\n=== H1 편상관 (EV, 진실 연결 같은 ID) ===")
o, has, E = oE, hasE, EE
oa = np.where(prevT >= 0, o[np.maximum(prevT, 0)], np.nan); ra23 = np.where(prevT >= 0, E[np.maximum(prevT, 0), 23], np.nan)
g23 = np.where(prevT >= 0, (Y[:, 23] - I["in_temp"][:, 23])[np.maximum(prevT, 0)], np.nan)
def pcorr(a, b, c, m):
    A, B, Cc = [rankdata(v[m]) for v in (a, b, c)]
    ra = A - np.polyval(np.polyfit(Cc, A, 1), Cc); rb = B - np.polyval(np.polyfit(Cc, B, 1), Cc); return np.corrcoef(ra, rb)[0, 1]
for nm, sel in (("F47", FARM == "F47"), ("all", np.ones(N, bool))):
    m = has & sel & np.isfinite(oa) & np.isfinite(g23) & np.isfinite(ra23)
    for v, vv in (("g23", g23), ("ra23", ra23)):
        obs = pcorr(o, vv, oa, m); grp = (FARM[m] + P2[m].astype(str)); pm = []
        for _ in range(2000):
            oo = o[m].copy()
            for gg in np.unique(grp):
                ii = np.where(grp == gg)[0]; oo[ii] = oo[rng.permutation(ii)]
            full = o.copy(); full[np.where(m)[0]] = oo; pm.append(abs(pcorr(full, vv, oa, m)))
        print(f"  {nm}: ρ(o,{v}|oa) {obs:+.3f} n {m.sum()} perm p {(np.array(pm) >= abs(obs)).mean():.4f}")
print(f"  큰 날 부호 일치 14/16 정확 이항 p (단측) {binomtest(14, 16, .5, alternative='greater').pvalue:.4f}  (f1 로그 .88 of 16)")
# ---- H7 step / changepoint ----
print("\n=== H7 계단(번호 순서 같은 부호 이어짐) EV ===")
for f in ("F13", "F47"):
    ii = np.where(has & (FARM == f))[0]; ii = ii[np.argsort(NUM[ii])]; s = np.sign(o[ii])
    for lag in (1, 2):
        print(f"  {f} lag{lag} 같은 부호 비율 {np.mean(s[lag:] == s[:-lag]):.2f} | ρ(o_t,o_t-lag) {spearmanr(o[ii][lag:], o[ii][:-lag])[0]:+.2f}")
    rm9 = pd.Series(o[ii], index=NUM[ii]).rolling(9, center=True).mean(); print(f"  {f} 9기록 이동평균 최대/최소 {rm9.max():+.2f}/{rm9.min():+.2f} at {rm9.idxmax()}/{rm9.idxmin()}")
