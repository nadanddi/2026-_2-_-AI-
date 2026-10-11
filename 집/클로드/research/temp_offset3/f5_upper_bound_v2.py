# -*- coding: utf-8 -*-
"""temp_offset3 F5: 비평 반영 상한 재측정(무작위 작은 덩어리 겹, 1차로 축소 선택·2차 판정, 추정기 4종) + H9' (계획 04)."""
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
LOWD = ["qg0", "in_temp_n", "inout_d", "act_heating_n", "act_thermal_n"]
def feats(Q, o, E):
    base = S.copy(); base["q_d"] = np.nanmean(Q, 1); base["qg0"] = Q[:, 0] - I["in_temp"][:, 0]; base["qg23"] = Q[:, 23] - I["in_temp"][:, 23]
    base["inout_d"] = base["in_temp_d"] - base["out_temp_d"]
    pvc = [c for c in base.columns if c.endswith("_d") or c.endswith("_e") or c == "qg23"]
    B = pd.DataFrame({("p7_" + c): np.where(prev7 >= 0, base[c].values[np.maximum(prev7, 0)], np.nan) for c in pvc}); B["p7_has"] = (prev7 >= 0) * 1.
    U1a = pd.concat([base, B], axis=1)
    oa = np.where(prevT >= 0, o[np.maximum(prevT, 0)], 0.); ra = np.where(prevT >= 0, E[np.maximum(prevT, 0), 23], 0.)
    oa = np.nan_to_num(oa); ra = np.nan_to_num(ra)
    C = pd.DataFrame({"oa": oa, "ra23": ra, "has": (prevT >= 0) * 1.})
    return U1a, C, base[LOWD]
def cv(Xall, y, idx, folds, kind):
    pr = np.zeros(len(idx)); Xf = Xall.iloc[idx].values.astype(float)
    for k in range(10):
        tr, te = folds != k, folds == k
        Xtr, Xte = Xf[tr].copy(), Xf[te].copy()
        med = np.nanmedian(Xtr, 0); med = np.nan_to_num(med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        if kind == "lgb":
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=7, learning_rate=.03, min_child_samples=10, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=7, verbose=-1).fit(Xtr, y[tr]); pr[te] = m.predict(Xte)
        else:
            lo, hi = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0); Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
            sc = StandardScaler().fit(Xtr); a = 300. if kind == "ridge" else 10.
            pr[te] = Ridge(a).fit(sc.transform(Xtr), y[tr]).predict(sc.transform(Xte))
    return pr
out = []
for tag, Q in (("EV", QE), ("DIAG10", QD)):
    E = Y - Q; has = np.isfinite(E).all(1); o = np.nanmean(E, 1); idx = np.where(has)[0]; y = o[idx]
    U1a, C, LD = feats(Q, o, E); p2 = P2[idx]
    for seed in (1, 2, 3):
        rs = np.random.default_rng(seed); folds = np.zeros(len(idx), int)
        for f in ("F13", "F47"):
            ii = np.where(FARM[idx] == f)[0]; ii = ii[np.argsort(NUM[idx][ii])]
            blocks = np.array_split(ii, int(np.ceil(len(ii) / 4))); asg = rs.permutation(len(blocks)) % 10
            for b_, a_ in zip(blocks, asg): folds[b_] = a_
        for nm, Xall, kind in (("A U1a lgb", U1a, "lgb"), ("B U1a ridge300", U1a, "ridge"), ("C U1b lowdim(truth)", C, "low"), ("D U1a lowdim", LD, "low")):
            pr = cv(Xall, y, idx, folds, kind)
            red = lambda m, lam: 1 - np.sum((y[m] - lam * pr[m]) ** 2) / np.sum(y[m] ** 2)
            lam = max((0, .25, .5, .75, 1), key=lambda l: red(~p2, l))
            Eh = E[idx]; r0 = np.sqrt(np.mean(Eh[p2] ** 2)); r1 = np.sqrt(np.mean((Eh[p2] - lam * pr[p2, None]) ** 2))
            row = dict(val=tag, seed=seed, est=nm, p1_raw=red(~p2, 1), lam=lam, p2_sel=red(p2, lam), p2_raw=red(p2, 1),
                       p2F47_sel=red(p2 & (FARM[idx] == "F47"), lam), p2_hourly=r1 / r0 - 1)
            out.append(row)
D = pd.DataFrame(out); D.to_csv(os.path.join(OUTD, "f5_upper_bound_v2.csv"), index=False)
pd.set_option("display.width", 200)
print((D.set_index(["val", "est", "seed"]) * 1).round(3).to_string())
print("\n3시드 평균:"); print(D.groupby(["val", "est"])[["p1_raw", "lam", "p2_sel", "p2_raw", "p2F47_sel", "p2_hourly"]].mean().round(3).to_string())
# ---- H9' sibling deviation (training diagnostic) ----
print("\n=== H9' 밤 실내 − 같은 날짜 형제 밤 실내 평균 ===")
E = Y - QE; has = np.isfinite(E).all(1); o = np.nanmean(E, 1); tr = np.where(has)[0]
Xv = R["OUT"][tr].reshape(len(tr), -1); tin_n = np.nanmean(I["in_temp"][:, :6], 1); dev = np.full(N, np.nan)
for k, i in enumerate(tr):
    eq = (np.abs(Xv - Xv[k]) < 1e-6).mean(1); sib = tr[(eq > .7) & (tr != i)]
    if len(sib): dev[i] = tin_n[i] - np.nanmean(tin_n[sib])
for thr in (-3, -2):
    m = has & (dev <= thr)
    print(f" dev ≤ {thr}: n {m.sum()} | + 부호 {np.mean(o[m] > 0):.2f} | 평균 o {np.nanmean(o[m]):+.2f} | 기록 " + ", ".join(f"{ids[i]}({o[i]:+.1f},{dev[i]:+.1f})" for i in np.where(m)[0]))
m = has & np.isfinite(dev); print(" ρ(o, dev) 전체", round(spearmanr(o[m], dev[m])[0], 3), "| F47", round(spearmanr(o[m & (FARM == 'F47')], dev[m & (FARM == 'F47')])[0], 3))
