# -*- coding: utf-8 -*-
"""temp_offset3 F1 (2026-10-12 집 클로드 하위 에이전트): 온도 하루 오프셋 포렌식 H1~H6 (계획 00_계획_사전고정.md).
서술 단계 — 진실 연결(truth.pkl)은 여기서만 진단용으로 사용. 모델 없음."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd, pickle
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
RES = r"C:\work\farmai\집\클로드\research"; OUTD = os.path.join(RES, "temp_offset3")
rng = np.random.default_rng(20261212)
R = pickle.load(open(os.path.join(RES, r"local\linker_v3\base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
ids = np.array([f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]); N = len(r); IDX = {k: i for i, k in enumerate(ids)}
FARM = r.farm.values; NUM = r.day.values; TR = (r.split == "train").values; OUT4 = R["OUT"]
VARS = ['out_temp', 'out_hum', 'out_rad', 'out_wspd', 'in_temp', 'in_hum', 'in_co2', 'act_vent', 'act_shade', 'act_thermal',
        'act_heating', 'act_circfan', 'act_co2', 'act_fog']
Xd = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Xd = Xd[Xd.row_id.str[:3].isin(["F13", "F47"])]
Yd = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yd = Yd[Yd.row_id.str[:3].isin(["F13", "F47"])]
def tens(df, col):
    a = np.full((N, 24), np.nan); rid = df.row_id.str[:7].values; h = df.row_id.str[-2:].astype(int).values
    ii = np.array([IDX.get(x, -1) for x in rid]); m = ii >= 0; a[ii[m], h[m]] = df[col].values[m]; return a
I = {v: tens(Xd, v) for v in VARS}; Y = tens(Yd, "sub_temp")
# ---- baseline EV v22 ----
T = pd.read_pickle(os.path.join(RES, r"ev_validator\temp\local_ckpt\ev_all.pkl")); T["rec"] = T.row_id.str[:7]
vc = [c for c in T.columns if c.startswith("v22_")]; T["q"] = T[vc].mean(1)
Q = np.full((N, 24), np.nan)
for rec, g in T.groupby("rec"): Q[IDX[rec], g.hour.values] = g.q.values
E = Y - Q; o = np.nanmean(E, 1); HAS = np.isfinite(E).all(1)
P2 = NUM >= 179
print("EV records", HAS.sum(), "pass-2", (HAS & P2).sum(), "| v22 cols", len(vc))
# ---- big days ----
big = np.zeros(N, bool)
for p in (False, True):
    m = HAS & (P2 == p); thr = np.quantile(np.abs(o[m]), .9); big |= m & (np.abs(o) >= thr)
F47 = FARM == "F47"
print("big days", big.sum(), "of which F47", (big & F47).sum())
# ---- truth links (same ID, cert) ----
TT = pd.read_pickle(os.path.join(RES, r"local\linker_v3\truth.pkl"))
prev = np.full(N, -1)
for _, t in TT.iterrows():
    if t.cert and t.same_farm: prev[int(t.b)] = int(t.a1)
hp = prev >= 0
def at(A, i, h): return A[i, h] if i >= 0 else np.nan
ra23 = np.array([at(E, prev[b], 23) if hp[b] else np.nan for b in range(N)])
oa = np.array([o[prev[b]] if hp[b] else np.nan for b in range(N)])
g23 = np.array([Y[prev[b], 23] - I["in_temp"][prev[b], 23] if hp[b] else np.nan for b in range(N)])
qg23 = np.array([Q[prev[b], 23] - I["in_temp"][prev[b], 23] if hp[b] else np.nan for b in range(N)])
e0 = E[:, 0]; g0 = Y[:, 0] - I["in_temp"][:, 0]
# ---- day features (raw) ----
def mh(v, h0, h1, A=None):
    A = I[v] if A is None else A; return np.nanmean(A[:, h0:h1 + 1], 1)
def prevf(x): return np.array([x[prev[b]] if hp[b] else np.nan for b in range(N)])
F = pd.DataFrame({"rec": ids, "farm": FARM, "num": NUM, "p2": P2, "o": o, "big": big, "e0": e0, "e6": E[:, 6], "e12": E[:, 12], "e23": E[:, 23]})
night = {"heat_n": mh("act_heating", 0, 6), "fan_n": mh("act_circfan", 0, 6), "therm_n": mh("act_thermal", 0, 6),
         "tin_n": mh("in_temp", 0, 6), "vent_d": mh("act_vent", 9, 16), "shade_d": mh("act_shade", 9, 16)}
eve = {"heat_ev": prevf(mh("act_heating", 18, 23)), "fan_ev": prevf(mh("act_circfan", 18, 23)), "therm_ev": prevf(mh("act_thermal", 18, 23)),
       "tin_ev": prevf(mh("in_temp", 18, 23))}
wx = {"tout": mh("out_temp", 0, 23), "rad": np.nansum(I["out_rad"], 1) / 1000, "tout_n": mh("out_temp", 0, 6),
      "dtout": mh("out_temp", 0, 23) - prevf(mh("out_temp", 0, 23)), "hout": mh("out_hum", 0, 23), "wsp": mh("out_wspd", 0, 23),
      "tin": mh("in_temp", 0, 23), "dtin": mh("in_temp", 0, 23) - prevf(mh("in_temp", 0, 23)), "co2": mh("in_co2", 0, 23), "hin": mh("in_hum", 0, 23)}
for d in (night, eve, wx):
    for k, v in d.items(): F[k] = v
F["ra23"] = ra23; F["oa"] = oa; F["g23"] = g23; F["qg23"] = qg23; F["g0"] = g0; F["hasprev"] = hp
F = F[HAS].reset_index(drop=True)
F.to_csv(os.path.join(OUTD, "f1_records_v1.csv"), index=False, encoding="utf-8-sig")
def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b); return (spearmanr(a[m], b[m])[0], int(m.sum())) if m.sum() > 5 else (np.nan, int(m.sum()))
print("\n=== 오차 모양 ===")
for p in (False, True):
    m = (F.p2 == p)
    for fm in ("F13", "F47"):
        mm = m & (F.farm == fm); print(f" pass-{2 if p else 1} {fm}: n {mm.sum()} offset sd {F.o[mm].std():.3f} mean {F.o[mm].mean():+.3f} | big {int((mm & F.big).sum())}")
print("\n 큰 날 목록(2차 + 1차 F47 큰 날):")
cols = ["rec", "o", "e0", "e6", "e12", "e23", "hasprev", "oa", "ra23", "g23", "qg23", "g0", "tin_n", "heat_n", "fan_n", "therm_n", "tout", "rad", "dtout", "dtin"]
print(F[F.big & ((F.farm == "F47") | F.p2)].sort_values("o")[cols].round(2).to_string(index=False))
# ---- H6 shape ----
print("\n=== H6 시간 모양: 큰 날 부호 정렬 잔차 시각별 평균 (F47) ===")
Eh = E[HAS]; s = np.sign(F.o.values)
for nm, m in (("F47 big", (F.big & (F.farm == "F47")).values), ("F47 big +", (F.big & (F.farm == "F47") & (F.o > 0)).values),
              ("F47 big -", (F.big & (F.farm == "F47") & (F.o < 0)).values), ("all other", (~F.big).values)):
    prof = np.nanmean(Eh[m] * s[m, None], 0); print(f" {nm:10s} n {m.sum():3d} | " + " ".join(f"{x:+.1f}" for x in prof[::2]))
# ---- H1 ----
print("\n=== H1 전날 이월(진실 연결 같은 ID cert) ===")
for nm, m in (("all", np.ones(len(F), bool)), ("F47", (F.farm == "F47").values), ("F47 p1", ((F.farm == "F47") & ~F.p2).values),
              ("F47 p2", ((F.farm == "F47") & F.p2).values), ("F13", (F.farm == "F13").values)):
    out = []
    for v in ("ra23", "oa", "g23", "g0", "e0"):
        rho, n = sp(F.o.values[m], F[v].values[m]); out.append(f"{v} {rho:+.2f}(n{n})")
    print(f" {nm:7s} ρ(o, ·): " + " | ".join(out))
mb = (F.big & (F.farm == "F47") & F.hasprev).values
agree = np.mean(np.sign(F.o[mb]) == np.sign(F.ra23[mb])) if mb.sum() else np.nan
print(f" F47 큰 날 중 전날 연결 있음 {mb.sum()} / {(F.big & (F.farm=='F47')).sum()}; 부호 일치(o vs ra23) {agree:.2f}")
# ---- H2/H3 permutation max-stat within farm x pass ----
def maxstat(cols, m, nperm=2000):
    sub = F[m].reset_index(drop=True); grp = (sub.farm + sub.p2.astype(str)).values
    def stats(oo): return np.array([abs(sp(oo, sub[c].values)[0]) for c in cols])
    obs = stats(sub.o.values); mx = np.zeros(nperm)
    for k in range(nperm):
        oo = sub.o.values.copy()
        for g in np.unique(grp):
            ii = np.where(grp == g)[0]; oo[ii] = oo[rng.permutation(ii)]
        mx[k] = np.nanmax(stats(oo))
    return obs, np.array([(mx >= v).mean() for v in obs])
H2 = list(night) + list(eve); H3 = list(wx)
for nm, m in (("F47", (F.farm == "F47").values), ("all", np.ones(len(F), bool))):
    for hn, cols_ in (("H2 밤운영", H2), ("H3 외기·실내", H3)):
        obs, p = maxstat(cols_, m, 1000)
        print(f"\n=== {hn} [{nm}] |ρ(o,·)| 와 최대통계 순열 p ===")
        print("  " + " | ".join(f"{c} {sp(F.o.values[m], F[c].values[m])[0]:+.2f} p{pp:.3f}" for c, pp in zip(cols_, p)))
# ---- H4 siblings ----
print("\n=== H4 같은 날짜 형제 ===")
tr = np.where(HAS)[0]; Xv = OUT4[tr].reshape(len(tr), -1); pairs = []
for k, i in enumerate(tr):
    eq = (np.abs(Xv[k + 1:] - Xv[k]) < 1e-6).mean(1)
    for j in tr[k + 1:][eq > .7]: pairs.append((i, j))
P = pd.DataFrame(pairs, columns=["i", "j"]); P["same_farm"] = FARM[P.i] == FARM[P.j]
P["oi"] = o[P.i]; P["oj"] = o[P.j]; P["fi"] = FARM[P.i]; P["fj"] = FARM[P.j]; P["ri"] = ids[P.i]; P["rj"] = ids[P.j]
for c in ("tin", "tin_n", "heat_n", "fan_n", "therm_n", "vent_d", "shade_d", "co2", "hin"):
    full = np.full(N, np.nan); full[HAS] = F[c].values; P["d_" + c] = full[P.i] - full[P.j]
P["do"] = P.oi - P.oj; P["big_any"] = big[P.i] | big[P.j]
P.to_csv(os.path.join(OUTD, "f1_sibling_pairs_v1.csv"), index=False, encoding="utf-8-sig")
for nm, m in (("same farm", P.same_farm), ("cross farm", ~P.same_farm), ("same farm F47", P.same_farm & (P.fi == "F47"))):
    if m.sum() < 4: print(" ", nm, "n", m.sum()); continue
    rho = sp(P.oi[m].values, P.oj[m].values)[0]; sdo = np.std(np.r_[P.oi[m], P.oj[m]])
    print(f" {nm:14s} pairs {m.sum():3d} | ρ(o_i,o_j) {rho:+.2f} | sd Δo {P.do[m].std():.3f} vs sd o {sdo:.3f} | ratio {P.do[m].std()/sdo:.2f}")
    dc = [c for c in P.columns if c.startswith("d_") and c != "do"]
    Z = P.loc[m, dc].fillna(0).values; z = P.do[m].values
    if m.sum() >= 10:
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); pr = np.zeros(len(z))
        for k in range(len(z)):
            tr_ = np.arange(len(z)) != k; pr[k] = Ridge(1.0, fit_intercept=False).fit(Z[tr_], z[tr_]).predict(Z[k:k + 1])[0]
        print(f"    Δo ~ Δ입력 릿지 LOO R² {1 - np.sum((z - pr) ** 2) / np.sum(z ** 2):+.3f}  | 단변수 ρ: " +
              " ".join(f"{c[2:]} {sp(z, P.loc[m, c].values)[0]:+.2f}" for c in dc))
print(" 큰 날이 낀 같은 농장 형제 쌍:")
print(P[P.same_farm & P.big_any][["ri", "rj", "oi", "oj", "do", "d_tin", "d_tin_n", "d_heat_n", "d_fan_n", "d_therm_n"]].round(2).to_string(index=False))
print(" 큰 날이 낀 다른 농장 형제 쌍:")
print(P[~P.same_farm & P.big_any][["ri", "rj", "oi", "oj", "do", "d_tin", "d_tin_n", "d_heat_n"]].round(2).to_string(index=False))
# ---- H5 sign ----
print("\n=== H5 부호별: 큰 + 날 vs 큰 − 날 (F47, 1·2차 합침; 표본 작음) ===")
mpos = (F.big & (F.farm == "F47") & (F.o > 0)).values; mneg = (F.big & (F.farm == "F47") & (F.o < 0)).values
cand = H2 + H3 + ["g23", "ra23", "oa", "g0"]
lab = np.r_[np.ones(mpos.sum()), np.zeros(mneg.sum())]
def aucs(lb, A):
    out = []
    for c in cand:
        x = A[c].values; mm = np.isfinite(x); out.append(roc_auc_score(lb[mm], x[mm]) if len(np.unique(lb[mm])) == 2 and mm.sum() > 5 else np.nan)
    return np.array(out)
A = pd.concat([F[mpos], F[mneg]]); au = aucs(lab, A); dev = np.abs(au - .5)
mx = np.array([np.nanmax(np.abs(aucs(rng.permutation(lab), A) - .5)) for _ in range(1000)])
print(f" n+ {mpos.sum()} n- {mneg.sum()}")
for c, a in sorted(zip(cand, au), key=lambda t: -abs(t[1] - .5) if np.isfinite(t[1]) else 0)[:10]:
    print(f"  {c:9s} AUC {a:.2f}  perm-max p {(mx >= abs(a - .5)).mean():.3f}  | +mean {F.loc[mpos, c].mean():.2f} -mean {F.loc[mneg, c].mean():.2f} 기타F47 {F.loc[(F.farm=='F47')&~F.big, c].mean():.2f}")
# DIAG10 robustness of offsets (same records)
import importlib.util
sys.argv = ["x"]; s_ = importlib.util.spec_from_file_location("ta2", os.path.join(RES, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A2)
M, cells = A2.A1.frames()["DIAG10"]; M = M.copy(); M["q"] = np.mean(np.vstack(cells), 0); M["e"] = M.sub_temp - M.q
od = M.groupby(M.row_id.str[:7]).e.mean(); F["o_diag"] = F.rec.map(od)
print("\n EV o vs DIAG10 o  ρ", round(sp(F.o.values, F.o_diag.values)[0], 3), "n", F.o_diag.notna().sum())
print(F[F.big & (F.farm == "F47")][["rec", "o", "o_diag"]].round(2).to_string(index=False))
F.to_csv(os.path.join(OUTD, "f1_records_v1.csv"), index=False, encoding="utf-8-sig")
