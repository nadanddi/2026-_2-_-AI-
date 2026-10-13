# -*- coding: utf-8 -*-
"""HX9-OPS 판정 v1 (2026-10-13 집 클로드) — 관리·구동기 지문 가족 변수(ops_vars_v1.py 산출)의 고EC 크기 연관 시험.
사전 예측·판정 규칙은 ops_vars_v1.py 머리말 (계산 전 기록). 여기 코드에 같은 규칙을 고정:
  signal = |r_resid_high| >= .35 AND 사슬 부트스트랩 95% CI 0 제외 AND 사슬 LOCO R2 이득 > 0
  weak   = signal 아님 AND (|r_resid_high| >= .25 OR CI 0 제외 OR LOCO 이득 > 0)
  none   = 나머지
결과 변수:
  Y1  = 하루 평균 EC, hi10 기록(35)                       [정답 수준]
  Y2h = ec_mean − sg_mean (DIAG10 시드평균), hi10 & OOF(31) [모델 잔차]
  Y2a = 같은 잔차, OOF 있는 전체 360                        [전체 잔차]
  Y3  = ec_mean(b) − ec_mean(clean 전날), b·전날 모두 hi10 [에피소드 안 하루 변화; 링크는 서술용]
  Y3i = ec_h23 − ec_h0, hi10 (35)                           [하루 안 변화, flush 포함]
교차적합: 고EC = 에피소드(ep_id, clean 링크 연결 성분) 하나씩 빼기(LOCO). 전체 = DIAG10 폴드 빼기.
다중검정: 변수 수 보고 + Y2h 기록 순열 max-|r| (FWER, 사슬 자기상관 때문에 반보수적) + BH q.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings, json
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(20261013)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
R["season_out"] = HO.groupby("rec").out_temp.mean().loc[R.index]
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec").loc[R.index]
M = pd.read_csv(os.path.join(HERE, "vars_v1_meta.csv")).set_index("variable")
VARS = [v for v in M.index if v != "op_cluster"]          # cluster label is nominal; its encodings are tested instead
print("variables tested:", len(VARS), "(usable", (M.loc[VARS].usable == "usable").sum(), ")")
R = R.join(F[VARS])   # (crash fix 1st run: F was not joined; criteria unchanged)

hi = R.hi10 == 1
S1 = R[hi].copy()                                   # Y1, Y3i
S2 = R[hi & (R.has_sg == 1)].copy()                 # Y2h
SA = R[R.has_sg == 1].copy()                        # Y2a
S3 = R[hi & (R.prev_hi10 == 1)].copy()              # Y3
S1["Y1"] = S1.ec_mean; S1["Y3i"] = S1.ec_h23 - S1.ec_h0
S2["Y2"] = S2.resid_mean; SA["Y2"] = SA.resid_mean
S3["Y3"] = S3.ec_mean - S3.prev_ec_mean
print(f"n: Y1 {len(S1)} (chains {S1.ep_id.nunique()}), Y2h {len(S2)} (chains {S2.ep_id.nunique()}), Y2a {len(SA)}, Y3 {len(S3)} (chains {S3.ep_id.nunique()})")

def rr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.nanstd(x[m]) < 1e-12: return np.nan, np.nan, int(m.sum())
    return stats.pearsonr(x[m], y[m])[0], stats.spearmanr(x[m], y[m])[0], int(m.sum())

def partial_r(x, y, Zc):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Zc), 1)
    if m.sum() < 8 or np.std(x[m]) < 1e-12: return np.nan
    Zm = np.column_stack([np.ones(m.sum()), Zc[m]])
    rx = x[m] - Zm @ np.linalg.lstsq(Zm, x[m], rcond=None)[0]
    ry = y[m] - Zm @ np.linalg.lstsq(Zm, y[m], rcond=None)[0]
    return stats.pearsonr(rx, ry)[0] if np.std(rx) > 1e-12 else np.nan

def chain_boot(x, y, g, B=2000):
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; out = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        xx, yy = x[ii], y[ii]; m = np.isfinite(xx) & np.isfinite(yy)
        if m.sum() < 5 or np.std(xx[m]) < 1e-12 or np.std(yy[m]) < 1e-12: continue
        out.append(np.corrcoef(xx[m], yy[m])[0, 1])
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if len(out) > 100 else (np.nan, np.nan)

def rec_boot(x, y, B=2000):
    n = len(x); out = []
    for _ in range(B):
        ii = rng.integers(0, n, n); xx, yy = x[ii], y[ii]; m = np.isfinite(xx) & np.isfinite(yy)
        if m.sum() < 5 or np.std(xx[m]) < 1e-12: continue
        out.append(np.corrcoef(xx[m], yy[m])[0, 1])
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

def cv_gain(x, y, base, g):
    """LOGO linear: y ~ 1 + base  vs  y ~ 1 + base + x ; NaN in x imputed by training-fold median. gain = (SSEb−SSEa)/SST."""
    pb = np.zeros_like(y); pa = np.zeros_like(y)
    for u in np.unique(g):
        te = g == u; tr = ~te
        xm = np.nanmedian(x[tr]) if np.isfinite(x[tr]).any() else 0.0
        xx = np.where(np.isfinite(x), x, xm)
        Xb = np.column_stack([np.ones(tr.sum()), base[tr]]); Xa = np.column_stack([Xb, xx[tr]])
        bb = np.linalg.lstsq(Xb, y[tr], rcond=None)[0]; ba = np.linalg.lstsq(Xa, y[tr], rcond=None)[0]
        pb[te] = np.column_stack([np.ones(te.sum()), base[te]]) @ bb
        pa[te] = np.column_stack([np.ones(te.sum()), base[te], xx[te]]) @ ba
    sst = np.sum((y - y.mean()) ** 2)
    gain = (np.sum((y - pb) ** 2) - np.sum((y - pa) ** 2)) / sst
    nimp = int(np.sum(np.abs(y - pa) < np.abs(y - pb)))
    r2b = 1 - np.sum((y - pb) ** 2) / sst
    return gain, nimp, r2b

g2 = S2.ep_id.values.astype(int); y2 = S2.Y2.values; b2 = S2.sg_mean.values
gA = SA.fold.values.astype(int); yA = SA.Y2.values; bA = SA.sg_mean.values
rows = []
for v in VARS:
    d = dict(variable=v, usable=M.at[v, "usable"], avail_h=M.at[v, "avail_h"], definition=M.at[v, "definition"])
    x1 = S1[v].values.astype(float); x2 = S2[v].values.astype(float); xA = SA[v].values.astype(float); x3 = S3[v].values.astype(float)
    d["r_Y1"], d["rho_Y1"], d["n_Y1"] = rr(x1, S1.Y1.values)
    d["r_Y2h"], d["rho_Y2h"], d["n_Y2h"] = rr(x2, y2)
    d["r_Y2a"], d["rho_Y2a"], d["n_Y2a"] = rr(xA, yA)
    d["r_Y3"], d["rho_Y3"], d["n_Y3"] = rr(x3, S3.Y3.values)
    d["r_Y3i"], d["rho_Y3i"], d["n_Y3i"] = rr(x1, S1.Y3i.values)
    d["sd_in_high"] = float(np.nanstd(x2)); d["nuniq_high"] = int(pd.Series(x2).nunique())
    if np.isfinite(d["r_Y2h"]):
        d["ci_chain_lo"], d["ci_chain_hi"] = chain_boot(x2, y2, g2)
        d["ci_rec_lo"], d["ci_rec_hi"] = rec_boot(x2, y2)
        d["cv_gain_Y2h"], d["n_improved_Y2h"], d["r2_base_Y2h"] = cv_gain(x2, y2, b2, g2)
        d["pr_Y2h_season_sg"] = partial_r(x2, y2, np.column_stack([S2.season_out.values, b2]))
        loo = []
        for u in np.unique(g2):
            k = g2 != u; loo.append(rr(x2[k], y2[k])[0])
        d["loco_r_min"], d["loco_r_max"] = np.nanmin(loo), np.nanmax(loo)
        d["r_Y2h_F13"] = rr(x2[S2.farm.values == "F13"], y2[S2.farm.values == "F13"])[0]
        d["r_Y2h_F47"] = rr(x2[S2.farm.values == "F47"], y2[S2.farm.values == "F47"])[0]
        p1 = S2.pass2.values == 0
        d["r_Y2h_pass1"] = rr(x2[p1], y2[p1])[0]
        # pass2 high only 5 records -> report covariance sign via Spearman if possible
        p2 = ~p1; m2 = np.isfinite(x2[p2])
        d["rho_Y2h_pass2_n5"] = stats.spearmanr(x2[p2][m2], y2[p2][m2])[0] if m2.sum() >= 4 and np.std(x2[p2][m2]) > 0 else np.nan
    if np.isfinite(d["r_Y2a"]):
        d["cv_gain_Y2a"], d["n_improved_Y2a"], _ = cv_gain(xA, yA, bA, gA)
        pa1 = SA.pass2.values == 0
        d["r_Y2a_pass1"] = rr(xA[pa1], yA[pa1])[0]; d["r_Y2a_pass2"] = rr(xA[~pa1], yA[~pa1])[0]
    rows.append(d)
T = pd.DataFrame(rows).set_index("variable")

# ---------------- multiple testing on Y2h: record permutation max-|r| ----------------
Xh = S2[[v for v in VARS]].values.astype(float)
def all_r(yv):
    out = []
    for j in range(Xh.shape[1]):
        x = Xh[:, j]; m = np.isfinite(x)
        out.append(np.corrcoef(x[m], yv[m])[0, 1] if m.sum() >= 5 and np.std(x[m]) > 1e-12 else np.nan)
    return np.array(out)
obs = np.abs(all_r(y2)); NP = 3000; mx = np.zeros(NP); cnt = np.zeros(len(VARS))
for k in range(NP):
    rp = np.abs(all_r(rng.permutation(y2))); mx[k] = np.nanmax(rp); cnt += (rp >= obs - 1e-12)
T["p_perm_Y2h"] = (cnt + 1) / (NP + 1)
T["p_fwer_maxT_Y2h"] = [(np.sum(mx >= o - 1e-12) + 1) / (NP + 1) if np.isfinite(o) else np.nan for o in obs]
ps = T.p_perm_Y2h.values; o = np.argsort(ps); q = np.empty_like(ps); m_ = len(ps)
q[o] = np.minimum.accumulate((ps[o] * m_ / np.arange(1, m_ + 1))[::-1])[::-1]; T["q_bh_Y2h"] = np.minimum(q, 1)

# ---------------- verdicts (pre-fixed rule) ----------------
def verdict(d):
    r = d.r_Y2h; ci_ex = np.isfinite(d.ci_chain_lo) and (d.ci_chain_lo > 0 or d.ci_chain_hi < 0); gp = np.isfinite(d.cv_gain_Y2h) and d.cv_gain_Y2h > 0
    if not np.isfinite(r): return "none"
    if abs(r) >= .35 and ci_ex and gp: return "signal"
    if abs(r) >= .25 or ci_ex or gp: return "weak"
    return "none"
T["verdict"] = [verdict(T.loc[v]) for v in T.index]
T.to_csv(os.path.join(HERE, "test_v1_univariate.csv"), encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["usable", "n_Y1", "r_Y1", "n_Y2h", "r_Y2h", "rho_Y2h", "ci_chain_lo", "ci_chain_hi", "cv_gain_Y2h", "n_improved_Y2h", "pr_Y2h_season_sg",
        "loco_r_min", "loco_r_max", "r_Y2h_F13", "r_Y2h_F47", "r_Y2h_pass1", "rho_Y2h_pass2_n5", "r_Y2a", "cv_gain_Y2a", "r_Y2a_pass1", "r_Y2a_pass2",
        "n_Y3", "r_Y3", "r_Y3i", "p_perm_Y2h", "p_fwer_maxT_Y2h", "q_bh_Y2h", "verdict"]
print(T[cols].round(3).sort_values("r_Y2h", key=np.abs, ascending=False).to_string())
print("verdict counts", T.verdict.value_counts().to_dict())
print("baseline LOCO R2 (Y2h ~ sg):", round(T.r2_base_Y2h.dropna().iloc[0], 3))

# ---------------- multivariate non-causal upper bound ----------------
def mv_eval(Xm, y, g, model, seeds=(0,)):
    p = np.zeros_like(y)
    for u in np.unique(g):
        te = g == u; tr = ~te
        med = np.nanmedian(Xm[tr], 0); med = np.where(np.isfinite(med), med, 0)
        Xi = np.where(np.isfinite(Xm), Xm, med)
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9; Xs = (Xi - mu) / sd
        if model == "ridge":
            # nested alpha choice by inner LOGO
            best, ba = np.inf, 10
            gi = g[tr]
            for a in [1, 3, 10, 30, 100, 300]:
                sse = 0
                for w in np.unique(gi):
                    t2 = gi == w
                    mdl = Ridge(alpha=a).fit(Xs[tr][~t2], y[tr][~t2]); sse += np.sum((y[tr][t2] - mdl.predict(Xs[tr][t2])) ** 2)
                if sse < best: best, ba = sse, a
            p[te] = Ridge(alpha=ba).fit(Xs[tr], y[tr]).predict(Xs[te])
        else:
            pp = np.zeros(te.sum())
            for s in seeds:
                mdl = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=s)
                pp += mdl.fit(Xs[tr], y[tr]).predict(Xs[te]) / len(seeds)
            p[te] = pp
    sst = np.sum((y - y.mean()) ** 2); sse0 = np.sum(y ** 2)
    return dict(R2_vs_mean=1 - np.sum((y - p) ** 2) / sst, R2_vs_nocorr=1 - np.sum((y - p) ** 2) / sse0,
                n_improved=int(np.sum(np.abs(y - p) < np.abs(y))), n=len(y)), p

MV = {}
sets = {"all49": VARS, "usable": [v for v in VARS if M.at[v, "usable"] == "usable"]}
for nm, vs in sets.items():
    for addsg in [False, True]:
        Xm = S2[vs].values.astype(float)
        if addsg: Xm = np.column_stack([Xm, b2])
        for model in ["ridge", "gbm"]:
            res, p = mv_eval(Xm, y2, g2, model, seeds=(0, 1, 2))
            # permutation null (record-level shuffle of y)
            NPm = 200 if model == "ridge" else 100
            null = []
            for k in range(NPm):
                yp = rng.permutation(y2); null.append(mv_eval(Xm, yp, g2, model, seeds=(0,))[0]["R2_vs_mean"])
            res["p_perm"] = (np.sum(np.array(null) >= res["R2_vs_mean"]) + 1) / (NPm + 1)
            res["null_q95"] = float(np.percentile(null, 95))
            key = f"Y2h|{nm}{'+sg' if addsg else ''}|{model}"; MV[key] = res
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
# all-record residual, DIAG10 folds (usable set + sg)
for nm, vs in sets.items():
    Xm = np.column_stack([SA[vs].values.astype(float), bA])
    for model in ["ridge", "gbm"]:
        res, p = mv_eval(Xm, yA, gA, model, seeds=(0, 1, 2))
        hm = SA.hi10.values == 1
        res["hi_SSE_ratio"] = float(np.sum((yA[hm] - p[hm]) ** 2) / np.sum(yA[hm] ** 2))
        res["hi_n_improved"] = int(np.sum(np.abs(yA[hm] - p[hm]) < np.abs(yA[hm]))); res["hi_n"] = int(hm.sum())
        p1 = SA.pass2.values == 0
        res["pass1_R2_vs_nocorr"] = float(1 - np.sum((yA[p1] - p[p1]) ** 2) / np.sum(yA[p1] ** 2))
        res["pass2_R2_vs_nocorr"] = float(1 - np.sum((yA[~p1] - p[~p1]) ** 2) / np.sum(yA[~p1] ** 2))
        key = f"Y2a|{nm}+sg|{model}"; MV[key] = res
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
json.dump(MV, open(os.path.join(HERE, "test_v1_multivariate.json"), "w"), indent=1, default=float)

# ---------------- episode entry (P4): schedule change on entry days ----------------
E = R.join(F[[c for c in F.columns if c.startswith("chg_")]], rsuffix="_v")
ent = E[(E.hi10 == 1) & (E.prev_hi10 == 0)]
oth = E[E.prev_rec.notna() & ~E.index.isin(ent.index)]
print("\nP4 entry days (clean prev not hi10 -> hi10):", list(ent.index))
for c in ["chg_clean_L1", "chg_clean_L1am", "chg_inp_L1"]:
    pr = [stats.percentileofscore(oth[c].dropna(), x) for x in ent[c]]
    print(f"  {c}: entry values {np.round(ent[c].values, 3)} percentile vs other transitions {np.round(pr, 0)}; other median {oth[c].median():.3f}")
# flush-like day-to-day drops within episodes
dr = S3.sort_values("Y3")
print("\nY3 largest drops / rises with chg_clean_L1 (other-transition median %.3f):" % oth.chg_clean_L1.median())
print(dr[["prev_rec", "Y3", "ec_h0", "ec_h23"]].join(F[["chg_clean_L1", "chg_clean_L1am", "chg_clean_co2h", "chg_clean_heath", "fog_hours", "vent_hours"]]).round(3).iloc[np.r_[0:5, -3:0]].to_string())
print("\nhigh-record intraday drops (Y3i) top 6:")
print(S1.sort_values("Y3i")[["Y3i", "ec_h0", "ec_h23"]].join(F[["chg_clean_L1", "fog_hours", "fog_pulses", "vent_hours", "heat_hours", "co2_hours"]]).head(6).round(3).to_string())
