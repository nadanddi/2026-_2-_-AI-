# -*- coding: utf-8 -*-
"""TD13: can MORNING inputs tell in advance that the substrate will not warm up in the afternoon?
(2026-10-10 집 클로드, user: "배지가 낮에 안 데워지는 날을 아침 입력으로 미리 알 수 있는지 확인해봐")  Diagnostic.  Fixed before running:
Targets per day (F13/F47 training days, labels used only as targets):
  PM  = mean W40G-S DIAG10 residual (seed 7, PFN A) over hours 13-23 (positive = afternoon overpredicted = did not warm)
  AMP = (sub max - sub min over 8-17 h) / (in_temp max - min over 8-17 h)   (substrate response to daytime warming)
  NOWARM = AMP in the lowest quartile of its farm (classification)
Feature sets (inputs only, same farm):
  MORN (causal for hours >= 10): hours 0-9 of the day: in_temp mean/min/slope, out_temp mean/min, in_hum, in_co2 mean,
       act_heating/act_circfan/act_thermal/act_vent mean, out_rad sum (6-9 h), midnight in_temp jump (0 h - prev 23 h),
       previous record day means of in_temp/out_temp/act_heating/act_thermal, previous day in_temp amplitude.
  FULL (non-causal upper bound): MORN + same summaries over hours 10-23 + day out_rad sum.
Models: leave-one-day-out Ridge(alpha 10, standardized) and LightGBM(100 trees, depth 2, min_child 15); R^2 with day
  bootstrap 95% CI; NOWARM: LODO logistic (C=.3) AUC with bootstrap CI.
Sets: ALL400, COLD41 (in_temp < 8 any hour), F47COLD24.
'Predictable from morning' iff MORN R^2 (PM) > .10 with CI lower > 0 on ALL400 AND > 0 point estimate on COLD41;
  stop rule: if FULL upper bound R^2 < .10 on a set, MORN is not examined further there.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td13_morning_predicts_nowarm_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
CK = os.path.join(env.LOCAL, "tt1_ckpt")
ACT = ["act_heating", "act_circfan", "act_thermal", "act_vent"]


def summ(x, tag):
    s = {}
    for c in ["in_temp", "out_temp", "in_hum", "in_co2"] + ACT:
        s["%s_%s_mean" % (tag, c)] = x[c].mean()
    s["%s_in_min" % tag] = x.in_temp.min(); s["%s_out_min" % tag] = x.out_temp.min()
    s["%s_in_slope" % tag] = np.polyfit(x.hour, x.in_temp.fillna(x.in_temp.mean()), 1)[0] if x.in_temp.notna().sum() > 2 else np.nan
    return s


def build():
    tX, ty, _ = common.load_raw()
    a = tX[tX.farm.isin(["F13", "F47"])].merge(ty[["row_id", "sub_temp"]], on="row_id").sort_values(["farm", "t"])
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.validator == "DIAG10"]
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["e"] = .4 * G.base_REF_7 + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G.pfn_A - G.sub_temp
    a = a.merge(G[["row_id", "e"]], on="row_id")
    rows = []
    for (f, d), x in a.groupby(["farm", "day"]):
        m, af, dt = x[x.hour <= 9], x[x.hour >= 10], x[(x.hour >= 8) & (x.hour <= 17)]
        p = a[(a.farm == f) & (a.day == d - 1)]
        r = dict(farm=f, day=d, PM=x[x.hour >= 13].e.mean(),
                 AMP=(dt.sub_temp.max() - dt.sub_temp.min()) / max(dt.in_temp.max() - dt.in_temp.min(), .5),
                 cold=bool((x.in_temp < 8).any()))
        r.update(summ(m, "m"))
        r["m_rad"] = x[(x.hour >= 6) & (x.hour <= 9)].out_rad.sum()
        r["m_jump"] = (m[m.hour == 0].in_temp.mean() - p[p.hour == 23].in_temp.mean()) if len(p) else np.nan
        r["p_in"] = p.in_temp.mean() if len(p) else np.nan; r["p_out"] = p.out_temp.mean() if len(p) else np.nan
        r["p_heat"] = p.act_heating.mean() if len(p) else np.nan; r["p_th"] = p.act_thermal.mean() if len(p) else np.nan
        r["p_amp"] = (p.in_temp.max() - p.in_temp.min()) if len(p) else np.nan
        r.update(summ(af, "a")); r["a_rad"] = x.out_rad.sum()
        rows.append(r)
    D = pd.DataFrame(rows)
    D["NOWARM"] = D.groupby("farm").AMP.transform(lambda s: s <= s.quantile(.25)).astype(int)
    return D


def lodo(X, y, model):
    p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        p[i] = model().fit(X[m], y[m]).predict(X[i:i + 1])[0]
    return p


def r2ci(y, p, n=3000):
    R = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    rng = np.random.default_rng(0); b = [R(y[j], p[j]) for j in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return R(y, p), *np.percentile(b, [2.5, 97.5])


def main():
    D = build()
    MORN = [c for c in D.columns if c.startswith(("m_", "p_"))]
    FULL = MORN + [c for c in D.columns if c.startswith("a_")]
    sets = {"ALL400": D, "COLD41": D[D.cold], "F47COLD24": D[D.cold & (D.farm == "F47")]}
    ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    gbm = lambda: lgb.LGBMRegressor(n_estimators=100, max_depth=2, min_child_samples=15, learning_rate=.05, verbose=-1)
    res = {}
    for nm, S in sets.items():
        S = S.reset_index(drop=True)
        print("\n######## %s (%d일)  PM 평균 %+.2f sd %.2f | AMP 중앙 %.2f" % (nm, len(S), S.PM.mean(), S.PM.std(), S.AMP.median()))
        for tgt in ("PM", "AMP"):
            y = S[tgt].values
            for fs_name, fs in (("FULL(상한)", FULL), ("MORN(아침)", MORN)):
                X = S[fs].values
                out = []
                for mn, mdl in (("Ridge", ridge), ("LGB", gbm)):
                    if mn == "LGB" and len(S) < 60:
                        continue
                    R, lo, hi = r2ci(y, lodo(X, y, mdl)); out.append("%s R² %+.3f [%+.3f, %+.3f]" % (mn, R, lo, hi))
                    res[(nm, tgt, fs_name, mn)] = (R, lo)
                print("  %-3s %-10s %s" % (tgt, fs_name, " | ".join(out)))
        y = S.NOWARM.values
        if y.sum() >= 4:
            for fs_name, fs in (("FULL(상한)", FULL), ("MORN(아침)", MORN)):
                X = S[fs].values; p = np.zeros(len(y))
                for i in range(len(y)):
                    m = np.arange(len(y)) != i
                    if len(set(y[m])) < 2:
                        p[i] = y[m].mean(); continue
                    p[i] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.3, max_iter=2000)).fit(X[m], y[m]).predict_proba(X[i:i + 1])[0, 1]
                rng = np.random.default_rng(1); b = []
                for _ in range(2000):
                    j = rng.integers(0, len(y), len(y))
                    if len(set(y[j])) == 2:
                        b.append(roc_auc_score(y[j], p[j]))
                print("  NOWARM(안 데워지는 날 %d일) %-10s AUC %.3f [%.3f, %.3f]" % (y.sum(), fs_name, roc_auc_score(y, p), *np.percentile(b, [2.5, 97.5])))
        if nm == "F47COLD24":
            from scipy.stats import spearmanr
            print("  아침 특징별 Spearman(PM):", {c: round(spearmanr(S[c], S.PM, nan_policy="omit")[0], 2) for c in MORN})
    ok = res[("ALL400", "PM", "MORN(아침)", "Ridge")][0] > .10 and res[("ALL400", "PM", "MORN(아침)", "Ridge")][1] > 0 and res[("COLD41", "PM", "MORN(아침)", "Ridge")][0] > 0
    ok2 = res[("ALL400", "PM", "MORN(아침)", "LGB")][0] > .10 and res[("ALL400", "PM", "MORN(아침)", "LGB")][1] > 0 and res[("COLD41", "PM", "MORN(아침)", "Ridge")][0] > 0
    print("\n판정(오후 오차 PM을 아침 입력으로): Ridge %s, LGB %s" % ("예측 가능" if ok else "정보 부족", "예측 가능" if ok2 else "정보 부족"))


if __name__ == "__main__":
    main()
