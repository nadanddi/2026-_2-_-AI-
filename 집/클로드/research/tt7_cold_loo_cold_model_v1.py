# -*- coding: utf-8 -*-
"""TT7: (1) cold-day leave-one-day-out validation (COLD-LOO) and (2) cold-substrate-level dedicated models.
(2026-10-10 집 클로드, user: "추운날 부분만 loocv 돌리면 안되나?", "추운날의 배지 온도 크기만 따로 판별하는 모델을 만들어 보자")
Fixed before running.
COLD-LOO days = F13/F47 training days whose in_temp (MASK world) drops below 8 C at any hour (41 days).  Per held-out day d
  of farm f: train on all rows EXCEPT farm f days d-3..d+3 and the other farm days d-5..d+5 (other-farm same-date leak:
  F47 day = F13 day - 2).  Refit per fold: BASE seed 7 (TK2 base_predict), CODEX seed 726 (season mapping, written to
  my local folder), CM1, CM2.  PFN term = stored DIAG10 OOF (families A/B; trained on fewer days - same for ref & cand).
Dedicated cold models (train rows: in_temp < 12, weights wp; features PH = BASE phc + CODEX PHYSICS_COLUMNS +
  act_circfan, act_thermal, in_hum):
  CM1 = median impute + StandardScaler + Ridge(alpha 10)                     (linear: extrapolates)
  CM2 = CM1 + LightGBM residual (150 trees, depth 3, lr .05, min_child 40)  (adds shape)
  Candidate blend:  g * W40G-S + (1 - g) * CM,  g = clip((in_temp-8)/2, 0, 1)  -> only rows with in_temp < 10 change.
Validators: DIAG10, EXT10, EXT12, EL1 (CM refit on each fold's train part; REF from TT1 checkpoints) + COLD-LOO.
k = 2 -> alpha .0125.
USER RULE (scope COLD): every seed{7,101} x family{A,B} better on EXT10 + EXT10 farm x 5-day block bootstrap P(worse)
  < .0125; DIAG10/EXT12/EL1 fail iff seed-mean worse AND bootstrap share(better) < .0125.
LOO RULE (user-requested validator, required in addition): COLD-LOO better in F13 and F47 separately, for both
  families, and day bootstrap P(worse) < .0125.
Reported: rows in_temp < 6, bias on rows in_temp < 8 (ref vs cand), REF error on COLD-LOO vs DIAG10 for the same days.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt7_cold_loo_cold_model_v1.py   (sum = summary only)
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import lightgbm as lgb
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt7_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt7_season"
SEEDS = (7, 101); FAMS = ("A", "B"); VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def cm_fit_predict(X, tr, va, w):
    """X: frame with PH columns + sub_temp + in_temp; tr/va boolean masks."""
    PH = [c for c in X.columns if c not in ("sub_temp", "in_temp")]
    m = tr & (X.in_temp < 12).to_numpy()
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))
    lin.fit(X.loc[m, PH], X.sub_temp[m], ridge__sample_weight=w[m])
    p1 = lin.predict(X.loc[va, PH])
    res = X.sub_temp[m] - lin.predict(X.loc[m, PH])
    gb = lgb.LGBMRegressor(n_estimators=150, max_depth=3, learning_rate=.05, min_child_samples=40, verbose=-1, random_state=7)
    gb.fit(X.loc[m, PH], res, sample_weight=w[m])
    return p1, p1 + gb.predict(X.loc[va, PH])


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    assert (lab.row_id.values == pfn.row_id.values).all()
    wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    PC = list(W.TM.PHYSICS_COLUMNS)
    X = pd.concat([lab[phc + ["act_circfan", "act_thermal", "in_hum", "sub_temp", "in_temp"]].reset_index(drop=True),
                   pfn[[c for c in PC if c not in lab.columns or c in ("in_temp",)]].drop(columns=["in_temp"], errors="ignore").reset_index(drop=True).add_prefix("cx_")], axis=1)
    # A) CM on the standard validators
    for name, folds in sets:
        path = os.path.join(CK, "cm_%s.csv" % name)
        if os.path.exists(path):
            continue
        outs = []
        for k, fd in enumerate(folds):
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            p1, p2 = cm_fit_predict(X, tm, vm, wp)
            outs.append(pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values, "cm1": p1, "cm2": p2}))
        pd.concat(outs).to_csv(path, index=False); print("CM %s done" % name, flush=True)
    # B) COLD-LOO
    dmin = lab.groupby(["farm", "day"]).in_temp.min()
    days = [k for k, v in dmin.items() if v < 8]
    print("COLD-LOO days: %d" % len(days), flush=True)
    FC = list(W.FEATURE_COLUMNS)
    for f, d in days:
        path = os.path.join(CK, "loo_%s_%d.csv" % (f, d))
        if os.path.exists(path):
            continue
        same, oth = (lab.farm == f).to_numpy(), (lab.farm != f).to_numpy()
        dd = lab.day.to_numpy()
        vm = same & (dd == d)
        tm = ~((same & (np.abs(dd - d) <= 3)) | (oth & (np.abs(dd - d) <= 5)))
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt7_LOO", 0)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        out["base_7"] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, 7)
        saveFC = W.TM.FEATURE_COLUMNS
        W.TM.FEATURE_COLUMNS = [c if c != "day" else "season" for c in FC]
        try:
            out["codex"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
        finally:
            W.TM.FEATURE_COLUMNS = saveFC
        out["cm1"], out["cm2"] = cm_fit_predict(X, tm, vm, wp)
        out.to_csv(path, index=False); print("LOO %s %d done" % (f, d), flush=True)


def gate(t):
    return np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1))


def boot(cl, a, b, y, rng):
    dd = pd.Series((b - y) ** 2 - (a - y) ** 2).groupby(cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = rng.integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261010)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, "cm_%s.csv" % v)) for v in VALS])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G.cm1.notna().all()
    g = gate(G.in_temp.values)
    for s in SEEDS:
        for f in FAMS:
            G["ref_%d%s" % (s, f)] = .4 * G["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G["pfn_%s" % f]
    L = [pd.read_csv(os.path.join(CK, x)) for x in sorted(os.listdir(CK)) if x.startswith("loo_")]
    LO = pd.concat(L, ignore_index=True) if L else None
    if LO is not None:
        D = G[G.validator == "DIAG10"].set_index("row_id")
        gl = gate(LO.in_temp.values)
        for f in FAMS:
            LO["pfn_%s" % f] = D["pfn_%s" % f].reindex(LO.row_id).values
            LO["ref_%s" % f] = .4 * LO.base_7 + (.2 + .4 * (1 - gl)) * LO.codex + .4 * gl * LO["pfn_%s" % f]
            LO["diag_ref_%s" % f] = D["ref_7%s" % f].reindex(LO.row_id).values
        print("COLD-LOO %d days loaded. same days, REF RMSE: LOO %.3f vs DIAG10 %.3f (more training cold days -> LOO)" % (LO.groupby(["farm", "day"]).ngroups, r(LO.ref_A - LO.sub_temp), r(LO.diag_ref_A - LO.sub_temp)))
    for cm in ("cm1", "cm2"):
        print("\n==== %s" % cm.upper()); user_ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in SEEDS:
            for f in FAMS:
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; gg = gate(X.in_temp.values)
                    a = X["ref_%d%s" % (s, f)].values; b = gg * a + (1 - gg) * X[cm].values; y = X.sub_temp.values
                    p = boot((X.farm + "_" + (X.day // 5).astype(str)).values, a, b, y, rng)
                    rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    c6 = X.in_temp.values < 6; c8 = X.in_temp.values < 8
                    line += " %s %+.2f%% P %.3f (<6C %.3f->%.3f, bias<8C %+.2f->%+.2f) |" % (v, 100 * rel[v][-1], p, r((a - y)[c6]), r((b - y)[c6]), (a - y)[c8].mean(), (b - y)[c8].mean())
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v == "EXT10":
                user_ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                user_ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        loo_ok = LO is not None
        if LO is not None:
            gl = gate(LO.in_temp.values); y = LO.sub_temp.values
            for f in FAMS:
                a = LO["ref_%s" % f].values; b = gl * a + (1 - gl) * LO[cm].values
                p = boot((LO.farm + "_" + LO.day.astype(str)).values, a, b, y, rng)
                byf = {fm: r((b - y)[LO.farm == fm]) / r((a - y)[LO.farm == fm]) - 1 for fm in ("F13", "F47")}
                c6 = LO.in_temp.values < 6
                print("  COLD-LOO PFN %s: %.3f -> %.3f (%+.2f%%) F13 %+.2f%% F47 %+.2f%% P(worse) %.4f | <6C %.3f->%.3f n%d | bias<8C %+.2f->%+.2f" % (
                    f, r(a - y), r(b - y), 100 * (r(b - y) / r(a - y) - 1), 100 * byf["F13"], 100 * byf["F47"], p, r((a - y)[c6]), r((b - y)[c6]), c6.sum(),
                    (a - y)[LO.in_temp.values < 8].mean(), (b - y)[LO.in_temp.values < 8].mean()))
                loo_ok &= byf["F13"] < 0 and byf["F47"] < 0 and p < ALPHA
        print("VERDICT %s: user rule %s | COLD-LOO %s" % (cm.upper(), "PASS" if user_ok else "FAIL", "PASS" if loo_ok else "FAIL"))


if __name__ == "__main__":
    if MODE != "sum":
        run()
    summarize()
