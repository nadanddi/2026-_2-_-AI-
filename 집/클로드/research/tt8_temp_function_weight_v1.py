# -*- coding: utf-8 -*-
"""TT8: temperature model FUNCTION and WEIGHT adjustments (2026-10-10 집 클로드, user: "모델 함수 및 가중치도 조정해보자").
Fixed before running (after plan-stage critic).  Reference W40G-S = 0.4 BASE + (0.2+0.4(1-g)) CODEX + 0.4 g PFN, TT1 members.
 F1 (function, scope ALL): BASE physics LinearRegression (no penalty, 14 collinear ewm features, cond ~3800, TD11) ->
    median impute + StandardScaler + RidgeCV(alphas 10^-2..10^3, 12 values, GCV) with the same sample weights; LGB
    huber residual (same seed), ridge and nystroem members unchanged.  BASE seeds 7/101.
 F2 (function, scope ALL): CODEX LGB residual objective L2 -> huber (LightGBM default alpha .9 = same as BASE res).
    CODEX seeds 726 and 727 (727 reference refitted).  |residual| > .9 share reported.
 W  (weights, scope ALL): constrained least squares (w >= 0, sum 1) on stored members: warm rows (g = 1) over
    (BASE, CODEX, PFN), cold rows (g = 0) over (BASE, CODEX); prediction = g*warm + (1-g)*cold.  DIAG10: cross-fit by
    farm x 5-day blocks (fit on other blocks excluding +-1 neighbour blocks).  EXT10/EXT12/EL1: fit once on DIAG10 OOF
    rows outside that validator's days (same farm +-3, other farm +-5).  Caveats: similar in spirit to rejected 6.178
    TGATE / TT2; reference weights were partly chosen post hoc on these validators (6.70) -> comparison tilted against
    W; DIAG10 fold models saw EXT labels (small leak, ~5 dof).  EXPECTATION written before running: W likely FAILS
    (DIAG10 gain, EXT10 loss, as TGATE).
 Reproduction gate: before any candidate, the unchanged path (LinearRegression BASE seed 7, CODEX L2 seed 726) is refit
    on DIAG10 fold 0 and must match TT1 stored REF within 1e-9, else abort.
k = 3 -> alpha .025/3 = .00833.
RULE (scope ALL, user temperature rule): every seed x PFN family better on DIAG10 AND EXT10, farm x 5-day block
  bootstrap P(worse) < .00833 on both; EXT12/EL1 fail iff seed-mean worse AND share(better) < .00833 (loose by design).
Reported (not judged): pass-2 rows, rows in_temp < 6 (21 train label rows - report only), clean rows (weight 1).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt8_temp_function_weight_v1.py   (sum = summary only)
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
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt8_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt8_season"
SG = W.temp_members.__globals__          # screen_v6 namespace: lgbh, ridge, nys, fit_w, cold_v5 seed
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .025 / 3
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def base_members(tr, va, ct, phc, w, seed, physics):
    W.cold_v5.SEED = seed
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    if physics == "LR":
        b = SG["fit_w"](LinearRegression(), imp.transform(tr[phc]), tr.sub_temp.values, w, None)
        btr, bva = b.predict(imp.transform(tr[phc])), b.predict(imp.transform(va[phc]))
    else:
        b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        b.fit(tr[phc], tr.sub_temp.values, ridgecv__sample_weight=w)
        btr, bva = b.predict(tr[phc]), b.predict(va[phc])
    y = tr.sub_temp.values
    rr = SG["fit_w"](SG["lgbh"](), tr[ct], y - btr, w, None).predict(va[ct])
    res = bva + rr
    rid = SG["fit_w"](SG["ridge"](), tr[ct], y, w, "ridge").predict(va[ct])
    ny = SG["fit_w"](SG["nys"](), tr[ct], y, w, "ridge").predict(va[ct])
    return .65 * res + .25 * rid + .10 * ny, float(b[-1].alpha_) if physics == "RCV" else np.nan


def codex(tr, va, w, seed, obj):
    TM = W.TM
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[TM.PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    kw = dict(objective="huber") if obj == "HUB" else {}
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=seed, **kw)
    res = tr.sub_temp.values - lin.predict(tr[TM.PHYSICS_COLUMNS])
    m.fit(tr[TM.FEATURE_COLUMNS], res, sample_weight=w)
    return lin.predict(va[TM.PHYSICS_COLUMNS]) + m.predict(va[TM.FEATURE_COLUMNS]), float((np.abs(res) > .9).mean())


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    FC = list(W.FEATURE_COLUMNS)
    ref = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt8_" + name, k)
            tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True); out["validator"] = name
            saveFC = W.TM.FEATURE_COLUMNS
            W.TM.FEATURE_COLUMNS = [c if c != "day" else "season" for c in FC]
            try:
                if name == "DIAG10" and k == 0:
                    lr7, _ = base_members(lab[tm], lab[vm].reset_index(drop=True), ct, phc, wb[tm], 7, "LR")
                    c726, _ = codex(tr, va, wp[tm], 726, "L2")
                    R = ref[ref.validator == "DIAG10"].set_index("row_id").reindex(out.row_id)
                    d1, d2 = np.abs(lr7 - R.base_REF_7.values).max(), np.abs(c726 - R.codex_REF.values).max()
                    print("REPRO gate: BASE LR seed7 max diff %.2e, CODEX L2 726 max diff %.2e" % (d1, d2), flush=True)
                    assert d1 < 1e-9 and d2 < 1e-9, "reproduction failed - abort"
                for s in (7, 101):
                    out["base_F1_%d" % s], out["alpha_%d" % s] = base_members(lab[tm], lab[vm].reset_index(drop=True), ct, phc, wb[tm], s, "RCV")
                out["codex_L2_727"], _ = codex(tr, va, wp[tm], 727, "L2")
                for s in (726, 727):
                    out["codex_HUB_%d" % s], out["big_res_share"] = codex(tr, va, wp[tm], s, "HUB")
            finally:
                W.TM.FEATURE_COLUMNS = saveFC
            out.to_csv(path, index=False)
            print("%s/%d done (RidgeCV alpha %.3g/%.3g, |res|>.9 share %.3f)" % (name, k, out.alpha_7[0], out.alpha_101[0], out.big_res_share[0]), flush=True)


def gate(t):
    return np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1))


def cls(X, cols):
    """constrained LS weights (>=0, sum 1)."""
    A, y = X[cols].values, X.sub_temp.values
    n = len(cols)
    f = lambda w: np.mean((A @ w - y) ** 2)
    res = minimize(f, np.ones(n) / n, bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return res.x


def w_predict(fit, app, s, f):
    B, P = "base_REF_%d" % s, "pfn_%s" % f
    gw, ga = gate(fit.in_temp.values), gate(app.in_temp.values)
    ww = cls(fit[gw == 1], [B, "codex_REF", P]); wc = cls(fit[gw == 0], [B, "codex_REF"])
    warm = app[[B, "codex_REF", P]].values @ ww; cold = app[[B, "codex_REF"]].values @ wc
    return ga * warm + (1 - ga) * cold, ww, wc


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series(((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2)).groupby(cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = rng.integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261010)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    have = os.path.isdir(CK) and len([x for x in os.listdir(CK) if x.endswith(".csv")]) >= 24
    if have:
        C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.endswith(".csv")])
        G = G.merge(C, on=["validator", "row_id"], how="left"); assert G.codex_HUB_726.notna().all()
    lab_w = None
    try:
        import train_flags_v6 as TF  # noqa
    except Exception:
        TF = None
    g = gate(G.in_temp.values)
    blend = lambda X, b, c, p: .4 * X[b] + (.2 + .4 * (1 - gate(X.in_temp.values))) * X[c] + .4 * gate(X.in_temp.values) * X[p]
    # W predictions
    D = G[G.validator == "DIAG10"].copy(); D["blk"] = D.farm + "_" + (D.day // 5).astype(str)
    for s in (7, 101):
        for f in ("A", "B"):
            G["W_%d%s" % (s, f)] = np.nan
            pd_ = np.full(len(D), np.nan)
            for b in D.blk.unique():
                fm, bi = b.split("_")[0], int(b.split("_")[1])
                excl = (D.farm == fm) & (np.abs(D.day // 5 - bi) <= 1)
                app = (D.blk == b).values
                pd_[app] = w_predict(D[~excl.values], D[app], s, f)[0]
            G.loc[G.validator == "DIAG10", "W_%d%s" % (s, f)] = pd_
            for v in ("EXT10", "EXT12", "EL1"):
                V = G[G.validator == v]; vd = set(zip(V.farm, V.day))
                near = np.array([any((ff == fa and abs(dd - da) <= 3) or (ff != fa and abs(dd - da) <= 5) for (fa, da) in vd) for ff, dd in zip(D.farm, D.day)])
                p, ww, wc = w_predict(D[~near], V, s, f)
                G.loc[G.validator == v, "W_%d%s" % (s, f)] = p
                if s == 7 and f == "A":
                    print("W weights fit for %s: warm (B,C,P) %s cold (B,C) %s" % (v, np.round(ww, 2), np.round(wc, 2)))
    cands = {"W": lambda X, s, f: X["W_%d%s" % (s, f)]}
    if have:
        cands["F1"] = lambda X, s, f: blend(X, "base_F1_%d" % s, "codex_REF", "pfn_%s" % f)
    out = {}
    for cn, fn in cands.items():
        print("\n==== %s" % cn); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in (7, 101):
            for f in ("A", "B"):
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; a = blend(X, "base_REF_%d" % s, "codex_REF", "pfn_%s" % f).values; b = fn(X, s, f).values
                    y = X.sub_temp.values; p = boot(X, a, b, rng)
                    rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    L = X.day.values >= 179; c6 = X.in_temp.values < 6
                    line += " %s %+.2f%% P %.3f (p2 %+.2f%%, <6C %.2f->%.2f) |" % (v, 100 * rel[v][-1], p, 100 * (r((b - y)[L]) / r((a - y)[L]) - 1), r((a - y)[c6]) if c6.any() else np.nan, r((b - y)[c6]) if c6.any() else np.nan)
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in ("DIAG10", "EXT10"):
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT %s: %s" % (cn, "PASS" if ok else "FAIL"))
    if have:
        print("\n==== F2 (CODEX huber vs L2, CODEX seeds 726/727 x BASE 7/101 x PFN A/B)"); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for cs in (726, 727):
            refc = "codex_REF" if cs == 726 else "codex_L2_727"
            for s in (7, 101):
                for f in ("A", "B"):
                    line = "codex %d seed %3d PFN %s |" % (cs, s, f)
                    for v in VALS:
                        X = G[G.validator == v]; y = X.sub_temp.values
                        a = blend(X, "base_REF_%d" % s, refc, "pfn_%s" % f).values; b = blend(X, "base_REF_%d" % s, "codex_HUB_%d" % cs, "pfn_%s" % f).values
                        p = boot(X, a, b, rng); rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                        line += " %s %+.2f%% P %.3f |" % (v, 100 * rel[v][-1], p)
                    print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in ("DIAG10", "EXT10"):
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT F2: %s" % ("PASS" if ok else "FAIL"))
    else:
        print("\nF1/F2: checkpoints incomplete")


if __name__ == "__main__":
    if MODE != "sum":
        run()
    summarize()
