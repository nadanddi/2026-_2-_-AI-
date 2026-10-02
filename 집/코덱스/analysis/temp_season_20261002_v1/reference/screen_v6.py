# -*- coding: utf-8 -*-
"""Round 5 screening, judged ONLY by validators calibrated on real scores.

Calibration (research/데이터_단서_카탈로그.md 7.7):
  sub_temp -> EXTRAP fold (every greenhouse-day colder than 10 C held out at
              once): reproduced both real temperature changes (0.862 vs 0.895,
              0.858 vs 0.844).  Geometry A/B only as a "do no harm" check.
  sub_ec   -> geometry A: right direction on both real EC changes (0.746 vs
              0.937, 0.938 vs 0.899).  B as the secondary check.
Round 4 failed because the choice was made on uncalibrated thresholds.

Member out-of-fold predictions are computed once per fold set and stored;
blend weights are then combined offline.

Temperature candidates
  weights   re-balance resid-LGB / Ridge / Nystroem.  The real gains came from
            the extrapolating linear members; 0.65/0.25/0.10 was chosen on the
            warm geometry CV.
  cleanw    down-weight training rows flagged by input-only physical checks
            (eda_forensic_11 V1,V3-V7) and +-3 h around them to 0.2.  1.74% of
            training rows, 0.56% of test rows; flagged rows have 2.4x residuals.
EC candidates
  weights   re-balance ET(fp) / LGB tweedie / MLP, never re-tuned since the
            fingerprint went into ET.
  fp mix    blend ET with and without the fingerprint (the fingerprint makes
            day-to-day alternation of 0.4 vs 1.3-1.5 in some test blocks).

Run:  cd research && PYTHONPATH="" <python> -u screen_v6.py
"""
import itertools

import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import lgbh, ridge, nys, THRESH
from ec_v6 import et, ltw, mlp
from make_submission_v3 import causal_shrink
from feat_lib import paired_block_boot

CLEAN_W = 0.2


def flag_weights(lab):
    f = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv")
    bad = set(f.loc[(f.set == "train") & f.Vany.astype(bool), "row_id"])
    key = lab[["row_id", "farm", "t"]].copy()
    hit = key[key.row_id.isin(bad)]
    w = np.ones(len(lab))
    for farm, g in hit.groupby("farm"):
        ts = g.t.values
        m = (key.farm.values == farm) & (np.abs(key.t.values[:, None] - ts[None, :]) <= 3).any(1)
        w[m] = CLEAN_W
    return w


def fit_w(model, X, y, w, step):
    if w is None:
        return model.fit(X, y)
    if step is None:
        return model.fit(X, y, sample_weight=w)
    return model.fit(X, y, **{step + "__sample_weight": w})


def temp_members(tr, va, ct, phc, w=None):
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    b = fit_w(LinearRegression(), imp.transform(tr[phc]), tr.sub_temp.values, w, None)
    btr, bva = b.predict(imp.transform(tr[phc])), b.predict(imp.transform(va[phc]))
    y = tr.sub_temp.values
    r = fit_w(lgbh(), tr[ct], y - btr, w, None).predict(va[ct])
    return {"res": bva + r,
            "ridge": fit_w(ridge(), tr[ct], y, w, "ridge").predict(va[ct]),
            "nys": fit_w(nys(), tr[ct], y, w, "ridge").predict(va[ct])}


def collect(lab, fds, fn):
    out = {}
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        if not vam.sum():
            continue
        idx = np.where(vam)[0]
        for k, p in fn(lab[trm], lab[vam].reset_index(drop=True), trm).items():
            out.setdefault(k, np.full(len(lab), np.nan))[idx] = p
    return out


def boot(lab, target, a, b):
    g = ~np.isnan(a) & ~np.isnan(b)
    sub = lab[g].reset_index(drop=True)
    return paired_block_boot(sub, target, a[g], b[g], n_boot=2000, seed=0, level="row")


def main():
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab_t = (lab_t0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                   .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc, fpc = F4.names(ph), F4.names(fp)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    yt, ye = lab_t.sub_temp.values, lab_e.sub_ec.values
    W_all = flag_weights(lab_t)
    print("clean-weight rows: %d of %d (weight %.1f)" % (int((W_all < 1).sum()), len(W_all), CLEAN_W))

    # ---------------- temperature ------------------------------------------
    dmin = lab_t.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    tsets = [("EXT10", ext), ("geomA", folds("A")), ("geomB", folds("B"))]
    TM = {}
    for sname, fds in tsets:
        TM[(sname, "plain")] = collect(lab_t, fds, lambda tr, va, m: temp_members(tr, va, ct, phc))
        TM[(sname, "cleanw")] = collect(lab_t, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, W_all[m]))
        print("  temp %s done" % sname, flush=True)

    def tb(M, wr, wg, wn):
        return wr * M["res"] + wg * M["ridge"] + wn * M["nys"]

    grid = [(wr, wg, round(1 - wr - wg, 2)) for wr in (0.45, 0.55, 0.65, 0.75)
            for wg in (0.15, 0.25, 0.35, 0.45) if 1 - wr - wg >= -1e-9]
    print("\n== sub_temp: weights x cleanw  (primary EXT10; geomA/geomB = do-no-harm) ==")
    print("%-20s %-7s %9s %9s %9s" % ("res/ridge/nys", "train", "EXT10", "geomA", "geomB"))
    rows = []
    for mode in ("plain", "cleanw"):
        for wr, wg, wn in grid:
            s = []
            for sname, _ in tsets:
                p = tb(TM[(sname, mode)], wr, wg, wn)
                g = ~np.isnan(p)
                s.append(rmse(p[g], yt[g]))
            rows.append((mode, wr, wg, wn, *s))
    base = next(r for r in rows if r[0] == "plain" and (r[1], r[2], r[3]) == (0.65, 0.25, 0.10))
    for r in sorted(rows, key=lambda r: r[4])[:10] + [base]:
        mark = "  <- round 3" if r is base else ""
        print("%-20s %-7s %9.4f %9.4f %9.4f%s" % ("%.2f/%.2f/%.2f" % r[1:4], r[0], r[4], r[5], r[6], mark))
    ok = [r for r in rows if r[5] <= base[5] + 0.002 and r[6] <= base[6] + 0.002]
    best_t = min(ok, key=lambda r: r[4])
    print("best EXT10 among do-no-harm: %s %.2f/%.2f/%.2f  EXT10 %.4f (round 3 %.4f)"
          % (best_t[0], best_t[1], best_t[2], best_t[3], best_t[4], base[4]))
    a = tb(TM[("EXT10", "plain")], 0.65, 0.25, 0.10)
    b = tb(TM[("EXT10", best_t[0])], *best_t[1:4])
    pr, lo, hi, pw = boot(lab_t, "sub_temp", a, b)
    print("  paired EXT10: %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f" % (pr, lo, hi, pw))

    # ---------------- EC -----------------------------------------------------
    def ec_members(tr, va, m):
        y = tr.sub_ec.values
        return {"et_fp": et().fit(tr[f14 + fpc], y).predict(va[f14 + fpc]),
                "et": et().fit(tr[f14], y).predict(va[f14]),
                "ltw": ltw().fit(tr[f14], y).predict(va[f14]),
                "mlp": mlp().fit(tr[f14], y).predict(va[f14])}

    esets = [("A", folds("A")), ("B", folds("B"))]
    EM = {s: collect(lab_e, fds, ec_members) for s, fds in esets}
    print("\n  ec members done", flush=True)

    def eb(M, fr, w_et, w_l, w_m):
        p = w_et * (fr * M["et_fp"] + (1 - fr) * M["et"]) + w_l * M["ltw"] + w_m * M["mlp"]
        g = ~np.isnan(p)
        out = np.full(len(p), np.nan)
        out[g] = np.clip(causal_shrink(p[g], lab_e[g].reset_index(drop=True), 0.5), 0.062, 3.46)
        return out

    egrid = [(fr, we, wl, round(1 - we - wl, 2)) for fr in (1.0, 0.75, 0.5)
             for we in (0.5, 0.6, 0.7) for wl in (0.2, 0.3, 0.4) if 1 - we - wl >= -1e-9]
    print("\n== sub_ec: weights x fingerprint share  (primary A; B = do-no-harm) ==")
    print("%-8s %-16s %9s %9s" % ("fp share", "et/ltw/mlp", "A", "B"))
    er = []
    for fr, we, wl, wm in egrid:
        s = []
        for sname, _ in esets:
            p = eb(EM[sname], fr, we, wl, wm)
            g = ~np.isnan(p)
            s.append(rmse(p[g], ye[g]))
        er.append((fr, we, wl, wm, *s))
    ebase = next(r for r in er if r[:4] == (1.0, 0.6, 0.3, 0.1))
    for r in sorted(er, key=lambda r: r[4])[:8] + [ebase]:
        mark = "  <- round 3" if r is ebase else ""
        print("%-8.2f %-16s %9.4f %9.4f%s" % (r[0], "%.2f/%.2f/%.2f" % r[1:4], r[4], r[5], mark))
    eok = [r for r in er if r[5] <= ebase[5] + 0.001]
    best_e = min(eok, key=lambda r: r[4])
    print("best A among do-no-harm: fp %.2f  %.2f/%.2f/%.2f  A %.4f (round 3 %.4f)  B %.4f (%.4f)"
          % (best_e[0], best_e[1], best_e[2], best_e[3], best_e[4], ebase[4], best_e[5], ebase[5]))
    for sname, fds in esets:
        a = eb(EM[sname], 1.0, 0.6, 0.3, 0.1)
        b = eb(EM[sname], *best_e[:4])
        idx = [np.where(split_mask(lab_e, fd)[1])[0] for fd in fds]
        d = [rmse(b[i], ye[i]) - rmse(a[i], ye[i]) for i in idx]
        pr, lo, hi, pw = boot(lab_e, "sub_ec", a, b)
        print("  paired %s: %+.4f | folds %d/5 | CI [%+.4f,%+.4f] P(worse)=%.3f"
              % (sname, pr, sum(x < 0 for x in d), lo, hi, pw))


if __name__ == "__main__":
    main()
