# -*- coding: utf-8 -*-
"""sub_ec: paired decision test for the ExtraTrees + boosting blend.

Why this file exists
--------------------
Comparing two candidates by asking whether their RMSE gap exceeds the
fold-to-fold standard deviation is the wrong test.  Both candidates see the
same folds, so fold difficulty cancels in the difference and the variance of
the DELTA is far smaller than the variance of either RMSE.  The right tests
are (a) the sign of the per-fold delta, which must be consistent, and (b) a
paired greenhouse-day block bootstrap of the delta, taken from feat_lib.

Compute note: harness.Blend fits its members independently and averages their
predictions, and harness.score averages over seeds, so a weighted blend's OOF
is exactly the same weighted average of the members' OOF vectors.  Each member
is therefore fitted once per fold set and every weight in the sweep is formed
arithmetically.  The identity is asserted against a real Blend fit.

Run:  cd research && PYTHONPATH="" <python> -u model_ec_paired.py
"""
import numpy as np

import env  # noqa: F401  MUST be first project import
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from harness import load, score, views, folds, blend_factory
from common import split_mask, rmse
from model_common import seg_stats
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)
HGBP = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)

REF = lambda s: make_pipeline(SimpleImputer(strategy="median"),
                              ExtraTreesRegressor(random_state=s, **ET1))
LGBHUB = lambda s: lgb.LGBMRegressor(random_state=s, objective="huber",
                                     **DET, **LGBP)
HGBSQ = lambda s: HistGradientBoostingRegressor(random_state=s,
                                                loss="squared_error", **HGBP)


def fold_index(lab, kind):
    """Row indices held out by each fold of the given placement."""
    return [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]


def per_fold(lab, target, oof, idx):
    y = lab[target].values.astype(float)
    return [rmse(oof[i], y[i]) for i in idx]


def report(lab, target, oof_ref, oof_alt, idx, name):
    y = lab[target].values.astype(float)
    fr = per_fold(lab, target, oof_ref, idx)
    fa = per_fold(lab, target, oof_alt, idx)
    d = [b - a for a, b in zip(fr, fa)]
    nneg = sum(1 for x in d if x < 0)
    pr, lo, hi, pw = paired_block_boot(lab, target, oof_ref, oof_alt,
                                       n_boot=2000, seed=0, level="row")
    dr, dlo, dhi, dpw = paired_block_boot(lab, target, oof_ref, oof_alt,
                                          n_boot=2000, seed=0, level="day")
    g = ~np.isnan(oof_alt)
    s0, s1 = seg_stats(y[g], oof_ref[g]), seg_stats(y[g], oof_alt[g])
    print("  %s" % name)
    print("    per-fold delta : %s   (better in %d/5 folds)"
          % (" ".join("%+.4f" % x for x in d), nneg))
    print("    row-level      : %+.4f  95%% CI [%+.4f, %+.4f]  P(worse)=%.3f"
          % (pr, lo, hi, pw))
    print("    day-level      : %+.4f  95%% CI [%+.4f, %+.4f]  P(worse)=%.3f"
          % (dr, dlo, dhi, dpw))
    print("    y>1 segment    : rmse %.4f -> %.4f | bias %+.4f -> %+.4f"
          % (s0["rmse"], s1["rmse"], s0["bias"], s1["bias"]))


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    v5 = v["v5"]
    y = lab_e["sub_ec"].values.astype(float)

    members = [("ET600/1", REF), ("LGBhuber", LGBHUB), ("HGBsq", HGBSQ)]
    O, IDX = {}, {}
    print("== members (v5, 18 features) ==")
    print("  %-10s %8s %8s | %9s %9s" % ("", "A rmse", "B rmse",
                                         "A hi-bias", "B hi-bias"))
    for nm, fac in members:
        row = []
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab_e, "sub_ec", v5, fac, kind=kind,
                                      return_oof=True)
            O[(nm, kind)] = oof
            IDX[kind] = fold_index(lab_e, kind)
            g = ~np.isnan(oof)
            row.append((r, seg_stats(y[g], oof[g])["bias"]))
        print("  %-10s %8.4f %8.4f | %+9.4f %+9.4f"
              % (nm, row[0][0], row[1][0], row[0][1], row[1][1]))

    # ---- the arithmetic-blend identity, verified once, not assumed ---------
    w = 0.25
    chk = blend_factory([REF, LGBHUB], [1 - w, w])
    (_, _, _), oof_fit = score(lab_e, "sub_ec", v5, chk, kind="A",
                               return_oof=True)
    oof_arith = (1 - w) * O[("ET600/1", "A")] + w * O[("LGBhuber", "A")]
    g = ~np.isnan(oof_fit)
    print("\n  blend identity check: max|fitted - arithmetic| = %.2e"
          % float(np.max(np.abs(oof_fit[g] - oof_arith[g]))))

    # ---- 2-way weight sweep ----------------------------------------------
    print("\n== ET600/1 + LGBhuber : weight sweep ==")
    print("  %5s %8s %8s | %8s %8s | %8s %9s"
          % ("w", "A rmse", "dA", "B rmse", "dB", "hiRMSE", "hiBias"))
    ws = np.round(np.arange(0.00, 0.61, 0.05), 2)
    best = {}
    for kind in ("A", "B"):
        rs = [rmse((1 - w) * O[("ET600/1", kind)][~np.isnan(O[("ET600/1", kind)])]
                   + w * O[("LGBhuber", kind)][~np.isnan(O[("ET600/1", kind)])],
                   y[~np.isnan(O[("ET600/1", kind)])]) for w in ws]
        best[kind] = (ws[int(np.argmin(rs))], min(rs))
    for w in ws:
        cells = []
        for kind in ("A", "B"):
            gg = ~np.isnan(O[("ET600/1", kind)])
            p = (1 - w) * O[("ET600/1", kind)][gg] + w * O[("LGBhuber", kind)][gg]
            r0 = rmse(O[("ET600/1", kind)][gg], y[gg])
            cells.append((rmse(p, y[gg]), rmse(p, y[gg]) - r0))
        ga = ~np.isnan(O[("ET600/1", "A")])
        pa = (1 - w) * O[("ET600/1", "A")][ga] + w * O[("LGBhuber", "A")][ga]
        s = seg_stats(y[ga], pa)
        print("  %5.2f %8.4f %+8.4f | %8.4f %+8.4f | %8.4f %+9.4f"
              % (w, cells[0][0], cells[0][1], cells[1][0], cells[1][1],
                 s["rmse"], s["bias"]))
    print("  argmin: placement A w=%.2f (%.4f) | placement B w=%.2f (%.4f)"
          % (best["A"][0], best["A"][1], best["B"][0], best["B"][1]))

    # ---- paired verdict at the candidate weights -------------------------
    for w in (0.25, float(best["A"][0]), float(best["B"][0])):
        print("\n== paired test: ET600/1 vs ET600/1+LGBhuber w=%.2f ==" % w)
        for kind in ("A", "B"):
            alt = (1 - w) * O[("ET600/1", kind)] + w * O[("LGBhuber", kind)]
            print(" placement %s" % kind)
            report(lab_e, "sub_ec", O[("ET600/1", kind)], alt, IDX[kind],
                   "2-way w=%.2f" % w)

    # ---- 3-way grid -------------------------------------------------------
    print("\n== 3-way ET600/1 + LGBhuber(wl) + HGBsq(wh) ==")
    print("  %5s %5s %8s %8s %8s %9s" % ("wl", "wh", "A rmse", "B rmse",
                                         "hiRMSE", "hiBias"))
    grid = []
    for wl in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4):
        for wh in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4):
            if wl + wh > 0.6:
                continue
            rr = {}
            for kind in ("A", "B"):
                gg = ~np.isnan(O[("ET600/1", kind)])
                p = ((1 - wl - wh) * O[("ET600/1", kind)][gg]
                     + wl * O[("LGBhuber", kind)][gg]
                     + wh * O[("HGBsq", kind)][gg])
                rr[kind] = rmse(p, y[gg])
            ga = ~np.isnan(O[("ET600/1", "A")])
            pa = ((1 - wl - wh) * O[("ET600/1", "A")][ga]
                  + wl * O[("LGBhuber", "A")][ga] + wh * O[("HGBsq", "A")][ga])
            s = seg_stats(y[ga], pa)
            grid.append((wl, wh, rr["A"], rr["B"], s["rmse"], s["bias"]))
            print("  %5.2f %5.2f %8.4f %8.4f %8.4f %+9.4f"
                  % (wl, wh, rr["A"], rr["B"], s["rmse"], s["bias"]))
    bA = min(grid, key=lambda r: r[2])
    bB = min(grid, key=lambda r: r[3])
    print("  argmin A: wl=%.2f wh=%.2f | argmin B: wl=%.2f wh=%.2f"
          % (bA[0], bA[1], bB[0], bB[1]))

    for wl, wh, lbl in [(0.2, 0.2, "equalish 0.6/0.2/0.2"),
                        (bA[0], bA[1], "argmin-A"), (bB[0], bB[1], "argmin-B")]:
        print("\n== paired test: ET600/1 vs 3-way (%s: ET %.2f / LGB %.2f / "
              "HGB %.2f) ==" % (lbl, 1 - wl - wh, wl, wh))
        for kind in ("A", "B"):
            alt = ((1 - wl - wh) * O[("ET600/1", kind)]
                   + wl * O[("LGBhuber", kind)] + wh * O[("HGBsq", kind)])
            print(" placement %s" % kind)
            report(lab_e, "sub_ec", O[("ET600/1", kind)], alt, IDX[kind],
                   "3-way")
            alt2 = 0.75 * O[("ET600/1", kind)] + 0.25 * O[("LGBhuber", kind)]
            report(lab_e, "sub_ec", alt2, alt, IDX[kind],
                   "   ...vs the 2-way w=0.25")

    np.save("local/ec_oof_members.npy",
            np.array([[O[(nm, k)] for k in ("A", "B")]
                      for nm, _ in members]))
    print("\nsaved member OOF to local/ec_oof_members.npy")


if __name__ == "__main__":
    main()
