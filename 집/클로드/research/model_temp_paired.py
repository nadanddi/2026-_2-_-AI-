# -*- coding: utf-8 -*-
"""sub_temp: paired decision test for LightGBM + a smooth extrapolating partner.

On sub_temp every non-tree family loses on its own (Ridge .899/.832,
Nystroem 1.01/.97, SVR 1.00/1.00, MLP 1.22/1.25 against LGB huber .860/.786),
but the blend table showed 75 LGB / 25 Ridge at .8478/.7752 on both
placements.  That is the claim tested here, with per-fold deltas and the
paired greenhouse-day block bootstrap rather than a fold-sd comparison.

Run:  cd research && PYTHONPATH="" <python> -u model_temp_paired.py
"""
import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import load, score, views, folds
from common import split_mask, rmse
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)


def LGBH(s):
    return lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)


def RIDGE(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=100.0))


def NYS(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=s),
                         Ridge(alpha=1.0))


def fold_index(lab, kind):
    return [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]


def report(lab, target, ref, alt, idx, name):
    y = lab[target].values.astype(float)
    d = [rmse(alt[i], y[i]) - rmse(ref[i], y[i]) for i in idx]
    nb = sum(1 for x in d if x < 0)
    pr, lo, hi, pw = paired_block_boot(lab, target, ref, alt, n_boot=2000,
                                       seed=0, level="row")
    dr, dlo, dhi, dpw = paired_block_boot(lab, target, ref, alt, n_boot=2000,
                                          seed=0, level="day")
    print("  %s" % name)
    print("    per-fold delta : %s   (better in %d/5)"
          % (" ".join("%+.4f" % x for x in d), nb))
    print("    row  %+.4f  CI [%+.4f, %+.4f]  P(worse)=%.3f" % (pr, lo, hi, pw))
    print("    day  %+.4f  CI [%+.4f, %+.4f]  P(worse)=%.3f"
          % (dr, dlo, dhi, dpw))


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    tv = v["temp"]
    y = lab_t["sub_temp"].values.astype(float)

    O = {}
    for nm, fac, seeds in [("LGB", LGBH, (7, 101, 2024)),
                           ("Ridge", RIDGE, (7,)),
                           ("Nys", NYS, (7, 101, 2024))]:
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab_t, "sub_temp", tv, fac, kind=kind,
                                      seeds=seeds, return_oof=True)
            O[(nm, kind)] = oof
            print("member %-6s %s : rmse %.4f (fold sd %.4f)" % (nm, kind, r, sd))
    IDX = {k: fold_index(lab_t, k) for k in ("A", "B")}

    for nm in ("Ridge", "Nys"):
        print("\n== LGB huber + %s : weight sweep ==" % nm)
        print("  %5s %8s %9s | %8s %9s" % ("w", "A rmse", "dA", "B rmse", "dB"))
        ws = np.round(np.arange(0.0, 0.51, 0.05), 2)
        rA, rB = [], []
        for w in ws:
            row = []
            for kind, acc in (("A", rA), ("B", rB)):
                g = ~np.isnan(O[("LGB", kind)])
                p = (1 - w) * O[("LGB", kind)][g] + w * O[(nm, kind)][g]
                acc.append(rmse(p, y[g]))
                row.append(acc[-1])
            print("  %5.2f %8.4f %+9.4f | %8.4f %+9.4f"
                  % (w, row[0], row[0] - rA[0], row[1], row[1] - rB[0]))
        wa, wb = ws[int(np.argmin(rA))], ws[int(np.argmin(rB))]
        print("  argmin: A w=%.2f (%.4f) | B w=%.2f (%.4f)"
              % (wa, min(rA), wb, min(rB)))
        for w in sorted({0.25, float(wa), float(wb)}):
            print("\n-- paired: LGB vs LGB+%s w=%.2f --" % (nm, w))
            for kind in ("A", "B"):
                print(" placement %s" % kind)
                alt = (1 - w) * O[("LGB", kind)] + w * O[(nm, kind)]
                report(lab_t, "sub_temp", O[("LGB", kind)], alt, IDX[kind],
                       "LGB+%s w=%.2f" % (nm, w))

    print("\n== 3-way LGB + Ridge(wr) + Nystroem(wn) ==")
    print("  %5s %5s %8s %8s" % ("wr", "wn", "A rmse", "B rmse"))
    grid = []
    for wr in (0.0, 0.1, 0.2, 0.25, 0.3):
        for wn in (0.0, 0.1, 0.2, 0.25, 0.3):
            if wr + wn > 0.51:
                continue
            rr = {}
            for kind in ("A", "B"):
                g = ~np.isnan(O[("LGB", kind)])
                p = ((1 - wr - wn) * O[("LGB", kind)][g]
                     + wr * O[("Ridge", kind)][g] + wn * O[("Nys", kind)][g])
                rr[kind] = rmse(p, y[g])
            grid.append((wr, wn, rr["A"], rr["B"]))
            print("  %5.2f %5.2f %8.4f %8.4f" % (wr, wn, rr["A"], rr["B"]))
    bA = min(grid, key=lambda r: r[2])
    bB = min(grid, key=lambda r: r[3])
    print("  argmin A: wr=%.2f wn=%.2f | argmin B: wr=%.2f wn=%.2f"
          % (bA[0], bA[1], bB[0], bB[1]))
    for wr, wn in {(bA[0], bA[1]), (bB[0], bB[1])}:
        print("\n-- paired: LGB vs 3-way LGB %.2f / Ridge %.2f / Nys %.2f --"
              % (1 - wr - wn, wr, wn))
        for kind in ("A", "B"):
            print(" placement %s" % kind)
            alt = ((1 - wr - wn) * O[("LGB", kind)]
                   + wr * O[("Ridge", kind)] + wn * O[("Nys", kind)])
            report(lab_t, "sub_temp", O[("LGB", kind)], alt, IDX[kind], "3-way")


if __name__ == "__main__":
    main()
