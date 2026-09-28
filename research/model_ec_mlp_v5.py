# -*- coding: utf-8 -*-
"""sub_ec: put the MLP into the v5 (18-feature) space and blend it there.

The earlier MLP numbers were measured on the 68-column ec view, whose own
tree baseline is ~.365 -- the wrong yardstick.  The real reference is the
18-column v5 view, where ExtraTrees600/1 scores A .2877 / B .2588.  This
script re-measures the MLP there (imputer + standard scaler in the pipeline,
since an MLP is scale sensitive) and then sweeps blend weights against
ExtraTrees, with the paired tests the coordinator specified: per-fold deltas
plus a paired greenhouse-day block bootstrap.

A second boosting member is carried along: LGB tweedie(1.5) on v5 scored
A .2891 / B .2541, better than the LGB huber used in the earlier blend, so
the blend search uses whichever boosting partner is actually stronger.

Stages (argv):
  mlp    -- MLP configuration table on v5
  blend  -- weight sweeps + paired tests (needs local/ec_oof_members.npy)
  f14    -- re-confirm the chosen blend on v5 minus the 4 outside-weather cols

Run:  cd research && PYTHONPATH="" <python> -u model_ec_mlp_v5.py mlp blend
"""
import sys

import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import load, score, views, folds
from common import split_mask, rmse, OUT_COLS
from model_common import run_table, seg_stats
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)
OOF_CACHE = "local/ec_oof_members.npy"   # [ET600/1, LGBhuber, HGBsq] x [A, B]


def REF(s):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(random_state=s, **ET1))


def LGBHUB(s):
    return lgb.LGBMRegressor(random_state=s, objective="huber", **DET, **LGBP)


def LGBTW(s):
    return lgb.LGBMRegressor(random_state=s, objective="tweedie",
                             tweedie_variance_power=1.5, **DET, **LGBP)


def mlp(h, alpha, early=True, it=800):
    def f(s):
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=h, alpha=alpha,
                         learning_rate_init=1e-3, max_iter=it,
                         early_stopping=early, n_iter_no_change=25,
                         validation_fraction=0.12, random_state=s))
    return f


MLP_BEST = mlp((128, 64), 1e-2)


def fold_index(lab, kind):
    return [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]


def report(lab, target, oof_ref, oof_alt, idx, name):
    y = lab[target].values.astype(float)
    d = [rmse(oof_alt[i], y[i]) - rmse(oof_ref[i], y[i]) for i in idx]
    nb = sum(1 for x in d if x < 0)
    pr, lo, hi, pw = paired_block_boot(lab, target, oof_ref, oof_alt,
                                       n_boot=2000, seed=0, level="row")
    dr, dlo, dhi, dpw = paired_block_boot(lab, target, oof_ref, oof_alt,
                                          n_boot=2000, seed=0, level="day")
    g = ~np.isnan(oof_alt) & ~np.isnan(oof_ref)
    s0, s1 = seg_stats(y[g], oof_ref[g]), seg_stats(y[g], oof_alt[g])
    print("  %s" % name)
    print("    per-fold delta : %s   (better in %d/5)"
          % (" ".join("%+.4f" % x for x in d), nb))
    print("    row  %+.4f  CI [%+.4f, %+.4f]  P(worse)=%.3f" % (pr, lo, hi, pw))
    print("    day  %+.4f  CI [%+.4f, %+.4f]  P(worse)=%.3f"
          % (dr, dlo, dhi, dpw))
    print("    y>1  rmse %.4f -> %.4f | bias %+.4f -> %+.4f"
          % (s0["rmse"], s1["rmse"], s0["bias"], s1["bias"]))


def sweep2(name, y, Oa, Ob, ref_a, ref_b):
    """1-D weight sweep of ref*(1-w) + partner*w on both placements."""
    print("\n== ET600/1 + %s : weight sweep ==" % name)
    print("  %5s %8s %9s | %8s %9s | %8s %9s"
          % ("w", "A rmse", "dA", "B rmse", "dB", "A hiRMSE", "A hiBias"))
    ws = np.round(np.arange(0.0, 0.66, 0.05), 2)
    rA, rB = [], []
    for w in ws:
        ga, gb = ~np.isnan(ref_a), ~np.isnan(ref_b)
        pa = (1 - w) * ref_a[ga] + w * Oa[ga]
        pb = (1 - w) * ref_b[gb] + w * Ob[gb]
        a, b = rmse(pa, y[ga]), rmse(pb, y[gb])
        rA.append(a)
        rB.append(b)
        s = seg_stats(y[ga], pa)
        print("  %5.2f %8.4f %+9.4f | %8.4f %+9.4f | %8.4f %+9.4f"
              % (w, a, a - rA[0], b, b - rB[0], s["rmse"], s["bias"]))
    wa, wb = ws[int(np.argmin(rA))], ws[int(np.argmin(rB))]
    print("  argmin: A w=%.2f (%.4f) | B w=%.2f (%.4f)"
          % (wa, min(rA), wb, min(rB)))
    return float(wa), float(wb)


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    v5 = v["v5"]
    y = lab_e["sub_ec"].values.astype(float)
    stages = sys.argv[1:] or ["mlp", "blend"]

    if "mlp" in stages:
        cands = [
            ("[T] v5 ET600/1 (reference)", v5, REF),
            ("[XS] v5 MLP 64-32 a=1e-3", v5, mlp((64, 32), 1e-3)),
            ("[XS] v5 MLP 128-64 a=1e-2", v5, mlp((128, 64), 1e-2)),
            ("[XS] v5 MLP 128-64 a=1e-3", v5, mlp((128, 64), 1e-3)),
            ("[XS] v5 MLP 256-128 a=1e-2", v5, mlp((256, 128), 1e-2)),
            ("[XS] v5 MLP 64-64-64 a=1e-3", v5, mlp((64, 64, 64), 1e-3)),
            ("[XS] v5 MLP 128-64 noES it300", v5,
             mlp((128, 64), 1e-2, early=False, it=300)),
        ]
        run_table(lab_e, "sub_ec", cands, segment=True, tag="[v5 MLP]")

    if "blend" in stages:
        cache = np.load(OOF_CACHE)
        O = {("ET", "A"): cache[0, 0], ("ET", "B"): cache[0, 1],
             ("LGBhub", "A"): cache[1, 0], ("LGBhub", "B"): cache[1, 1]}
        for nm, fac in [("MLP", MLP_BEST), ("LGBtw", LGBTW)]:
            for kind in ("A", "B"):
                (r, sd, per), oof = score(lab_e, "sub_ec", v5, fac, kind=kind,
                                          return_oof=True)
                O[(nm, kind)] = oof
                g = ~np.isnan(oof)
                print("member %-7s %s : rmse %.4f  hi-bias %+.4f"
                      % (nm, kind, r, seg_stats(y[g], oof[g])["bias"]))
        IDX = {k: fold_index(lab_e, k) for k in ("A", "B")}

        for nm in ("MLP", "LGBtw", "LGBhub"):
            wa, wb = sweep2(nm, y, O[(nm, "A")], O[(nm, "B")],
                            O[("ET", "A")], O[("ET", "B")])
            for w in sorted({round(0.5 * (wa + wb), 2), wa, wb}):
                print("\n-- paired: ET600/1 vs ET+%s w=%.2f --" % (nm, w))
                for kind in ("A", "B"):
                    print(" placement %s" % kind)
                    alt = (1 - w) * O[("ET", kind)] + w * O[(nm, kind)]
                    report(lab_e, "sub_ec", O[("ET", kind)], alt, IDX[kind],
                           "ET+%s w=%.2f" % (nm, w))

        # ---- 3-way: ExtraTrees + one boosting + the MLP -------------------
        print("\n== 3-way ET + LGBtweedie(wl) + MLP(wm) ==")
        print("  %5s %5s %8s %8s %8s %9s"
              % ("wl", "wm", "A rmse", "B rmse", "A hiRMSE", "A hiBias"))
        grid = []
        for wl in (0.0, 0.1, 0.2, 0.3, 0.4):
            for wm in (0.0, 0.1, 0.15, 0.2, 0.3):
                if wl + wm > 0.6:
                    continue
                rr = {}
                for kind in ("A", "B"):
                    g = ~np.isnan(O[("ET", kind)])
                    p = ((1 - wl - wm) * O[("ET", kind)][g]
                         + wl * O[("LGBtw", kind)][g] + wm * O[("MLP", kind)][g])
                    rr[kind] = rmse(p, y[g])
                ga = ~np.isnan(O[("ET", "A")])
                pa = ((1 - wl - wm) * O[("ET", "A")][ga]
                      + wl * O[("LGBtw", "A")][ga] + wm * O[("MLP", "A")][ga])
                s = seg_stats(y[ga], pa)
                grid.append((wl, wm, rr["A"], rr["B"]))
                print("  %5.2f %5.2f %8.4f %8.4f %8.4f %+9.4f"
                      % (wl, wm, rr["A"], rr["B"], s["rmse"], s["bias"]))
        bA = min(grid, key=lambda r: r[2])
        bB = min(grid, key=lambda r: r[3])
        print("  argmin A: wl=%.2f wm=%.2f | argmin B: wl=%.2f wm=%.2f"
              % (bA[0], bA[1], bB[0], bB[1]))
        for wl, wm in {(bA[0], bA[1]), (bB[0], bB[1]), (0.3, 0.1)}:
            print("\n-- paired: ET600/1 vs 3-way ET %.2f / LGBtw %.2f / MLP %.2f --"
                  % (1 - wl - wm, wl, wm))
            for kind in ("A", "B"):
                print(" placement %s" % kind)
                alt = ((1 - wl - wm) * O[("ET", kind)]
                       + wl * O[("LGBtw", kind)] + wm * O[("MLP", kind)])
                report(lab_e, "sub_ec", O[("ET", kind)], alt, IDX[kind], "3-way")
                two = 0.7 * O[("ET", kind)] + 0.3 * O[("LGBtw", kind)]
                report(lab_e, "sub_ec", two, alt, IDX[kind],
                       "   ...vs 2-way ET+LGBtw w=0.30")
        np.save("local/ec_oof_mlp_lgbtw.npy",
                np.array([[O[(n, k)] for k in ("A", "B")]
                          for n in ("MLP", "LGBtw")]))

    if "f14" in stages:
        # v5 minus the four outside-weather columns, as the feature agent found
        f14 = [c for c in v5 if c not in OUT_COLS]
        print("\n== re-confirmation on %d features (v5 minus %s) =="
              % (len(f14), ",".join(OUT_COLS)))
        Of = {}
        for nm, fac in [("ET", REF), ("LGBtw", LGBTW), ("MLP", MLP_BEST)]:
            for kind in ("A", "B"):
                (r, sd, per), oof = score(lab_e, "sub_ec", f14, fac, kind=kind,
                                          return_oof=True)
                Of[(nm, kind)] = oof
                g = ~np.isnan(oof)
                print("  %-6s %s rmse %.4f  hi-bias %+.4f"
                      % (nm, kind, r, seg_stats(y[g], oof[g])["bias"]))
        IDX = {k: fold_index(lab_e, k) for k in ("A", "B")}
        print("\n  %5s %5s %8s %8s" % ("wl", "wm", "A rmse", "B rmse"))
        for wl in (0.0, 0.2, 0.3, 0.4, 0.5):
            for wm in (0.0, 0.1, 0.15, 0.2, 0.3):
                if wl + wm > 0.61:
                    continue
                rr = {}
                for kind in ("A", "B"):
                    g = ~np.isnan(Of[("ET", kind)])
                    p = ((1 - wl - wm) * Of[("ET", kind)][g]
                         + wl * Of[("LGBtw", kind)][g] + wm * Of[("MLP", kind)][g])
                    rr[kind] = rmse(p, y[g])
                print("  %5.2f %5.2f %8.4f %8.4f" % (wl, wm, rr["A"], rr["B"]))
        for wl, wm in ((0.40, 0.15), (0.30, 0.10)):
            print("\n-- f14 paired: ET vs 3-way ET %.2f / LGBtw %.2f / MLP %.2f --"
                  % (1 - wl - wm, wl, wm))
            for kind in ("A", "B"):
                print(" placement %s" % kind)
                alt = ((1 - wl - wm) * Of[("ET", kind)]
                       + wl * Of[("LGBtw", kind)] + wm * Of[("MLP", kind)])
                report(lab_e, "sub_ec", Of[("ET", kind)], alt, IDX[kind],
                       "3-way on 14 features")


if __name__ == "__main__":
    main()
