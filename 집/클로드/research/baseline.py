# -*- coding: utf-8 -*-
"""Baseline: re-score the SUBMITTED models in THIS environment.

The documented numbers were produced under Python 3.7.9 / sklearn 1.0.2 /
lightgbm 4.6.0.  Here we run 3.12 / 1.9.1 / 4.7.0, so the documented values
cannot be compared directly to anything measured now.  Everything later is
compared against the numbers this script prints.
"""
import time

import env  # noqa: F401  MUST be first project import (DLL path + sys.path)
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from harness import load, score, views, blend_factory


DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)
ET2 = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4)


def lgbf(p):
    return lambda s: lgb.LGBMRegressor(random_state=s, **DET, **p)


def etf(p):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   ExtraTreesRegressor(random_state=s, **p))


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    rows = []

    cands_t = [("submitted LGB-huber 98f", v["temp"], lgbf(T_HUB))]
    cands_e = [
        ("submitted blend(ET100/8, LGB-huber) 68f", v["ec"],
         blend_factory([etf(ET8), lgbf(E_HUB)])),
        ("ET100/8 alone 68f", v["ec"], etf(ET8)),
        ("LGB-huber alone 68f", v["ec"], lgbf(E_HUB)),
        ("v5 recipe ET300/2 18f", v["v5"], etf(ET2)),
    ]

    for target, lab, cands in [("sub_temp", lab_t, cands_t), ("sub_ec", lab_e, cands_e)]:
        print("\n===== %s =====" % target)
        print("  %-42s %8s %8s %8s" % ("candidate", "foldsA", "foldsB", "sec"))
        for nm, cols, fac in cands:
            t0 = time.time()
            ra, sa, _ = score(lab, target, cols, fac, kind="A")
            rb, sb, _ = score(lab, target, cols, fac, kind="B")
            dt = time.time() - t0
            print("  %-42s %8.4f %8.4f %8.1f   (fold std %.3f/%.3f, %d feат)"
                  .replace("feат", "feat") % (nm, ra, rb, dt, sa, sb, len(cols)))
            rows.append((target, nm, ra, rb))
    print("\nreference: real test scored sub_temp 0.7450 / sub_ec 0.2442")


if __name__ == "__main__":
    main()
