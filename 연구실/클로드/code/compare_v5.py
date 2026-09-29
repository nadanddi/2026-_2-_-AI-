# -*- coding: utf-8 -*-
"""Apples-to-apples: the GitHub v5 EC recipe vs the current EC model.

v5 recipe (from analysis/ec_removal_followup.py + train_all_farms_cv.py):
  features = 14 raw inputs + day + hour_sin + hour_cos + midnight
             (+ presence flags, which are constant on F13/F47 and were dropped
             by its own notna() filter; farm_code removed in v5)
  model    = median-impute -> ExtraTrees(300 trees, max_features=1, leaf=2)
  CV       = 10-day blocks assigned round-robin to 5 folds, 1-day embargo,
             exact-weather-day purge (skipped here; it removes few rows)

Both recipes are scored on BOTH validation schemes, on the whole period and
on the late period the test set occupies.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import make_folds, split_mask, rmse, TARGET_FARMS, USABLE
from model_v2 import get_panel, N_FOLDS
from submit_v2 import slim

SEEDS = (7, 101, 2024)


def et(seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(n_estimators=300, max_features=1.0,
                                             min_samples_leaf=2, random_state=seed,
                                             n_jobs=4))


def prior_folds(lab):
    """Round-robin 10-day blocks -> 5 folds, per farm (train_all_farms_cv.folds)."""
    start = lab.groupby("farm").day.transform("min")
    return ((lab.day - start) // 10 % 5).astype(int).values


def oof_scheme(lab, fcols, folds_iter):
    oof = np.full(len(lab), np.nan)
    for trm, vam in folds_iter:
        tr, va = lab[trm], lab[vam]
        ps = [et(sd).fit(tr[fcols], tr.sub_ec).predict(va[fcols]) for sd in SEEDS]
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    return oof


def my_scheme(lab, days, seed):
    for fd in make_folds(days, n_folds=N_FOLDS, seed=seed):
        trm, vam = split_mask(lab, fd)
        if vam.any():
            yield trm, vam


def their_scheme(lab):
    a = prior_folds(lab)
    for k in range(5):
        vam = a == k
        near = set()
        for farm in TARGET_FARMS:
            vd = lab.day.values[vam & (lab.farm == farm).values]
            near |= {(farm, d + o) for d in vd for o in (-1, 0, 1)}
        key = list(zip(lab.farm, lab.day))
        buf = np.array([k_ in near for k_ in key])
        yield ~buf, vam


def report(name, lab, oof):
    y = lab.sub_ec.values
    got = ~np.isnan(oof)
    late = got & (lab.day.values >= 183)
    return "%-28s all %.4f | late %.4f" % (name, rmse(oof[got], y[got]),
                                             rmse(oof[late], y[late]))


def main():
    panel, tX, ty, sX = get_panel()
    lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
    lab["midnight"] = (lab.hour == 0).astype(float)
    days = {f: sorted(lab[lab.farm == f].day.unique()) for f in TARGET_FARMS}

    v5_cols = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    cur_cols = slim(panel, "sub_ec")
    print("v5 recipe: %d features | current: %d features\n" % (len(v5_cols), len(cur_cols)))

    print("=== scheme A: current block CV (5/10/10/5-day blocks, 1-day buffer), 3 partitions ===")
    for name, cols in [("GitHub v5 recipe", v5_cols), ("current (slim + ET)", cur_cols)]:
        alls, lates = [], []
        for s in (0, 1, 2):
            oof = oof_scheme(lab, cols, my_scheme(lab, days, s))
            y = lab.sub_ec.values; got = ~np.isnan(oof); late = got & (lab.day.values >= 183)
            alls.append(rmse(oof[got], y[got])); lates.append(rmse(oof[late], y[late]))
        print("  %-28s all %.4f | late %.4f +-%.3f" % (name, np.mean(alls), np.mean(lates), np.std(lates)))

    print("\n=== scheme B: GitHub v5's own CV (10-day round-robin 5-fold, 1-day embargo) ===")
    for name, cols in [("GitHub v5 recipe", v5_cols), ("current (slim + ET)", cur_cols)]:
        oof = oof_scheme(lab, cols, their_scheme(lab))
        print("  " + report(name, lab, oof))
    print("\n(GitHub README reports v5 = 0.2654 on scheme B, single seed, with weather purge)")


if __name__ == "__main__":
    main()
