# -*- coding: utf-8 -*-
"""Finalists only, evaluated on 5 block partitions to cut selection noise.

Kept from the search, each for a reason that holds up physically:
  * huber loss -- train_X (and only train_X) contains statistically restored /
    perturbed values, so down-weighting large residuals is principled, not a
    tuning artefact.  It helped both targets independently.
  * recency weighting for sub_ec ONLY -- EC drifts all season with crop stage
    and the fertigation EC ramp, so early days are a different regime.  It did
    NOT help sub_temp, which has no such regime drift, and was dropped there.
  * seed averaging -- removes the +-0.01 run-to-run wobble.
"""
import numpy as np
import pandas as pd

from common import make_folds, rmse, TARGET_FARMS
from model_v2 import get_panel, score, N_FOLDS
from model_v3 import run
import features_v2 as F2

FOLD_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEEDS = (7, 101, 2024)

T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
T_L2 = dict(T_HUB, objective="regression")

E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
E_L2 = dict(E_HUB, objective="regression")


def blend_multi(panel, days, target, cfgs, fold_seeds=FOLD_SEEDS):
    alls, lates = [], []
    for fs in fold_seeds:
        folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
        ps = []
        for kw in cfgs:
            lab, oof = run(panel, folds, target, **kw)
            ps.append(oof)
        a, l = score(lab, np.mean(ps, axis=0), target)
        alls.append(a); lates.append(l)
    return np.mean(alls), np.mean(lates), np.std(lates)


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    vt = F2.view(panel, "sub_temp")
    ve = F2.view(panel, "sub_ec")

    def rep(n, r):
        print("  %-38s all %.4f | late %.4f +-%.3f" % (n, r[0], r[1], r[2]))

    print("===== sub_temp finalists (%d partitions) =====" % len(FOLD_SEEDS))
    rep("huber x3 seeds",
        blend_multi(panel, days, "sub_temp",
                    [dict(fcols=vt, params=T_HUB, seeds=MODEL_SEEDS)]))
    rep("huber + l2, x3 seeds",
        blend_multi(panel, days, "sub_temp",
                    [dict(fcols=vt, params=T_HUB, seeds=MODEL_SEEDS),
                     dict(fcols=vt, params=T_L2, seeds=MODEL_SEEDS)]))

    print("\n===== sub_ec finalists (%d partitions) =====" % len(FOLD_SEEDS))
    rep("huber x3 seeds",
        blend_multi(panel, days, "sub_ec",
                    [dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS)]))
    rep("huber + recency120, x3 seeds",
        blend_multi(panel, days, "sub_ec",
                    [dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS,
                          recency_tau=120)]))
    rep("blend huber / huber-recency120",
        blend_multi(panel, days, "sub_ec",
                    [dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS),
                     dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS,
                          recency_tau=120)]))
    rep("blend huber / l2 / log",
        blend_multi(panel, days, "sub_ec",
                    [dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS),
                     dict(fcols=ve, params=E_L2, seeds=MODEL_SEEDS),
                     dict(fcols=ve, params=E_HUB, seeds=MODEL_SEEDS,
                          log_target=True)]))


if __name__ == "__main__":
    main()
