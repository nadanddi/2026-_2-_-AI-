# -*- coding: utf-8 -*-
"""Focused EC tuning + temp confirmation, all on 3 independent block partitions.

The day-level EC model was dropped after it lost decisively (late .3946 vs
.3215): holding one constant per day means using only information settled at
hour 00, and same-day climate turns out to carry real signal about that day's
EC level.

What is tested here:
  * regularisation strength around the row-level EC model;
  * huber vs l2 for EC (huber clearly helps sub_temp);
  * recency weighting -- the EC regime drifts upward all season and the test
    blocks sit in the late period, so training days close to them should count
    more.  Uses only the day index, so nothing leaks;
  * small ensembles of the survivors.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import make_folds, split_mask, rmse, TARGET_FARMS
from model_v2 import get_panel, score, N_FOLDS
import features_v2 as F2

FOLD_SEEDS = [0, 1, 2]


def run(panel, folds, target, fcols, params, seeds=(7,), log_target=False,
        recency_tau=None):
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if not len(va):
            continue
        w = None
        if recency_tau:
            # weight training days by closeness to the end of the record
            dmax = lab.day.max()
            w = np.exp(-(dmax - tr.day.values) / float(recency_tau))
        ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1,
                                  **params)
            y = np.log(tr[target]) if log_target else tr[target]
            m.fit(tr[fcols], y, sample_weight=w)
            p = m.predict(va[fcols])
            ps.append(np.exp(p) if log_target else p)
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    return lab, oof


def multi(panel, days, target, **kw):
    alls, lates, oofs = [], [], []
    for fs in FOLD_SEEDS:
        folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
        lab, oof = run(panel, folds, target, **kw)
        a, l = score(lab, oof, target)
        alls.append(a); lates.append(l); oofs.append(oof)
    return np.mean(alls), np.mean(lates), np.std(lates), lab, oofs


def multi_blend(panel, days, target, cfgs):
    """Average several configurations inside each partition, then score."""
    alls, lates = [], []
    for fs in FOLD_SEEDS:
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
    v_temp = F2.view(panel, "sub_temp")
    v_ec = F2.view(panel, "sub_ec")

    def rep(name, res):
        print("  %-40s all %.4f | late %.4f +-%.3f" % (name, res[0], res[1], res[2]))

    e_base = dict(objective="regression", n_estimators=400, learning_rate=0.02,
                  num_leaves=7, min_child_samples=240, subsample=0.7,
                  subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)

    print("===== sub_ec regularisation (ref: ec view .3215) =====")
    for nm, upd in [("mcs=120", dict(min_child_samples=120)),
                    ("mcs=360", dict(min_child_samples=360)),
                    ("colsample .25", dict(colsample_bytree=0.25)),
                    ("colsample .7", dict(colsample_bytree=0.7)),
                    ("leaves15 mcs240", dict(num_leaves=15)),
                    ("lr .01 x900", dict(learning_rate=0.01, n_estimators=900)),
                    ("huber", dict(objective="huber")),
                    ("lambda_l1=2", dict(reg_alpha=2.0))]:
        pr = dict(e_base); pr.update(upd)
        rep(nm, multi(panel, days, "sub_ec", fcols=v_ec, params=pr)[:3])

    print("\n===== sub_ec recency weighting =====")
    for tau in (60, 120, 240):
        rep("tau=%d days" % tau,
            multi(panel, days, "sub_ec", fcols=v_ec, params=e_base,
                  recency_tau=tau)[:3])

    print("\n===== sub_temp recency weighting (ref: huber .8968) =====")
    t_hub = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
                 num_leaves=63, min_child_samples=40, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
    for tau in (120, 240):
        rep("tau=%d days" % tau,
            multi(panel, days, "sub_temp", fcols=v_temp, params=t_hub,
                  recency_tau=tau)[:3])
    t_l2 = dict(t_hub); t_l2["objective"] = "regression"
    print("\n===== sub_temp ensembles =====")
    rep("huber + l2", multi_blend(panel, days, "sub_temp",
                                  [dict(fcols=v_temp, params=t_hub),
                                   dict(fcols=v_temp, params=t_l2)]))
    rep("huber x3 seeds", multi(panel, days, "sub_temp", fcols=v_temp,
                                params=t_hub, seeds=(7, 101, 2024))[:3])


if __name__ == "__main__":
    main()
