# -*- coding: utf-8 -*-
"""Evaluate the curated v2 feature set with explicit overfitting controls.

Controls under test, in the order they matter:
  1. target-specific feature views (physics, decided before any tuning);
  2. regularisation expressed in DAYS rather than rows -- for sub_ec the rows
     inside a day are near-duplicates, so min_child_samples=60 was really
     "2.5 days" and barely constrained anything;
  3. a day-level EC model (one row per greenhouse-day) that matches the true
     effective sample size of ~400;
  4. leak-free importance pruning (importance is computed on fold-train only).

Everything is scored on the block CV, and reported both overall and on the
late period (day >= 183) that the real test set occupies.
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, make_folds, split_mask, rmse, TARGET_FARMS
import features_v2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "panel_v2.pkl")
N_FOLDS = 6


def get_panel():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    tX, ty, sX = load_raw()
    panel = F2.build(tX, sX)
    panel = panel.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id",
                        how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    pd.to_pickle((panel, tX, ty, sX), CACHE)
    return panel, tX, ty, sX


def score(lab, oof, target):
    got = ~np.isnan(oof)
    late = got & (lab.day.values >= 183)
    return (rmse(oof[got], lab[target].values[got]),
            rmse(oof[late], lab[target].values[late]))


def run_rows(panel, folds, target, fcols, params, seeds=(7,), top_k=None,
             log_target=False):
    """Row-level model.  top_k enables leak-free importance pruning."""
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if not len(va):
            continue
        use = fcols
        if top_k:
            probe = lgb.LGBMRegressor(random_state=1, n_jobs=4, verbose=-1,
                                      **params)
            probe.fit(tr[fcols], np.log(tr[target]) if log_target else tr[target])
            imp = pd.Series(probe.booster_.feature_importance("gain"),
                            index=fcols)
            use = list(imp.sort_values(ascending=False).head(top_k).index)
        ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1,
                                  **params)
            y = np.log(tr[target]) if log_target else tr[target]
            m.fit(tr[use], y)
            p = m.predict(va[use])
            ps.append(np.exp(p) if log_target else p)
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    return lab, oof


def day_table(panel, target):
    """One row per (farm, day): features as known at hour 00, target = day mean."""
    dv = [c for c in F2.day_view(panel) if c not in ("farm", "day")]
    p = panel.sort_values(["farm", "day", "hour"])
    first = p.groupby(["farm", "day"])[dv].first()
    y = p.groupby(["farm", "day"])[target].mean()
    n = p.groupby(["farm", "day"])[target].count()
    d = first.copy()
    d[target] = y
    d["_n"] = n
    # "day" (crop stage) comes back as a column via reset_index
    return d.reset_index(), dv + ["day"]


def run_day_level(panel, folds, target, params, seeds=(7,), residual=False,
                  res_params=None):
    """Predict a constant per greenhouse-day, optionally + an intraday residual."""
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    dtab, dv = day_table(panel[(~panel.is_test) & panel[target].notna()], target)
    dtab = dtab[dtab[target].notna()].reset_index(drop=True)

    res_cols = sorted(set(
        [c for c in panel.columns
         if c.endswith("_tdmean") or c.endswith("_tdsum") or c in F2.INSTANT]
        + ["hour", "hr_sin", "hr_cos", "farm_id"]))

    oof = np.full(len(lab), np.nan)
    for fd in folds:
        # day-level split
        dval = np.zeros(len(dtab), bool)
        dbuf = np.zeros(len(dtab), bool)
        for farm, ds in fd.items():
            isf = (dtab.farm == farm).values
            dval |= isf & np.isin(dtab.day.values, list(ds))
            near = set()
            for x in ds:
                near.update([x - 1, x, x + 1])
            dbuf |= isf & np.isin(dtab.day.values, list(near))
        dtr = dtab[~dbuf]
        ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1,
                                  **params)
            m.fit(dtr[dv], dtr[target])
            ps.append(m.predict(dtab.loc[dval, dv]))
        dpred = pd.DataFrame({"farm": dtab.loc[dval, "farm"].values,
                              "day": dtab.loc[dval, "day"].values,
                              "dp": np.mean(ps, axis=0)})

        trm, vam = split_mask(lab, fd)
        va = lab[vam].merge(dpred, on=["farm", "day"], how="left")
        pred = va.dp.values

        if residual:
            tr = lab[trm].copy()
            dm = tr.groupby(["farm", "day"])[target].transform("mean")
            tr["_res"] = tr[target] - dm
            rm = lgb.LGBMRegressor(random_state=7, n_jobs=4, verbose=-1,
                                   **(res_params or params))
            rm.fit(tr[res_cols], tr["_res"])
            pred = pred + rm.predict(lab[vam][res_cols])

        oof[np.where(vam)[0]] = pred
    return lab, oof


FOLD_SEEDS = [0, 1, 2]


def multi(panel, days, target, fn, **kw):
    """Average a candidate over several block partitions.

    The late-period slice is only ~49 greenhouse-days, so a single partition
    gives a noisy score (~+-0.03 for EC).  Averaging over three independent
    partitions is what makes the model choice defensible rather than a pick of
    the luckiest split.
    """
    alls, lates = [], []
    for fs in FOLD_SEEDS:
        folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
        lab, oof = fn(panel, folds, target, **kw)
        a, l = score(lab, oof, target)
        alls.append(a)
        lates.append(l)
    return np.mean(alls), np.mean(lates), np.std(lates)


def main_multi():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    print("panel v2: %d rows x %d cols" % panel.shape)
    all_cols = [c for c in panel.columns if c not in F2.META]
    v_temp = F2.view(panel, "sub_temp")
    v_ec = F2.view(panel, "sub_ec")
    print("cols: all %d | temp view %d | ec view %d | day view %d"
          % (len(all_cols), len(v_temp), len(v_ec),
             len(F2.day_view(panel))))

    def rep(name, a, l, s):
        print("  %-38s all %.4f | late %.4f +-%.3f" % (name, a, l, s))

    print("\n===== sub_temp (%d fold partitions) =====" % len(FOLD_SEEDS))
    t_l2 = dict(objective="regression", n_estimators=1200, learning_rate=0.03,
                num_leaves=63, min_child_samples=40, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
    t_hub = dict(t_l2); t_hub["objective"] = "huber"
    t_hub31 = dict(t_hub); t_hub31.update(num_leaves=31, min_child_samples=120)
    for nm, fc, pr in [("all cols, l2", all_cols, t_l2),
                       ("temp view, l2", v_temp, t_l2),
                       ("temp view, huber", v_temp, t_hub),
                       ("temp view, huber leaves31 mcs120", v_temp, t_hub31)]:
        rep(nm, *multi(panel, days, "sub_temp", run_rows, fcols=fc, params=pr))

    print("\n===== sub_ec (%d fold partitions) =====" % len(FOLD_SEEDS))
    e_row = dict(objective="regression", n_estimators=400, learning_rate=0.02,
                 num_leaves=7, min_child_samples=240, subsample=0.7,
                 subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
    day_p = dict(objective="regression", n_estimators=500, learning_rate=0.02,
                 num_leaves=7, min_child_samples=10, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0)
    res_p = dict(objective="regression", n_estimators=200, learning_rate=0.03,
                 num_leaves=7, min_child_samples=200, colsample_bytree=0.5,
                 reg_lambda=5.0)
    rep("all cols, row-level",
        *multi(panel, days, "sub_ec", run_rows, fcols=all_cols, params=e_row))
    rep("ec view, row-level",
        *multi(panel, days, "sub_ec", run_rows, fcols=v_ec, params=e_row))
    rep("ec view, row-level + log",
        *multi(panel, days, "sub_ec", run_rows, fcols=v_ec, params=e_row,
               log_target=True))
    rep("day-level constant",
        *multi(panel, days, "sub_ec", run_day_level, params=day_p))
    rep("day-level + intraday residual",
        *multi(panel, days, "sub_ec", run_day_level, params=day_p,
               residual=True, res_params=res_p))


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)
    print("panel v2: %d rows x %d cols" % panel.shape)

    v_temp = F2.view(panel, "sub_temp")
    v_ec = F2.view(panel, "sub_ec")
    print("view sizes -> sub_temp %d, sub_ec %d, day-level %d"
          % (len(v_temp), len(v_ec), len(F2.day_view(panel))))

    def rep(name, lab, oof, target):
        a, l = score(lab, oof, target)
        print("  %-42s all %.4f | late %.4f" % (name, a, l))
        return l

    # ---------------------------- sub_temp ---------------------------------
    print("\n===== sub_temp (v1 reference: all .7613 / late .9270) =====")
    base_t = dict(objective="regression", n_estimators=1200,
                  learning_rate=0.03, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_lambda=1.0)
    lab, oof = run_rows(panel, folds, "sub_temp", v_temp, base_t)
    rep("v2 view, v1 params", lab, oof, "sub_temp")
    for nm, upd in [("mcs=120 (5 days)", dict(min_child_samples=120)),
                    ("mcs=240 (10 days)", dict(min_child_samples=240)),
                    ("leaves31 mcs120", dict(num_leaves=31, min_child_samples=120)),
                    ("leaves31 mcs120 l1=1", dict(num_leaves=31,
                                                  min_child_samples=120,
                                                  reg_alpha=1.0)),
                    ("huber leaves31 mcs120", dict(objective="huber",
                                                   num_leaves=31,
                                                   min_child_samples=120))]:
        pr = dict(base_t); pr.update(upd)
        _, o = run_rows(panel, folds, "sub_temp", v_temp, pr)
        rep(nm, lab, o, "sub_temp")
    for k in (40, 80):
        pr = dict(base_t); pr.update(dict(num_leaves=31, min_child_samples=120))
        _, o = run_rows(panel, folds, "sub_temp", v_temp, pr, top_k=k)
        rep("prune top-%d" % k, lab, o, "sub_temp")

    # ---------------------------- sub_ec -----------------------------------
    print("\n===== sub_ec (v1 reference: all .2550 / late .3179) =====")
    base_e = dict(objective="regression", n_estimators=400,
                  learning_rate=0.02, num_leaves=15, min_child_samples=60,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
                  reg_lambda=5.0)
    labe, oe = run_rows(panel, folds, "sub_ec", v_ec, base_e)
    rep("v2 view, v1 params", labe, oe, "sub_ec")
    for nm, upd in [("mcs=240 (10 days)", dict(min_child_samples=240)),
                    ("mcs=480 (20 days)", dict(min_child_samples=480)),
                    ("leaves7 mcs240", dict(num_leaves=7, min_child_samples=240)),
                    ("leaves7 mcs240 l1=1", dict(num_leaves=7,
                                                 min_child_samples=240,
                                                 reg_alpha=1.0))]:
        pr = dict(base_e); pr.update(upd)
        _, o = run_rows(panel, folds, "sub_ec", v_ec, pr)
        rep(nm, labe, o, "sub_ec")
    for k in (20, 40):
        pr = dict(base_e); pr.update(dict(num_leaves=7, min_child_samples=240))
        _, o = run_rows(panel, folds, "sub_ec", v_ec, pr, top_k=k)
        rep("prune top-%d" % k, labe, o, "sub_ec")

    print("  -- day-level EC models (effective n ~ 400) --")
    day_p = dict(objective="regression", n_estimators=500, learning_rate=0.02,
                 num_leaves=7, min_child_samples=10, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0)
    _, o = run_day_level(panel, folds, "sub_ec", day_p)
    rep("day-level constant", labe, o, "sub_ec")
    for nm, upd in [("day-level mcs=20", dict(min_child_samples=20)),
                    ("day-level leaves3", dict(num_leaves=3)),
                    ("day-level lr.05 x300", dict(learning_rate=0.05,
                                                  n_estimators=300))]:
        pr = dict(day_p); pr.update(upd)
        _, o = run_day_level(panel, folds, "sub_ec", pr)
        rep(nm, labe, o, "sub_ec")
    res_p = dict(objective="regression", n_estimators=200, learning_rate=0.03,
                 num_leaves=7, min_child_samples=200, colsample_bytree=0.5,
                 reg_lambda=5.0)
    _, o = run_day_level(panel, folds, "sub_ec", day_p, residual=True,
                         res_params=res_p)
    rep("day-level + intraday residual", labe, o, "sub_ec")


if __name__ == "__main__":
    if "--single" in sys.argv:
        main()
    else:
        main_multi()
