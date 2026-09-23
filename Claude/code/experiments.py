# -*- coding: utf-8 -*-
"""Variant search on top of the block CV, mainly to lift the EC score."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import (build_panel, feature_columns, make_folds, split_mask,
                    rmse, TARGET_FARMS)
from cv import prepare, N_FOLDS

BASE = {
    "sub_temp": dict(learning_rate=0.03, num_leaves=63, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                     reg_lambda=1.0, n_estimators=1204),
    "sub_ec": dict(learning_rate=0.02, num_leaves=15, min_child_samples=60,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
                   reg_lambda=5.0, n_estimators=411),
}


def oof_predict(panel, folds, target, params, log_target=False,
                objective=None, drop_cols=()):
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    fcols = feature_columns(panel, extra_drop=["is_test", "day_mod7"] + list(drop_cols))
    oof = np.full(len(lab), np.nan)
    kw = dict(params)
    if objective:
        kw["objective"] = objective
    for fd in folds:
        fd = {f: d for f, d in fd.items() if f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if len(va) == 0:
            continue
        y = np.log(tr[target]) if log_target else tr[target]
        m = lgb.LGBMRegressor(random_state=7, n_jobs=4, verbose=-1, **kw)
        m.fit(tr[fcols], y)
        p = m.predict(va[fcols])
        if log_target:
            p = np.exp(p)
        oof[np.where(vam)[0]] = p
    return lab, oof


def causal_daymean(lab, pred):
    """Expanding mean of the prediction within the current day (causal)."""
    s = pd.DataFrame({"farm": lab.farm.values, "day": lab.day.values,
                      "hour": lab.hour.values, "p": pred})
    s = s.sort_values(["farm", "day", "hour"])
    s["sm"] = s.groupby(["farm", "day"]).p.transform(lambda x: x.expanding().mean())
    return s.sort_index().sm.values


def blend_prev_day(lab, pred, w):
    """Blend each hour's prediction with the previous day's full-day mean."""
    s = pd.DataFrame({"farm": lab.farm.values, "day": lab.day.values, "p": pred})
    dm = s.groupby(["farm", "day"]).p.mean().rename("pdm").reset_index()
    dm["day"] = dm.day + 1
    s = s.merge(dm, on=["farm", "day"], how="left")
    out = np.where(s.pdm.notna(), (1 - w) * s.p + w * s.pdm, s.p)
    return out


def report(name, lab, pred, target):
    got = ~np.isnan(pred)
    per = {f: rmse(pred[got & (lab.farm == f).values],
                   lab[target].values[got & (lab.farm == f).values])
           for f in TARGET_FARMS}
    r = rmse(pred[got], lab[target].values[got])
    print("  %-34s %.4f   (F13 %.4f  F47 %.4f)" % (name, r, per["F13"], per["F47"]))
    return r


def main():
    panel, tX, ty, sX = prepare(use_aux=True)
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)

    print("\n===== sub_ec variants =====")
    lab, p_base = oof_predict(panel, folds, "sub_ec", BASE["sub_ec"])
    best = ("base", report("base (l2)", lab, p_base, "sub_ec"), p_base)

    for nm, kw in [("huber", dict(objective="huber")),
                   ("log target", dict(log_target=True))]:
        _, p = oof_predict(panel, folds, "sub_ec", BASE["sub_ec"], **kw)
        r = report(nm, lab, p, "sub_ec")
        if r < best[1]:
            best = (nm, r, p)

    # stronger / weaker regularisation
    for nm, upd in [("leaves7 mcs100", dict(num_leaves=7, min_child_samples=100)),
                    ("leaves31 mcs30", dict(num_leaves=31, min_child_samples=30)),
                    ("colsample .2", dict(colsample_bytree=0.2)),
                    ("lr .01 x1200", dict(learning_rate=0.01, n_estimators=1200))]:
        pr = dict(BASE["sub_ec"]); pr.update(upd)
        _, p = oof_predict(panel, folds, "sub_ec", pr)
        r = report(nm, lab, p, "sub_ec")
        if r < best[1]:
            best = (nm, r, p)

    print("  -- post-processing on best raw model (%s) --" % best[0])
    pb = best[2]
    report("causal within-day expanding mean", lab, causal_daymean(lab, pb), "sub_ec")
    for w in (0.3, 0.5, 0.7):
        report("blend prev-day mean w=%.1f" % w, lab, blend_prev_day(lab, pb, w), "sub_ec")

    print("\n===== sub_temp variants =====")
    labt, t_base = oof_predict(panel, folds, "sub_temp", BASE["sub_temp"])
    report("base (l2, aux prior)", labt, t_base, "sub_temp")
    _, t_noaux = oof_predict(panel, folds, "sub_temp", BASE["sub_temp"],
                             drop_cols=["aux_temp_prior"])
    report("without aux prior", labt, t_noaux, "sub_temp")
    _, t_hub = oof_predict(panel, folds, "sub_temp", BASE["sub_temp"],
                           objective="huber")
    report("huber", labt, t_hub, "sub_temp")
    for nm, upd in [("leaves127", dict(num_leaves=127)),
                    ("leaves31 mcs60", dict(num_leaves=31, min_child_samples=60)),
                    ("lr .015 x2400", dict(learning_rate=0.015, n_estimators=2400))]:
        pr = dict(BASE["sub_temp"]); pr.update(upd)
        _, p = oof_predict(panel, folds, "sub_temp", pr)
        report(nm, labt, p, "sub_temp")
    report("avg(base, huber)", labt, 0.5 * (t_base + t_hub), "sub_temp")


if __name__ == "__main__":
    main()
