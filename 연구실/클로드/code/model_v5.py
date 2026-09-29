# -*- coding: utf-8 -*-
"""EC round driven by the earlier repo's findings and the high-EC diagnosis.

Diagnosis on the current model: EC>1 rows are 8% of rows, 63% of squared
error, and are under-predicted by 0.54 dS/m on average.  Huber, which won the
last round on overall late RMSE, is exactly the kind of loss that shrinks
those large residuals -- so the trade-off has to be measured, not assumed.

Candidates carried over from the earlier work (its 5-fold CV, F13/F47):
  * dropping the greenhouse id from the EC model   (0.2998 -> 0.2654 there)
  * ExtraTrees(min_samples_leaf=2) instead of GBM   (0.3139 -> 0.2998 there)
  * a lean raw feature set (14 inputs + calendar)  (won for EC in its v1)
New here:
  * training weights that grow with EC level, so the regime that owns the
    error owns the gradient too (train labels only; nothing leaks).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import make_folds, split_mask, rmse, TARGET_FARMS, USABLE
from model_v2 import get_panel, N_FOLDS
from model_v4 import E_HUB, E_L2, MODEL_SEEDS
from submit_v2 import slim

FOLD_SEEDS = [0, 1, 2, 3, 4]
LEAN = USABLE + ["day", "hour", "hr_sin", "hr_cos", "day_par"]


def make_model(kind, params, seed):
    if kind == "lgb":
        return lgb.LGBMRegressor(random_state=seed, n_jobs=4, verbose=-1, **params)
    if kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=300, max_features=1.0,
                                min_samples_leaf=params.get("leaf", 2),
                                random_state=seed, n_jobs=4))
    raise ValueError(kind)


def run(panel, folds, fcols, kind="lgb", params=None, seeds=MODEL_SEEDS,
        recency_tau=None, y_power=None):
    lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if not len(va):
            continue
        w = np.ones(len(tr))
        if recency_tau:
            w *= np.exp(-(lab.day.max() - tr.day.values) / float(recency_tau))
        if y_power:
            w *= (tr.sub_ec.values / tr.sub_ec.mean()) ** y_power
        ps = []
        for sd in seeds:
            m = make_model(kind, params or {}, sd)
            if kind == "lgb":
                m.fit(tr[fcols], tr.sub_ec, sample_weight=w)
            else:
                m.fit(tr[fcols], tr.sub_ec, extratreesregressor__sample_weight=w)
            ps.append(m.predict(va[fcols]))
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    return lab, oof


def evaluate(panel, days, cfgs, name):
    """cfgs: list of run-kwargs blended inside each partition."""
    alls, lates, hi_r, hi_b = [], [], [], []
    for fs in FOLD_SEEDS:
        folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
        ps = []
        for kw in cfgs:
            lab, oof = run(panel, folds, **kw)
            ps.append(oof)
        oof = np.mean(ps, axis=0)
        y = lab.sub_ec.values
        got = ~np.isnan(oof)
        late = got & (lab.day.values >= 183)
        hi = got & (y > 1.0)
        alls.append(rmse(oof[got], y[got]))
        lates.append(rmse(oof[late], y[late]))
        hi_r.append(rmse(oof[hi], y[hi]))
        hi_b.append((oof - y)[hi].mean())
    print("  %-34s all %.4f | late %.4f +-%.3f | EC>1 rmse %.3f bias %+.3f"
          % (name, np.mean(alls), np.mean(lates), np.std(lates),
             np.mean(hi_r), np.mean(hi_b)))
    return np.mean(lates)


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    ve = slim(panel, "sub_ec")
    ve_nofarm = [c for c in ve if c != "farm_id"]
    lean = [c for c in LEAN if c in panel.columns]
    lean_nofarm = lean  # LEAN never had farm_id
    print("views: slim %d | slim-nofarm %d | lean %d" % (len(ve), len(ve_nofarm), len(lean)))

    print("\n===== reference =====")
    evaluate(panel, days, [dict(fcols=ve, params=E_HUB),
                           dict(fcols=ve, params=E_HUB, recency_tau=120)],
             "final v2: blend(huber, huber+rec120)")
    evaluate(panel, days, [dict(fcols=ve, params=E_L2)], "l2 (huber off)")

    print("\n===== from the earlier repo =====")
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_HUB)], "huber, no farm_id")
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_L2)], "l2, no farm_id")
    evaluate(panel, days, [dict(fcols=lean, params=E_L2)], "l2, lean 19 features")
    evaluate(panel, days, [dict(fcols=ve, kind="et", params=dict(leaf=2))],
             "ExtraTrees leaf2, slim")
    evaluate(panel, days, [dict(fcols=ve_nofarm, kind="et", params=dict(leaf=2))],
             "ExtraTrees leaf2, slim no farm_id")
    evaluate(panel, days, [dict(fcols=lean, kind="et", params=dict(leaf=2))],
             "ExtraTrees leaf2, lean")

    print("\n===== high-EC emphasis =====")
    for pw in (0.5, 1.0):
        evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_L2, y_power=pw)],
                 "l2, no farm_id, weight y^%.1f" % pw)
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_HUB, y_power=1.0)],
             "huber, no farm_id, weight y^1")
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_L2, y_power=0.5,
                                recency_tau=120)],
             "l2, no farm_id, y^.5 + rec120")

    print("\n===== blends =====")
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_HUB),
                           dict(fcols=ve_nofarm, kind="et", params=dict(leaf=2))],
             "blend lgb-huber / ET (no farm_id)")
    evaluate(panel, days, [dict(fcols=ve_nofarm, params=E_L2, y_power=0.5),
                           dict(fcols=ve_nofarm, kind="et", params=dict(leaf=2))],
             "blend lgb-l2-y^.5 / ET (no farm_id)")


if __name__ == "__main__":
    main()
