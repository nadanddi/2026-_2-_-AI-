# -*- coding: utf-8 -*-
"""CV that copies the real test layout, including its label-gap geometry.

The first submission scored 0.745 / 0.244 while the "late-period" CV said
0.887 / 0.296.  gap_check.py showed why: late held-out days sat 11 days from
the nearest label on average, the real test blocks only 3.6.  Here the exact
test layout  E5 . T8 . E10 . T8 . E10 . T3 . E5  is shifted into the dense
part of the record, so held-out days have labels ~2 days away on both sides,
exactly like the test.  Five shifts -> five folds; each trains on everything
else (1-day buffer around each held-out day, as always).

Candidates re-scored here, because the v5 verdict may flip under the right
geometry: v5 is a near-neighbour-in-time model that collapsed only when the
gap was 10 days.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import split_mask, rmse, TARGET_FARMS, USABLE
from model_v2 import get_panel
from make_submission import T_HUB, E_HUB, ET, SEEDS, DOMAIN_MARKS
import features_v2 as F2

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
LAYOUT = list(range(0, 5)) + list(range(15, 25)) + list(range(35, 45)) + list(range(50, 55))
SHIFTS = [70, 85, 100, 113, 126]        # last block ends at 126+54 = 180 (F47 dense run ends 181)


def geometry_folds():
    return [{f: {s + o for o in LAYOUT} for f in TARGET_FARMS} for s in SHIFTS]


def gap_report(lab, folds):
    g = []
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        for f, vd in fd.items():
            tr = np.array(sorted(lab[trm & (lab.farm == f).values].day.unique()))
            g += [np.abs(tr - d).min() for d in vd]
    g = np.array(g)
    print("held-out day -> nearest labelled day: mean %.1f median %.0f max %d  (real test: 3.6 / 3 / 6)"
          % (g.mean(), np.median(g), g.max()))


def lgb_fp(cols, params):
    def fp(tr, va, target):
        return np.mean([lgb.LGBMRegressor(random_state=s, **DET, **params)
                        .fit(tr[cols], tr[target]).predict(va[cols]) for s in SEEDS], axis=0)
    return fp


def et_fp(cols, et_params):
    def fp(tr, va, target):
        return np.mean([make_pipeline(SimpleImputer(strategy="median"),
                                      ExtraTreesRegressor(random_state=s, **et_params))
                        .fit(tr[cols], tr[target]).predict(va[cols]) for s in SEEDS], axis=0)
    return fp


def blend(*fps):
    def fp(tr, va, target):
        return np.mean([f(tr, va, target) for f in fps], axis=0)
    return fp


def score(lab, folds, target, fp):
    oof = np.full(len(lab), np.nan)
    per = []
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        p = fp(lab[trm], lab[vam], target)
        oof[np.where(vam)[0]] = p
        per.append(rmse(p, lab[vam][target].values))
    got = ~np.isnan(oof)
    return rmse(oof[got], lab[target].values[got]), np.std(per)


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    v_temp = F2.view(panel, "sub_temp")
    v_ec = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    folds = geometry_folds()

    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
    gap_report(lab_e, folds)
    print("held-out rows per fold: %d" % int(split_mask(lab_e, folds[0])[1].sum()))

    print("\n===== sub_temp (real test scored 0.7450) =====")
    r = score(lab_t, folds, "sub_temp", lgb_fp(v_temp, T_HUB))
    print("  %-40s %.4f  (fold std %.3f)" % ("submitted: huber, 98 feat, x3", *r))

    print("\n===== sub_ec (real test scored 0.2442) =====")
    cur_et = et_fp(v_ec, ET)
    cur_lgb = lgb_fp(v_ec, E_HUB)
    v5_et = et_fp(v5, dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4))
    cands = [
        ("submitted: blend(ET100/8, LGB-huber), 68f", blend(cur_et, cur_lgb)),
        ("current ET100/8 alone, 68f", cur_et),
        ("current LGB-huber alone, 68f", cur_lgb),
        ("GitHub v5 recipe: ET300/2, 18f", v5_et),
        ("blend(submitted, v5)", blend(blend(cur_et, cur_lgb), v5_et)),
        ("blend(current ET, v5)", blend(cur_et, v5_et)),
        ("blend(current ET, LGB, v5) equal", blend(cur_et, cur_lgb, v5_et)),
    ]
    for nm, fp in cands:
        r = score(lab_e, folds, "sub_ec", fp)
        print("  %-40s %.4f  (fold std %.3f)" % (nm, *r))


if __name__ == "__main__":
    main()
