# -*- coding: utf-8 -*-
"""CV score of exactly the submitted configuration (make_submission.py)."""
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import make_folds, split_mask, rmse, TARGET_FARMS
from model_v2 import get_panel, N_FOLDS
from make_submission import T_HUB, E_HUB, ET, SEEDS, DOMAIN_MARKS
import features_v2 as F2

FOLD_SEEDS = [0, 1, 2, 3, 4, 5, 6]
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)


def oof_for(lab, days, fit_predict):
    alls, lates = [], []
    for fs in FOLD_SEEDS:
        oof = np.full(len(lab), np.nan)
        for fd in make_folds(days, n_folds=N_FOLDS, seed=fs):
            trm, vam = split_mask(lab, fd)
            if vam.any():
                oof[np.where(vam)[0]] = fit_predict(lab[trm], lab[vam])
        y = lab[target].values; got = ~np.isnan(oof); late = got & (lab.day.values >= 183)
        alls.append(rmse(oof[got], y[got])); lates.append(rmse(oof[late], y[late]))
    return np.mean(alls), np.mean(lates), np.std(lates)


panel, tX, ty, sX = get_panel()
v_temp = F2.view(panel, "sub_temp")
v_ec = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]

for target in ["sub_temp", "sub_ec"]:
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    days = {f: sorted(lab[lab.farm == f].day.unique()) for f in TARGET_FARMS}
    if target == "sub_temp":
        def fp(tr, va):
            return np.mean([lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)
                            .fit(tr[v_temp], tr.sub_temp).predict(va[v_temp]) for s in SEEDS], axis=0)
    else:
        def fp(tr, va):
            et = np.mean([make_pipeline(SimpleImputer(strategy="median"),
                                        ExtraTreesRegressor(random_state=s, **ET))
                          .fit(tr[v_ec], tr.sub_ec).predict(va[v_ec]) for s in SEEDS], axis=0)
            gb = np.mean([lgb.LGBMRegressor(random_state=s, **DET, **E_HUB)
                          .fit(tr[v_ec], tr.sub_ec).predict(va[v_ec]) for s in SEEDS], axis=0)
            return 0.5 * (et + gb)
    a, l, s = oof_for(lab, days, fp)
    print("%-9s all %.4f | late %.4f +-%.3f  (%d partitions)" % (target, a, l, s, len(FOLD_SEEDS)))
