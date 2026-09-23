# -*- coding: utf-8 -*-
"""Can the EC ExtraTrees be made smaller without losing CV score?"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import make_folds, split_mask, rmse, TARGET_FARMS
from model_v2 import get_panel, N_FOLDS
from submit_v2 import slim

FOLD_SEEDS = [0, 1, 2, 3, 4]
SEEDS = (7, 101, 2024)


def run(lab, days, n_trees, leaf):
    lates = []
    for fs in FOLD_SEEDS:
        oof = np.full(len(lab), np.nan)
        for fd in make_folds(days, n_folds=N_FOLDS, seed=fs):
            trm, vam = split_mask(lab, fd)
            if not vam.any():
                continue
            tr, va = lab[trm], lab[vam]
            ps = []
            for sd in SEEDS:
                m = make_pipeline(SimpleImputer(strategy="median"),
                                  ExtraTreesRegressor(n_estimators=n_trees, max_features=1.0,
                                                      min_samples_leaf=leaf, random_state=sd,
                                                      n_jobs=4))
                ps.append(m.fit(tr[VE], tr.sub_ec).predict(va[VE]))
            oof[np.where(vam)[0]] = np.mean(ps, axis=0)
        got = ~np.isnan(oof); late = got & (lab.day.values >= 183)
        lates.append(rmse(oof[late], lab.sub_ec.values[late]))
    return np.mean(lates), np.std(lates)


panel, tX, ty, sX = get_panel()
VE = slim(panel, "sub_ec")
lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
days = {f: sorted(lab[lab.farm == f].day.unique()) for f in TARGET_FARMS}
for n, leaf in [(300, 2), (100, 2), (50, 2), (100, 4), (100, 8)]:
    m, s = run(lab, days, n, leaf)
    print("  trees=%3d x3 seeds  leaf=%d   late %.4f +-%.3f   (~%d MB uncompressed)"
          % (n, leaf, m, s, int(3 * n * 8200 * 64 / 1e6 * (2.0 / leaf) ** 0.8)))
