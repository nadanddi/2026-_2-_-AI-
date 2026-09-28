# -*- coding: utf-8 -*-
"""sub_temp: same capability-driven family survey, plus blends.

sub_temp is a smooth, roughly symmetric, physically low-pass-filtered target,
so the capability that should matter here is S (smooth surface) rather than
P (skew-aware likelihood).  The submitted model is a single LightGBM huber on
the 98-column temp view; it is re-measured in this same script.

Run:  cd research && PYTHONPATH="" <python> -u model_families_temp.py [group...]
Groups: base lin kern mlp blend   (default: base lin kern mlp)
"""
import sys

import env  # noqa: F401  MUST be first project import
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from model_common import load, views, run_table, blend_factory

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
ONE = (7,)


def lgbf(**p):
    q = dict(DET)
    q.update(p)
    return lambda s: lgb.LGBMRegressor(random_state=s, **q)


def sc(est_fn):
    def f(s):
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), est_fn(s))
    return f


def build(v):
    tv, v5 = v["temp"], v["v5"]
    G = {}
    G["base"] = [
        ("[T] submitted LGB huber 98f", tv, lgbf(**T_HUB)),
        ("[T] ExtraTrees 300/2 98f", tv,
         lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                 ExtraTreesRegressor(n_estimators=300,
                                                     min_samples_leaf=2,
                                                     max_features=1.0,
                                                     n_jobs=4, random_state=s))),
    ]
    G["lin"] = [
        ("[XS] Ridge a=10 98f", tv, sc(lambda s: Ridge(alpha=10.0)), ONE),
        ("[XS] Ridge a=100 98f", tv, sc(lambda s: Ridge(alpha=100.0)), ONE),
        ("[XS] Ridge a=1000 98f", tv, sc(lambda s: Ridge(alpha=1000.0)), ONE),
        ("[XS] BayesianRidge 98f", tv, sc(lambda s: BayesianRidge()), ONE),
        ("[XS] Ridge a=100 18f", v5, sc(lambda s: Ridge(alpha=100.0)), ONE),
    ]

    def nys(gamma, ncomp, alpha):
        return sc(lambda s: make_pipeline(
            Nystroem(gamma=gamma, n_components=ncomp, random_state=s),
            Ridge(alpha=alpha)))

    G["kern"] = [
        ("[S] Nystroem g=.005 n500 + Ridge1", tv, nys(0.005, 500, 1.0)),
        ("[S] Nystroem g=.02 n500 + Ridge.1", tv, nys(0.02, 500, 0.1)),
        ("[S] SVR rbf C=10 eps=.1", tv,
         sc(lambda s: SVR(C=10.0, epsilon=0.1, gamma="scale", cache_size=700)), ONE),
    ]

    def mlp(h, alpha, it=500):
        return sc(lambda s: MLPRegressor(hidden_layer_sizes=h, alpha=alpha,
                                         learning_rate_init=1e-3, max_iter=it,
                                         early_stopping=True, n_iter_no_change=25,
                                         validation_fraction=0.12,
                                         random_state=s))
    G["mlp"] = [
        ("[XS] MLP 64-32 a=1e-3 98f", tv, mlp((64, 32), 1e-3)),
        ("[XS] MLP 128-64 a=1e-2 98f", tv, mlp((128, 64), 1e-2)),
        ("[XS] MLP 256-128 a=1e-1 98f", tv, mlp((256, 128), 1e-1)),
    ]
    G["blend"] = [
        ("[T] submitted LGB huber 98f", tv, lgbf(**T_HUB)),
        ("[T+XS] 75 LGB / 25 MLP128", tv,
         blend_factory([lgbf(**T_HUB), mlp((128, 64), 1e-2)], [0.75, 0.25])),
        ("[T+XS] 50 LGB / 50 MLP128", tv,
         blend_factory([lgbf(**T_HUB), mlp((128, 64), 1e-2)])),
        ("[T+S] 75 LGB / 25 Nystroem", tv,
         blend_factory([lgbf(**T_HUB), nys(0.005, 500, 1.0)], [0.75, 0.25])),
        ("[T+XS] 75 LGB / 25 Ridge100", tv,
         blend_factory([lgbf(**T_HUB), sc(lambda s: Ridge(alpha=100.0))],
                       [0.75, 0.25])),
    ]
    return G


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    G = build(v)
    want = sys.argv[1:] or ["base", "lin", "kern", "mlp"]
    for g in want:
        run_table(lab_t, "sub_temp", G[g], segment=False, tag="[%s]" % g)


if __name__ == "__main__":
    main()
