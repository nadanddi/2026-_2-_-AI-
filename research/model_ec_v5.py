# -*- coding: utf-8 -*-
"""sub_ec on the 18-column v5 view, which is the strong reference here.

baseline.py in this environment scores
    submitted blend(ET100/8, LGB-huber) on the 68-col ec view : A .3649 B .3372
    v5 recipe ExtraTrees300/2 on the 18-col v5 view           : A .2898 B .2615
so every family comparison for sub_ec has to be made in the v5 feature space,
not against the 68-column blend.  This script re-measures the v5 reference and
puts the non-tree / distribution-aware families beside it, with the y>1
segment broken out.

Capability legend (see model_families_ec.py):
  T tree (no extrapolation, piecewise constant, NaN-native for LGB)
  X extrapolates   S smooth surface   P skew-aware likelihood/link
  H histogram tree with a selectable likelihood AND native NaN handling

Run:  cd research && PYTHONPATH="" <python> -u model_ec_v5.py [group...]
Groups: base hist skew nontree blend   (default: base hist skew nontree)
"""
import sys

import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from model_common import load, views, run_table, blend_factory

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET2 = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)
ONE = (7,)


def imp(est_fn):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"), est_fn(s))


def sc(est_fn):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(), est_fn(s))


def logt(fac):
    return lambda s: TransformedTargetRegressor(regressor=fac(s),
                                                func=np.log, inverse_func=np.exp)


def etf(**p):
    return imp(lambda s: ExtraTreesRegressor(random_state=s, **p))


def lgbf(**p):
    q = dict(DET)
    q.update(p)
    return lambda s: lgb.LGBMRegressor(random_state=s, **q)


def hgb(**p):
    q = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
             min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)
    q.update(p)
    return lambda s: HistGradientBoostingRegressor(random_state=s, **q)


def build(v):
    v5 = v["v5"]
    G = {}
    G["base"] = [
        ("[T] v5 ExtraTrees300/2  (reference)", v5, etf(**ET2)),
        ("[T] v5 ExtraTrees100/8", v5, etf(**ET8)),
        ("[T] v5 ExtraTrees600/1", v5, etf(n_estimators=600, max_features=1.0,
                                           min_samples_leaf=1, n_jobs=4)),
        ("[T] v5 RandomForest300/2", v5,
         imp(lambda s: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                             max_features=0.6, n_jobs=4,
                                             random_state=s))),
        ("[T] v5 LGB huber", v5,
         lgbf(objective="huber", n_estimators=800, learning_rate=0.03,
              num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)),
    ]

    # ---- H : histogram GBDT, selectable likelihood, native NaN -------------
    G["hist"] = [
        ("[H] HGB squared_error", v5, hgb(loss="squared_error")),
        ("[HP] HGB gamma", v5, hgb(loss="gamma")),
        ("[HP] HGB poisson", v5, hgb(loss="poisson")),
        ("[HP] HGB absolute_error", v5, hgb(loss="absolute_error")),
        ("[HP] HGB quantile .6", v5, hgb(loss="quantile", quantile=0.6)),
    ]

    # ---- P : skew handled inside the tree family ---------------------------
    lgbp = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0)
    G["skew"] = [
        ("[TP] v5 log-target ET300/2", v5, logt(etf(**ET2))),
        ("[TP] v5 sqrt-target ET300/2", v5,
         lambda s: TransformedTargetRegressor(regressor=etf(**ET2)(s),
                                              func=np.sqrt,
                                              inverse_func=np.square)),
        ("[P] v5 LGB gamma", v5, lgbf(objective="gamma", **lgbp)),
        ("[P] v5 LGB poisson", v5, lgbf(objective="poisson", **lgbp)),
        ("[P] v5 LGB tweedie 1.5", v5,
         lgbf(objective="tweedie", tweedie_variance_power=1.5, **lgbp)),
        ("[P] v5 LGB quantile .6", v5,
         lgbf(objective="quantile", alpha=0.6, **lgbp)),
        ("[P] v5 log-target LGB L2", v5,
         logt(lgbf(objective="regression", **lgbp))),
    ]

    # ---- X S : non-tree families in the same 18-column space ---------------
    G["nontree"] = [
        ("[XS] v5 Ridge a=10", v5, sc(lambda s: Ridge(alpha=10.0)), ONE),
        ("[XS] v5 PLS 10", v5, sc(lambda s: PLSRegression(n_components=10)), ONE),
        ("[S] v5 Nystroem g=.05 n500+R.1", v5,
         sc(lambda s: make_pipeline(Nystroem(gamma=0.05, n_components=500,
                                             random_state=s), Ridge(alpha=0.1)))),
        ("[S] v5 Nystroem g=.2 n800+R.01", v5,
         sc(lambda s: make_pipeline(Nystroem(gamma=0.2, n_components=800,
                                             random_state=s), Ridge(alpha=0.01)))),
        ("[S] v5 SVR rbf C=30 eps=.02", v5,
         sc(lambda s: SVR(C=30.0, epsilon=0.02, gamma="scale", cache_size=700)), ONE),
        ("[XS] v5 MLP 128-64 a=1e-3", v5,
         sc(lambda s: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                   learning_rate_init=1e-3, max_iter=600,
                                   early_stopping=True, n_iter_no_change=25,
                                   validation_fraction=0.12, random_state=s))),
        ("[XS] v5 MLP 256-128-64 a=1e-2", v5,
         sc(lambda s: MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1e-2,
                                   learning_rate_init=1e-3, max_iter=600,
                                   early_stopping=True, n_iter_no_change=25,
                                   validation_fraction=0.12, random_state=s))),
        ("[XSP] v5 log-target MLP 128-64", v5,
         logt(sc(lambda s: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                        learning_rate_init=1e-3, max_iter=600,
                                        early_stopping=True, n_iter_no_change=25,
                                        validation_fraction=0.12,
                                        random_state=s)))),
        ("[.] v5 KNN k=15 dist", v5,
         sc(lambda s: KNeighborsRegressor(n_neighbors=15, weights="distance",
                                          n_jobs=4)), ONE),
    ]
    return G


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    G = build(v)
    want = sys.argv[1:] or ["base", "hist", "skew", "nontree"]
    for g in want:
        run_table(lab_e, "sub_ec", G[g], segment=True, tag="[%s]" % g)


if __name__ == "__main__":
    main()
