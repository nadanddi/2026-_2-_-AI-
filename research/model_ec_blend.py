# -*- coding: utf-8 -*-
"""sub_ec: does a capability the tree lacks buy anything IN A BLEND?

Single-model tables say the non-tree families lose badly on their own.  That
does not settle the blend question: a partner with decorrelated errors can
still help.  Here the v5 ExtraTrees reference is blended with one partner at a
time over a weight grid, and the y>1 segment is reported next to overall RMSE.

Partner is chosen on the command line so each run stays short.
Run:  cd research && PYTHONPATH="" <python> -u model_ec_blend.py [partner...]
Partners: lgbhub hgbpoi hgbsq mlp nys svr ridge pls logtree lgbgam hgbgam knn
(default: the boosted ones, which are the only partners with a materially
smaller y>1 bias than the ExtraTrees reference)
"""
import sys

import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
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
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
WEIGHTS = (0.25, 0.40, 0.50, 0.65)


def imp(f):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"), f(s))


def sc(f):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(), f(s))


REF = imp(lambda s: ExtraTreesRegressor(random_state=s, **ET1))
HGBP = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)


def partners():
    lgbp = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0)
    return {
        "lgbhub": ("LGBhuber", lambda s: lgb.LGBMRegressor(
            random_state=s, objective="huber", **DET, **lgbp)),
        "hgbpoi": ("HGBpoisson", lambda s: HistGradientBoostingRegressor(
            random_state=s, loss="poisson", **HGBP)),
        "hgbsq": ("HGBsq", lambda s: HistGradientBoostingRegressor(
            random_state=s, loss="squared_error", **HGBP)),
        "mlp": ("MLP128-64", sc(lambda s: MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
            max_iter=600, early_stopping=True, n_iter_no_change=25,
            validation_fraction=0.12, random_state=s))),
        "nys": ("Nystroem.2/800", sc(lambda s: make_pipeline(
            Nystroem(gamma=0.2, n_components=800, random_state=s),
            Ridge(alpha=0.01)))),
        "svr": ("SVR C=30", sc(lambda s: SVR(C=30.0, epsilon=0.02,
                                             gamma="scale", cache_size=700))),
        "ridge": ("Ridge10", sc(lambda s: Ridge(alpha=10.0))),
        "pls": ("PLS10", sc(lambda s: PLSRegression(n_components=10))),
        "logtree": ("logET300/2", lambda s: TransformedTargetRegressor(
            regressor=REF(s), func=np.log, inverse_func=np.exp)),
        "lgbgam": ("LGBgamma", lambda s: lgb.LGBMRegressor(
            random_state=s, objective="gamma", **DET, **lgbp)),
        "hgbgam": ("HGBgamma", lambda s: HistGradientBoostingRegressor(
            random_state=s, loss="gamma", **HGBP)),
        "knn": ("KNN15", sc(lambda s: KNeighborsRegressor(
            n_neighbors=15, weights="distance", n_jobs=4))),
    }


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    v5 = v["v5"]
    P = partners()
    want = sys.argv[1:] or ["lgbhub", "hgbpoi", "hgbsq"]
    cands = [("[T] v5 ET600/1 (reference)", v5, REF)]
    for k in want:
        nm, fac = P[k]
        for w in WEIGHTS:
            cands.append(("+%s w=%.2f" % (nm, w), v5,
                          blend_factory([REF, fac], [1.0 - w, w])))
    run_table(lab_e, "sub_ec", cands, segment=True, tag="[blend]")


if __name__ == "__main__":
    main()
