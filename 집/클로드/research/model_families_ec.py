# -*- coding: utf-8 -*-
"""sub_ec: survey of model families grouped by CAPABILITY, not by name.

Capabilities that matter for this problem
-----------------------------------------
  X  extrapolate beyond the training range of y   (trees cannot; this is the
     stated reason the y>1 segment is under-predicted by ~0.5)
  S  smooth response surface vs. piecewise-constant
  E  sample efficiency at ~400 effective (greenhouse-day) samples
  M  native NaN handling (otherwise an imputer is mandatory)
  P  a positive, right-skewed target handled by the link/likelihood itself

Run:  cd research && PYTHONPATH="" <python> -u model_families_ec.py [group...]
Groups: base lin glm kern mlp lgbobj  (default: all)
"""
import sys

import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  GammaRegressor, HuberRegressor,
                                  PoissonRegressor, Ridge, TweedieRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.compose import TransformedTargetRegressor

from model_common import load, views, run_table, blend_factory

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)
ONE = (7,)  # deterministic estimators: one seed is enough


def lgbf(**p):
    q = dict(DET)
    q.update(p)
    return lambda s: lgb.LGBMRegressor(random_state=s, **q)


def etf(**p):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   ExtraTreesRegressor(random_state=s, **p))


def sc(est_fn, scaler="standard"):
    """Imputer + scaler + estimator.  est_fn(seed) -> estimator."""
    def f(s):
        steps = [SimpleImputer(strategy="median")]
        if scaler == "standard":
            steps.append(StandardScaler())
        elif scaler == "quantile":
            steps.append(QuantileTransformer(output_distribution="normal",
                                             n_quantiles=1000, random_state=0))
        steps.append(est_fn(s))
        return make_pipeline(*steps)
    return f


def logt(est_fn):
    """Fit on log(y), predict back through exp.  y is strictly positive."""
    return lambda s: TransformedTargetRegressor(
        regressor=est_fn(s), func=np.log, inverse_func=np.exp)


def build(v):
    ec, v5, raw = v["ec"], v["v5"], v["raw"]
    G = {}

    G["base"] = [
        ("[T] submitted blend(ET8,LGBhub)", ec,
         blend_factory([etf(**ET8), lgbf(**E_HUB)])),
        ("[T] ExtraTrees 100/8", ec, etf(**ET8)),
        ("[T] LGB huber", ec, lgbf(**E_HUB)),
    ]

    # ---- X S : linear, extrapolates, smooth, very sample efficient --------
    G["lin"] = [
        ("[XS] Ridge a=100 68f", ec, sc(lambda s: Ridge(alpha=100.0)), ONE),
        ("[XS] Ridge a=1000 68f", ec, sc(lambda s: Ridge(alpha=1000.0)), ONE),
        ("[XS] Ridge a=100 18f", v5, sc(lambda s: Ridge(alpha=100.0)), ONE),
        ("[XS] Ridge a=100 14f", raw, sc(lambda s: Ridge(alpha=100.0)), ONE),
        ("[XS] Ridge a=100 qt 68f", ec,
         sc(lambda s: Ridge(alpha=100.0), "quantile"), ONE),
        ("[XS] ElasticNet a=.01 l1=.5", ec,
         sc(lambda s: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)), ONE),
        ("[XS] BayesianRidge", ec, sc(lambda s: BayesianRidge()), ONE),
        ("[XS] ARD", ec, sc(lambda s: ARDRegression(max_iter=200)), ONE),
        ("[XS] Huber-loss linear", ec,
         sc(lambda s: HuberRegressor(alpha=1e-2, max_iter=500)), ONE),
        ("[XS] PLS 10 comp", ec,
         sc(lambda s: PLSRegression(n_components=10)), ONE),
        ("[XS] PLS 25 comp", ec,
         sc(lambda s: PLSRegression(n_components=25)), ONE),
        ("[XS] LinearSVR eps=.05", ec,
         sc(lambda s: LinearSVR(C=1.0, epsilon=0.05, max_iter=8000,
                                random_state=s)), ONE),
    ]

    # ---- X S P : generalised linear, positive skewed target ---------------
    G["glm"] = [
        ("[XSP] Gamma a=1", ec,
         sc(lambda s: GammaRegressor(alpha=1.0, max_iter=500)), ONE),
        ("[XSP] Gamma a=.1", ec,
         sc(lambda s: GammaRegressor(alpha=0.1, max_iter=500)), ONE),
        ("[XSP] Gamma a=.1 18f", v5,
         sc(lambda s: GammaRegressor(alpha=0.1, max_iter=500)), ONE),
        ("[XSP] Tweedie p=1.5 a=.1", ec,
         sc(lambda s: TweedieRegressor(power=1.5, alpha=0.1, max_iter=500)), ONE),
        ("[XSP] Tweedie p=2.5 a=.1", ec,
         sc(lambda s: TweedieRegressor(power=2.5, alpha=0.1, max_iter=500)), ONE),
        ("[XSP] Poisson a=.1", ec,
         sc(lambda s: PoissonRegressor(alpha=0.1, max_iter=500)), ONE),
        ("[XSP] log-target Ridge a=100", ec,
         logt(sc(lambda s: Ridge(alpha=100.0))), ONE),
        ("[XSP] log-target Ridge a=10", ec,
         logt(sc(lambda s: Ridge(alpha=10.0))), ONE),
    ]

    # ---- S (+partial X) : kernel, smooth, moderate sample efficiency ------
    def nys(gamma, ncomp, alpha):
        return sc(lambda s: make_pipeline(
            Nystroem(gamma=gamma, n_components=ncomp, random_state=s),
            Ridge(alpha=alpha)))

    G["kern"] = [
        ("[S] Nystroem g=.01 n300 + Ridge1", ec, nys(0.01, 300, 1.0)),
        ("[S] Nystroem g=.003 n500 + Ridge1", ec, nys(0.003, 500, 1.0)),
        ("[S] Nystroem g=.03 n500 + Ridge.1", ec, nys(0.03, 500, 0.1)),
        ("[S] Nystroem g=.01 n500 18f", v5, nys(0.01, 500, 1.0)),
        ("[S] SVR rbf C=10 eps=.05", ec,
         sc(lambda s: SVR(C=10.0, epsilon=0.05, gamma="scale", cache_size=700)), ONE),
        ("[S] SVR rbf C=100 eps=.02", ec,
         sc(lambda s: SVR(C=100.0, epsilon=0.02, gamma="scale", cache_size=700)), ONE),
    ]

    # ---- X S : MLP, extrapolates (linear output), smooth ------------------
    def mlp(h, alpha, lr=1e-3, it=400):
        return sc(lambda s: MLPRegressor(hidden_layer_sizes=h, alpha=alpha,
                                         learning_rate_init=lr, max_iter=it,
                                         early_stopping=True, n_iter_no_change=20,
                                         validation_fraction=0.12,
                                         random_state=s))
    G["mlp"] = [
        ("[XS] MLP 64-32 a=1e-3", ec, mlp((64, 32), 1e-3)),
        ("[XS] MLP 128-64 a=1e-2", ec, mlp((128, 64), 1e-2)),
        ("[XS] MLP 256 a=1e-1", ec, mlp((256,), 1e-1)),
        ("[XS] MLP 64-32 18f", v5, mlp((64, 32), 1e-3)),
        ("[XSP] log-target MLP 64-32", ec, logt(mlp((64, 32), 1e-3))),
    ]

    # ---- P : same trees, distribution-aware objectives --------------------
    base = dict(n_estimators=400, learning_rate=0.02, num_leaves=7,
                min_child_samples=240, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.4, reg_lambda=5.0)
    G["lgbobj"] = [
        ("[P] LGB gamma", ec, lgbf(objective="gamma", **base)),
        ("[P] LGB poisson", ec, lgbf(objective="poisson", **base)),
        ("[P] LGB tweedie p=1.3", ec,
         lgbf(objective="tweedie", tweedie_variance_power=1.3, **base)),
        ("[P] LGB tweedie p=1.7", ec,
         lgbf(objective="tweedie", tweedie_variance_power=1.7, **base)),
        ("[P] LGB quantile .5", ec,
         lgbf(objective="quantile", alpha=0.5, **base)),
        ("[P] LGB quantile .65", ec,
         lgbf(objective="quantile", alpha=0.65, **base)),
        ("[P] LGB L2", ec, lgbf(objective="regression", **base)),
        ("[P] log-target LGB huber", ec, logt(lgbf(**E_HUB))),
        ("[P] log-target LGB L2", ec, logt(lgbf(objective="regression", **base))),
    ]
    return G


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    G = build(v)
    want = sys.argv[1:] or list(G)
    for g in want:
        run_table(lab_e, "sub_ec", G[g], segment=True, tag="[%s]" % g)


if __name__ == "__main__":
    main()
