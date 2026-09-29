# -*- coding: utf-8 -*-
"""GPBoost, second and last attempt (web_gpboost_v1.py failed by design).

v1 diagnosis: the free random-effect variance was estimated at 9.6 (error
0.025), i.e. the day effect swallowed the day level that the INPUTS explain
(daily air temperature), the trees learned almost nothing, and new test days
(random effect 0) got a near-constant prediction (member RMSE 3.3 vs 0.85 for
plain LightGBM on the same fold).

v2: covariance parameters FIXED (not estimated) to the unexplained variance
components of our current models (training out-of-fold decomposition):
  temperature  error 0.21 (within-day residual 0.459^2), day 0.27 (day offset 0.524^2)
  EC           error 0.0074, day 0.0363 (83% of 0.209^2)
so the random effect can only take the part the inputs do NOT explain.
Everything else (features, folds, blend weights, pre-set rules) as v1.

Run:  cd research && PYTHONPATH="" <python> -u web_gpboost_v2.py
"""
import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
import gpboost as gpb

import web_gpboost_v1 as V1

COV = {"temp": (0.21, 0.27), "ec": (0.0074, 0.0363)}
CURRENT = {"cov": COV["temp"]}


def gpb_fit_predict_fixed(Xtr, ytr, gtr, Xva, w, seed):
    kw = {} if w is None else {"weights": np.asarray(w, dtype=float)}
    gm = gpb.GPModel(group_data=gtr, likelihood="gaussian", **kw)
    gm.set_optim_params(params={"init_cov_pars": np.array(CURRENT["cov"], dtype=float),
                                "estimate_cov_par_index": np.array([0, 0], dtype=np.int32)})
    ds = gpb.Dataset(Xtr, ytr)
    params = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
                  feature_fraction_seed=seed, seed=seed, verbose=-1, num_threads=4)
    bst = gpb.train(params=params, train_set=ds, gp_model=gm, num_boost_round=300)
    pr = bst.predict(data=Xva, group_data_pred=np.full(len(Xva), -1), pred_latent=True)
    return np.asarray(pr["fixed_effect"])


V1.gpb_fit_predict = gpb_fit_predict_fixed

if __name__ == "__main__":
    CURRENT["cov"] = COV["temp"]
    V1.temperature()
    CURRENT["cov"] = COV["ec"]
    V1.ec()
