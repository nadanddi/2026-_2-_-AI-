# -*- coding: utf-8 -*-
"""Independent re-check of Codex's resid_reset 20% blend (analysis/codex_independent/2차).

Codex's feature module is imported read-only; the folds, the buffer rule and the
scoring are Claude's own (harness split_mask, anal_q1_errors.diag_folds,
screen_v6.boot), so a mismatch in fold handling would show up as a different
number.  Model: Ridge(alpha 100) on the 19 physics columns + LightGBM on the
Ridge residual (89 columns), both with the round-5 training weights
(train_flags_v6.row_weights(lab, 0.2, w_noisy=0.2)).  Blend 0.8 * round-5
temperature OOF (eval_v6 F60ND) + 0.2 * new model.

Codex reported: DIAG10 0.69654 -> 0.68684, EXT10 0.88056 -> 0.86743,
EXT12 0.88578 -> 0.85202.

Run:  cd research && PYTHONPATH="" <python> -u verify_codex_resid_reset.py
"""
import os
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(env.LOCAL), "..", "analysis", "codex_independent", "2차"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402


def fit_predict(tr, va, w):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=726)
    m.fit(tr[FEATURE_COLUMNS], tr.sub_temp.values - lin.predict(tr[PHYSICS_COLUMNS]), sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS]) + m.predict(va[FEATURE_COLUMNS])


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].copy()
    lab = lab.join(F, on="row_id")
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values

    from features_v4 import phys_features
    ph = phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    for s, fds in sets:
        new = np.full(len(lab), np.nan)
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            new[vam] = fit_predict(lab[trm], lab[vam], w[trm])
        base = z["F60ND__%s" % s]
        g = ~np.isnan(base)
        assert (g == ~np.isnan(new)).all()
        bl = 0.8 * base + 0.2 * new
        pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", base[g], bl[g])
        print("%-7s base %.5f | new alone %.5f | blend %.5f | delta %+.5f [%+.5f, %+.5f] P(worse)=%.3f"
              % (s, rmse(base[g], y[g]), rmse(new[g], y[g]), rmse(bl[g], y[g]), pr, lo, hi, pw), flush=True)
        if s == "DIAG10":
            np.savez(env.LOCAL + "/verify_codex_resid_reset_diag.npz", row_id=lab.row_id.values, new=new)


if __name__ == "__main__":
    main()
