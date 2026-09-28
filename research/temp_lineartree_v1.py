# -*- coding: utf-8 -*-
"""Web research 3, idea 4: LightGBM linear_tree as an extra temperature member.
Leaves hold a ridge-regularised linear model, so predictions keep following
the input trend beyond the training range (the test is colder).

Member: Codex resid_reset features (within-day, causal, no cross-day path),
median-imputed + standardised (fitted on the training fold), LightGBM
linear_tree=True, linear_lambda=5, num_leaves=15, 400 trees, lr 0.03,
feature_fraction 0.8 + bagging 0.8 (so the seed matters), round-5 weights.
Predictions clipped to the training label range.

Reference R = 0.8 * MASK base + 0.2 * Codex (temp_mask_v1.py).
Candidate  = 0.7 * MASK base + 0.2 * Codex + 0.1 * linear-tree   (fixed in advance)
Pre-set rule: adopt only if the candidate beats R on DIAG10 AND EXT10 for
every combination of base seed (7, 101) x member seed (1, 2), and the DIAG10
block-bootstrap CI excludes 0 in every combination.

Run:  cd research && PYTHONPATH="" <python> -u temp_lineartree_v1.py
"""
import os
import sys

import env  # noqa: F401
import numpy as np
from lightgbm import LGBMRegressor

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402


def fit_predict(tr, va, w, seed, lo, hi):
    Xtr, Xva = tr[FEATURE_COLUMNS].astype(float), va[FEATURE_COLUMNS].astype(float)
    med = Xtr.median()
    Xtr, Xva = Xtr.fillna(med).fillna(0.0), Xva.fillna(med).fillna(0.0)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    m = LGBMRegressor(linear_tree=True, linear_lambda=5.0, num_leaves=15, n_estimators=400, learning_rate=0.03,
                      min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      random_state=seed, n_jobs=4, verbosity=-1)
    m.fit(Xtr.values, tr.sub_temp.values, sample_weight=w)
    return np.clip(m.predict(Xva.values), lo, hi)


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    from features_v4 import phys_features
    ph = phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    cd = dmin[dmin < 10.0]
    sets.append(("EXT10", [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    ok = True
    for s, fds in sets:
        lt = {}
        for ms in (1, 2):
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr = lab[trm]
                o[vam] = fit_predict(tr, lab[vam], w[trm], ms, tr.sub_temp.min(), tr.sub_temp.max())
            lt[ms] = o
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * cx
            for ms in (1, 2):
                cand = 0.7 * base + 0.2 * cx + 0.1 * lt[ms]
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("  %-6s base %3d member %d | linear-tree alone %.5f | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, bs, ms, rmse(lt[ms][g], y[g]), rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi),
                      flush=True)
                ok = ok and d < 0 and (s != "DIAG10" or hi < 0)
    print("\nPRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT")


if __name__ == "__main__":
    main()
