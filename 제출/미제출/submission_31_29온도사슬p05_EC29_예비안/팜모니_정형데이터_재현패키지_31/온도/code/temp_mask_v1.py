# -*- coding: utf-8 -*-
"""Close the indirect path found by inspector 4-2 (감사결과_2026-09-27.md).

Problem: day-crossing features (ewm / rolling / hours-since) of TRAINING rows
that lie after a test block are computed over train+test inputs, so the
fitted model - and therefore the predictions for EARLIER test rows - depend
on LATER test inputs.

Fix ("MASK world"): training-row features are built with every test_X input
set to NaN; test-row features keep the full history (only current/earlier
inputs of the same greenhouse, already verified cell by cell).  In cross-
validation every scored row is a training row, so both fitting and scoring
use the MASK world here.

Pre-set decision criteria (the user asked for skill, not luck):
  * FULL -> MASK is a compliance fix, adopted regardless; we only measure the
    cost.  |change| < 1% counts as "no measurable cost".
  * Codex 20% blend on top of MASK is kept only if it improves DIAG10 AND
    EXT10 for BOTH seeds.
Seeds: temperature members 7 and 101 (cold_v5.SEED); Codex model 726 and 727.
Weights: round-5 training weights (train_flags_v6, training inputs only).

Run:  cd research && PYTHONPATH="" <python> -u temp_mask_v1.py
"""
import os
import sys

import env  # noqa: F401
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
import harness
import cold_v5
from common import USABLE, split_mask, rmse, TARGET_FARMS
from harness import views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402

ORIG = common.load_raw
SEEDS_T = (7, 101)
SEEDS_C = (726, 727)


def masked_loader():
    tX, ty, sX = ORIG()
    sX = sX.copy()
    cols = [c for c in sX.columns if c not in ("row_id", "farm", "day", "hour", "t")]
    sX[cols] = np.nan
    return tX, ty, sX


def build_world():
    harness._CACHE.clear()
    panel, lab0, _ = harness.load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    ct = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"]) + F4.names(sg) + F4.names(fp)
    return lab, ct, F4.names(ph)


def codex_fit_predict(tr, va, w, seed):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=seed)
    m.fit(tr[FEATURE_COLUMNS], tr.sub_temp.values - lin.predict(tr[PHYSICS_COLUMNS]), sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS]) + m.predict(va[FEATURE_COLUMNS])


def main():
    labF, ct, phc = build_world()
    common.load_raw = masked_loader
    try:
        labM, ctM, phcM = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    assert ctM == ct and phcM == phc and (labM.row_id.values == labF.row_id.values).all()
    diffcols = [c for c in ct + phc if not np.allclose(labF[c].values.astype(float), labM[c].values.astype(float),
                                                       equal_nan=True)]
    nrow = int((~np.isclose(labF[ct].values.astype(float), labM[ct].values.astype(float), equal_nan=True)).any(1).sum())
    print("MASK vs FULL: %d feature columns differ, %d training rows affected" % (len(diffcols), nrow), flush=True)

    tX, ty, sX = ORIG()
    CF = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for lab in (labF, labM):
        for c in FEATURE_COLUMNS:
            if c not in lab.columns:
                lab[c] = CF.loc[lab.row_id, c].values
    cx = [c for c in FEATURE_COLUMNS if c in labM.columns]
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    y = labM.sub_temp.values
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    res = {}
    for s, fds in sets:
        for world, lab in (("FULL", labF), ("MASK", labM)):
            for sd in SEEDS_T:
                cold_v5.SEED = sd
                M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, w[m]))
                res[(s, world, sd)] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        for sc in SEEDS_C:
            o = np.full(len(labM), np.nan)
            for fd in fds:
                trm, vam = split_mask(labM, fd)
                o[vam] = codex_fit_predict(labM[trm], labM[vam], w[trm], sc)
            res[(s, "CODEX", sc)] = o
        cold_v5.SEED = 7
        print("  %s done" % s, flush=True)
    np.savez(env.LOCAL + "/temp_mask_v1_oof.npz", row_id=labM.row_id.values,
             **{"%s__%s__%s" % k: v for k, v in res.items()})

    print("\n== FULL -> MASK (compliance fix cost) ==")
    for s, _ in sets:
        for sd in SEEDS_T:
            a, b = res[(s, "FULL", sd)], res[(s, "MASK", sd)]
            g = ~np.isnan(a)
            pr, lo, hi, pw = boot(labM[g].reset_index(drop=True), "sub_temp", a[g], b[g])
            print("  %-6s seed %3d  FULL %.5f  MASK %.5f  %+.2f%% [%+.4f, %+.4f]"
                  % (s, sd, rmse(a[g], y[g]), rmse(b[g], y[g]), 100 * (rmse(b[g], y[g]) / rmse(a[g], y[g]) - 1), lo, hi))
    print("\n== MASK base -> 0.8 MASK + 0.2 Codex (all seed pairs) ==")
    for s, _ in sets:
        for sd in SEEDS_T:
            for sc in SEEDS_C:
                a = res[(s, "MASK", sd)]
                b = 0.8 * a + 0.2 * res[(s, "CODEX", sc)]
                g = ~np.isnan(a)
                pr, lo, hi, pw = boot(labM[g].reset_index(drop=True), "sub_temp", a[g], b[g])
                print("  %-6s T%3d C%3d  base %.5f  blend %.5f  %+.2f%% [%+.4f, %+.4f] P(worse)=%.3f"
                      % (s, sd, sc, rmse(a[g], y[g]), rmse(b[g], y[g]), 100 * (rmse(b[g], y[g]) / rmse(a[g], y[g]) - 1),
                         lo, hi, pw))
    print("\n== seed-to-seed spread of the base (luck scale) ==")
    for s, _ in sets:
        a, b = res[(s, "MASK", SEEDS_T[0])], res[(s, "MASK", SEEDS_T[1])]
        g = ~np.isnan(a)
        print("  %-6s seed7 %.5f seed101 %.5f diff %+.2f%%" % (s, rmse(a[g], y[g]), rmse(b[g], y[g]),
                                                             100 * (rmse(b[g], y[g]) / rmse(a[g], y[g]) - 1)))


if __name__ == "__main__":
    main()
