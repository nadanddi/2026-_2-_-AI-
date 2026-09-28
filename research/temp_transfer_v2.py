# -*- coding: utf-8 -*-
"""Other greenhouses, second design (temp_transfer_v1.py: blending a member
that sees only in_temp/in_hum/in_co2 hurt DIAG10 by ~+0.9..1.2% - the member
alone is 1.07-1.20 vs 0.69, it lacks weather and actuators).

Here the other greenhouses only supply a PRIOR: g = LightGBM trained on the
48 other greenhouses (F32 excluded; F13/F47 labels never used, so no fold
leakage) mapping the in_temp/in_hum/in_co2 history to sub_temp.  g is then an
extra FEATURE for our own models, which keep all inputs and decide how much
to trust it.  g for F13/F47 rows uses MASK-world inputs (test_X NaN).

Arms (fixed):
  GB : g added to the base members' features (ct + g), Codex unchanged
  GC : g added to the Codex member's features, base unchanged
Blend 0.8*base + 0.2*Codex in both; reference = the npz OOF (same pipeline,
no g).  Validators DIAG10 / EXT10 / EXT12, seeds base 7/101, Codex 726/727.
Rule (2026-09-27): same direction in every seed x validator, DIAG10 CI
excluding 0 (2 arms -> check the 97.5% level; boot gives 95%, reported).

Run:  cd research && PYTHONPATH="" <python> -u temp_transfer_v2.py
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
from common import split_mask, rmse, TARGET_FARMS
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader, build_world, ORIG
from temp_transfer_v1 import feats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402


def prior(lab_row_ids):
    tX, ty, _ = ORIG()
    oth = feats(tX[~tX.farm.isin(list(TARGET_FARMS) + ["F32"])]).merge(ty[["row_id", "sub_temp"]], on="row_id")
    oth = oth.dropna(subset=["sub_temp"])
    cols = [c for c in oth.columns if c not in ("row_id", "farm", "sub_temp", "is_F13", "is_F47")]
    g = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbosity=-1, n_jobs=6, random_state=0)
    g.fit(oth[cols], oth.sub_temp.values)
    mX, _, msX = masked_loader()
    import pandas as pd
    tgt = feats(pd.concat([mX[mX.farm.isin(TARGET_FARMS)], msX], ignore_index=True)).set_index("row_id")
    return g.predict(tgt.loc[lab_row_ids, cols])


def codex_fit_predict(tr, va, w, seed, fc):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=seed)
    m.fit(tr[fc], tr.sub_temp.values - lin.predict(tr[PHYSICS_COLUMNS]), sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS]) + m.predict(va[fc])


def main():
    common.load_raw = masked_loader
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    tX, ty, sX = ORIG()
    CF = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in FEATURE_COLUMNS:
        if c not in lab.columns:
            lab[c] = CF.loc[lab.row_id, c].values
    lab["g_prior"] = prior(lab.row_id.values)
    y = lab.sub_temp.values
    print("prior g alone on F13/F47 training rows: RMSE %.4f, corr %.3f"
          % (rmse(lab.g_prior.values, y), np.corrcoef(lab.g_prior.values, y)[0, 1]), flush=True)
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    def blend(M):
        return 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]

    ok = {"GB": True, "GC": True}
    for s, fds in sets:
        for bs, cs in ((7, 726), (101, 727)):
            base0 = z["%s__MASK__%d" % (s, bs)]
            cx0 = z["%s__CODEX__%d" % (s, cs)]
            g = ~np.isnan(base0)
            ref = 0.8 * base0 + 0.2 * cx0
            cold_v5.SEED = bs
            baseG = blend(collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct + ["g_prior"], phc, w[m])))
            cxG = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                cxG[vam] = codex_fit_predict(lab[trm], lab[vam], w[trm], cs, FEATURE_COLUMNS + ["g_prior"])
            for arm, cand in (("GB", 0.8 * baseG + 0.2 * cx0), ("GC", 0.8 * base0 + 0.2 * cxG)):
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("%-6s %s seeds %3d/%d | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, arm, bs, cs, rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi), flush=True)
                ok[arm] = ok[arm] and d < 0 and (s != "DIAG10" or hi < 0)
    cold_v5.SEED = 7
    for arm, v in ok.items():
        print("%s PRE-SET RULE VERDICT (95%% CI; check 97.5%%): %s" % (arm, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
