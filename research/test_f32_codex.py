# -*- coding: utf-8 -*-
"""Blind analyst B's idea: add greenhouse F32 (the only other house with
outdoor weather + some actuators) to the temperature training.

F32: 5,465 labelled rows, not spliced (midnight jump ratio 0.49), but its
substrate is on average +7.2 C WARMER than the air (F13/F47: -0.7 C) -
root-zone heating, a different system.  Tested on the Codex resid_reset
model (the only member cheap to rebuild for another greenhouse), with an
F32 indicator; F32 rows weighted 1.0 or 0.3 (fixed in advance).

Pre-set rule: adopt a variant only if 0.8*MASK base + 0.2*Codex(F32 variant)
beats 0.8*MASK base + 0.2*Codex(no F32) on DIAG10 AND EXT10 for BOTH base
seeds (7, 101), with the DIAG10 block-bootstrap CI excluding 0.
Only F13/F47 rows are scored; F32 rows are always in training.

Run:  cd research && PYTHONPATH="" <python> -u test_f32_codex.py
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
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
import resid_reset_features as R  # noqa: E402

FEATS = R.FEATURE_COLUMNS + ["is_f32"]
PHYS = R.PHYSICS_COLUMNS + ["is_f32"]


def build(tX, sX, farms):
    """Same computation as R.build_features, for any set of greenhouses."""
    a = pd.concat([R._read(tX), R._read(sX)], ignore_index=True)
    a = a[a.farm.isin(farms)].sort_values(["farm", "t"]).reset_index(drop=True)
    z = {c: a[c] for c in R.METADATA}
    z.update(farm_id=(a.farm == "F47").astype(float), sin=np.sin(a.hour * np.pi / 12),
             cos=np.cos(a.hour * np.pi / 12), second=(a.day >= 179).astype(float))
    for c in R.USABLE:
        v = a[c]
        g = v.groupby([a.farm, a.day])
        z[c] = v
        z[c + "_h0"] = v.where(a.hour == 0).groupby([a.farm, a.day]).ffill()
        z[c + "_mean"] = g.transform(lambda x: x.expanding().mean())
        z[c + "_std"] = g.transform(lambda x: x.expanding().std())
        z[c + "_diff"] = g.diff()
        if c in R.FILTERED:
            for h in [1, 3, 8]:
                z[c + "_reset" + str(h)] = g.transform(lambda x, h=h: x.ewm(halflife=h).mean())
    z["delta"] = a.in_temp - a.out_temp
    z["is_f32"] = (a.farm == "F32").astype(float)
    return pd.DataFrame(z)


def fit_predict(tr, va, w):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYS], tr.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=726)
    m.fit(tr[FEATS], tr.sub_temp.values - lin.predict(tr[PHYS]), sample_weight=w)
    return lin.predict(va[PHYS]) + m.predict(va[FEATS])


def main():
    tX, ty, sX = common.load_raw()
    Z = build(tX, sX, ["F13", "F47", "F32"]).merge(ty[["row_id", "sub_temp"]], on="row_id", how="inner")
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    tgt = Z[Z.farm.isin(TARGET_FARMS)].set_index("row_id").loc[z["row_id"]].reset_index()
    f32 = Z[Z.farm == "F32"].reset_index(drop=True)
    w_t = TF.row_weights(tgt, 0.2, w_noisy=0.2)
    y = tgt.sub_temp.values
    print("target rows %d | F32 rows %d" % (len(tgt), len(f32)), flush=True)

    from features_v4 import phys_features
    ph = phys_features().set_index("row_id").loc[tgt.row_id]
    dmin = ph.groupby([tgt.farm.values, tgt.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(tgt))]
    cd = dmin[dmin < 10.0]
    sets.append(("EXT10", [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    verdict = {}
    for s, fds in sets:
        preds = {k: np.full(len(tgt), np.nan) for k in ("none", "w1.0", "w0.3")}
        for fd in fds:
            trm, vam = split_mask(tgt, fd)
            tr, va = tgt[trm], tgt[vam]
            preds["none"][vam] = fit_predict(tr, va, w_t[trm])
            for k, wf in (("w1.0", 1.0), ("w0.3", 0.3)):
                tr2 = pd.concat([tr, f32], ignore_index=True)
                w2 = np.concatenate([w_t[trm], np.full(len(f32), wf)])
                preds[k][vam] = fit_predict(tr2, va, w2)
        for sd in (7, 101):
            base = z["%s__MASK__%d" % (s, sd)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * preds["none"]
            for k in ("w1.0", "w0.3"):
                cand = 0.8 * base + 0.2 * preds[k]
                pr, lo, hi, pw = boot(tgt[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("  %-6s seed %3d  F32 %s | Codex alone %.5f -> %.5f | blend %.5f -> %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, sd, k, rmse(preds["none"][g], y[g]), rmse(preds[k][g], y[g]),
                         rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi), flush=True)
                ok = d < 0 and (s != "DIAG10" or hi < 0)
                verdict[k] = verdict.get(k, True) and ok
    print("\nPRE-SET RULE VERDICT:", {k: ("ADOPT" if v else "REJECT") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
