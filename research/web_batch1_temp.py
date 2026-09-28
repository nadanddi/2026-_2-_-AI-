# -*- coding: utf-8 -*-
"""Web-research batch 1, temperature (research 1 & 3: RC thermal models; research
2 & 1: monotone constraints).  Both are tested inside the Codex resid_reset
member (within-day features only -> no cross-day / cross-test path), then in
the blend.

  RC    grey-box substrate state, reset at 00 h of every record day:
          x0 = in_temp(00) + d
          x(t+1) = x(t) + a (in_temp(t+1) - x(t)) + b heat(t+1) + c rad_eff(t+1)
        a, b, c, d fitted by least squares to sub_temp on the TRAINING fold
        (round-5 weights), then x(t) is added as a feature (to both the Ridge
        baseline and the LightGBM residual).
  MONO  LightGBM residual with monotone +1 on the in_temp family
        (in_temp, _h0, _mean, _reset1/3/8) - warmer air, warmer slab.

Reference R = 0.8 * MASK base + 0.2 * Codex (temp_mask_v1.py).
Candidate  = 0.8 * MASK base + 0.2 * Codex-variant.
Pre-set rule: adopt only if the candidate beats R on DIAG10 AND EXT10 for
both base seeds (7, 101) and the DIAG10 block-bootstrap CI excludes 0.

Run:  cd research && PYTHONPATH="" <python> -u web_batch1_temp.py
"""
import os
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import least_squares
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402

MONO_SET = {"in_temp", "in_temp_h0", "in_temp_mean", "in_temp_reset1", "in_temp_reset3", "in_temp_reset8"}


def rc_simulate(p, T, H, R, day_start):
    a, b, c, d = p
    x = np.empty_like(T)
    for i in range(len(T)):
        if day_start[i] or np.isnan(x[i - 1]):
            x[i] = T[i] + d
        else:
            x[i] = x[i - 1] + a * (T[i] - x[i - 1]) + b * H[i] + c * R[i]
    return x


def rc_inputs(df):
    T = df.in_temp.ffill().bfill().values.astype(float)
    H = df.act_heating.fillna(0).values.astype(float) / 100.0
    R = df.rad_eff.fillna(0).values.astype(float) / 1000.0
    start = np.r_[True, (df.farm.values[1:] != df.farm.values[:-1]) | (df.day.values[1:] != df.day.values[:-1])]
    return T, H, R, start


def rc_fit(tr, w):
    T, H, R, st = rc_inputs(tr)
    y = tr.sub_temp.values
    sw = np.sqrt(w)
    f = lambda p: sw * (rc_simulate(p, T, H, R, st) - y)
    return least_squares(f, x0=[0.3, 0.5, 0.5, 0.0], bounds=([0.01, -5, -5, -5], [1.0, 5, 5, 5])).x


def codex(tr, va, w, feats, phys, mono=False):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[phys], tr.sub_temp.values, ridge__sample_weight=w)
    kw = {}
    if mono:
        kw = dict(monotone_constraints=[1 if c in MONO_SET else 0 for c in feats], monotone_constraints_method="intermediate")
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=726, **kw)
    m.fit(tr[feats], tr.sub_temp.values - lin.predict(tr[phys]), sample_weight=w)
    return lin.predict(va[phys]) + m.predict(va[feats])


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp", "act_heating", "rad_eff"]].join(
        F.drop(columns=[c for c in ("act_heating",) if c in F.columns]), on="row_id")
    lab = lab.sort_values(["farm", "t"]).reset_index(drop=True)
    order = pd.Series(np.arange(len(lab)), index=lab.row_id).loc[z["row_id"]].values
    w_all = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    from features_v4 import phys_features
    ph = phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    cd = dmin[dmin < 10.0]
    sets.append(("EXT10", [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    verdict = {"RC": True, "MONO": True}
    for s, fds in sets:
        P = {k: np.full(len(lab), np.nan) for k in ("ref", "RC", "MONO")}
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            tr, va = lab[trm].copy(), lab[vam].copy()
            w = w_all[trm]
            P["ref"][vam] = codex(tr, va, w, FEATURE_COLUMNS, PHYSICS_COLUMNS)
            P["MONO"][vam] = codex(tr, va, w, FEATURE_COLUMNS, PHYSICS_COLUMNS, mono=True)
            p = rc_fit(tr, w)
            tr["rc_x"] = rc_simulate(p, *rc_inputs(tr))
            va["rc_x"] = rc_simulate(p, *rc_inputs(va))
            P["RC"][vam] = codex(tr, va, w, FEATURE_COLUMNS + ["rc_x"], PHYSICS_COLUMNS + ["rc_x"])
            print("  %s fold: RC params a=%.3f b=%.3f c=%.3f d=%.3f | rc alone %.4f"
                  % (s, *p, rmse(va.rc_x.values, va.sub_temp.values)), flush=True)
        for k in P:
            P[k] = P[k][order]
        yy = y[order]
        labz = lab.iloc[order].reset_index(drop=True)
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * P["ref"]
            for k in ("RC", "MONO"):
                cand = 0.8 * base + 0.2 * P[k]
                pr, lo, hi, pw = boot(labz[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], yy[g]) / rmse(ref[g], yy[g]) - 1
                print("  %-6s base %3d %-4s | member %.5f -> %.5f | blend %.5f -> %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, bs, k, rmse(P["ref"][g], yy[g]), rmse(P[k][g], yy[g]), rmse(ref[g], yy[g]),
                         rmse(cand[g], yy[g]), 100 * d, lo, hi), flush=True)
                verdict[k] = verdict[k] and d < 0 and (s != "DIAG10" or hi < 0)
    print("\nPRE-SET RULE VERDICT:", {k: ("ADOPT" if v else "REJECT") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
