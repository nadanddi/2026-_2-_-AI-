# -*- coding: utf-8 -*-
"""Use the 49 other greenhouses for TEMPERATURE (anal_f1_otherfarms.py):
they carry only in_temp/in_hum/in_co2/in_rad (integer resolution) and integer
sub_temp labels, no EC labels - but 10-35% of their hours are below 10 C,
exactly where F13/F47 lack training data and the test period lies.

Transfer member: LightGBM on features built from in_temp / in_hum / in_co2
only (causal, per greenhouse: current, lags, EWMs, 24-h rolling min/max,
differences, hour), target sub_temp, with indicators for F13 and F47 so the
greenhouse-specific substrate-air offset is not averaged away.
Training rows = all other greenhouses except F32 (root-zone heating, +7.2 C)
+ the F13/F47 training-fold rows.  F13/F47 features in the MASK world (test_X
inputs NaN), as for every current member.

Variants (fixed):  T1 = as above (other farms weight 1)
                   T2 = other farms weight 0.3 (F13/F47 dominate the fit)
Blend (as the TabPFN protocol): 0.7*MASK base + 0.2*Codex + 0.1*member
vs 0.8*MASK base + 0.2*Codex.  Validators DIAG10 / EXT10 / EXT12, base seeds
7/101 x member seeds 1/2.
Rule (2026-09-27): same direction in every seed x validator, DIAG10 CI
excluding 0 at level 1-0.05/2 (two variants).  Also reported: member alone.

Run:  cd research && PYTHONPATH="" <python> -u temp_transfer_v1.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader

BASE = ["in_temp", "in_hum", "in_co2"]
VARIANTS = {"T1": 1.0, "T2": 0.3}


def feats(df):
    out = []
    for f, g in df.sort_values(["farm", "t"]).groupby("farm", sort=False):
        g = g.set_index("t").reindex(range(int(g.t.min()), int(g.t.max()) + 1))
        d = pd.DataFrame({"row_id": g.row_id, "farm": f}, index=g.index)
        d["hr_sin"], d["hr_cos"] = np.sin(2 * np.pi * (g.index % 24) / 24), np.cos(2 * np.pi * (g.index % 24) / 24)
        for c in BASE:
            s = g[c].astype(float)
            d[c] = s
            for L in (1, 2, 3, 6, 12, 24):
                d["%s_l%d" % (c, L)] = s.shift(L)
            for h in (1, 3, 6, 12, 24):
                d["%s_e%d" % (c, h)] = s.ewm(halflife=h, ignore_na=True).mean()
            d["%s_d1" % c] = s.diff()
            d["%s_min24" % c] = s.rolling(24, min_periods=6).min()
            d["%s_max24" % c] = s.rolling(24, min_periods=6).max()
        d["is_F13"], d["is_F47"] = float(f == "F13"), float(f == "F47")
        out.append(d.dropna(subset=["row_id"]))
    return pd.concat(out, ignore_index=True)


def main():
    tX, ty, _ = common.load_raw()
    mX, _, msX = masked_loader()
    tgt = feats(pd.concat([mX[mX.farm.isin(TARGET_FARMS)], msX], ignore_index=True)).set_index("row_id")
    oth_x = tX[~tX.farm.isin(list(TARGET_FARMS) + ["F32"])]
    oth = feats(oth_x).merge(ty[["row_id", "sub_temp"]], on="row_id").dropna(subset=["sub_temp"])
    cols = [c for c in oth.columns if c not in ("row_id", "farm", "sub_temp")]
    print("other-farm rows %d, features %d" % (len(oth), len(cols)), flush=True)

    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].copy()
    assert (lab.row_id.values == z["row_id"]).all()
    X = tgt.loc[lab.row_id, cols].values
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    ph = F4.phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    Xo, yo = oth[cols].values, oth.sub_temp.values

    verdict = {v: True for v in VARIANTS}
    for s, fds in sets:
        cx = z["%s__CODEX__726" % s]
        for v, wo in VARIANTS.items():
            mem = {}
            for ms in (1, 2):
                o = np.full(len(lab), np.nan)
                for fd in fds:
                    trm, vam = split_mask(lab, fd)
                    m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                                      verbosity=-1, n_jobs=6, random_state=ms)
                    m.fit(np.vstack([Xo, X[trm]]), np.concatenate([yo, y[trm]]),
                          sample_weight=np.concatenate([np.full(len(yo), wo), w[trm]]))
                    o[vam] = m.predict(X[vam])
                mem[ms] = o
            for bs in (7, 101):
                base = z["%s__MASK__%d" % (s, bs)]
                g = ~np.isnan(base)
                ref = 0.8 * base + 0.2 * cx
                for ms in (1, 2):
                    cand = 0.7 * base + 0.2 * cx + 0.1 * mem[ms]
                    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                    d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                    print("%-6s %s base %3d member %d | alone %.5f | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                          % (s, v, bs, ms, rmse(mem[ms][g], y[g]), rmse(ref[g], y[g]), rmse(cand[g], y[g]),
                             100 * d, lo, hi), flush=True)
                    verdict[v] = verdict[v] and d < 0 and (s != "DIAG10" or hi < 0)
    # boot() reports a 95% CI; the Bonferroni level for 2 variants is 97.5% - flagged in the report
    for v, ok in verdict.items():
        print("%s PRE-SET RULE VERDICT (95%% CI; check 97.5%%): %s" % (v, "ADOPT" if ok else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
