# -*- coding: utf-8 -*-
"""Temperature: clean the injected indoor-sensor noise at the INPUT level.

Round 5 (temp 0.5624 -> 0.5456) down-weighted rough training days.  The
noise sits in in_temp / in_hum / in_co2 only (catalog 6.18), and the
temperature features lean on the indoor history (lags, EWMs, physics
baseline), so cleaning the inputs may help beyond down-weighting - and the
rough days' labels could then be used at full weight.

World S: on rough TRAINING days of F13/F47 (training noise score top
quarter, train_flags_v6, training inputs only) in_temp / in_hum / in_co2 are
replaced by a causal EWM (halflife H h) of the series; test rows untouched.
All features are then rebuilt from these inputs by the shipped builders, so
features of later rows (and of test rows) see the cleaned history.  The EWM
looks backwards only; no label and no later input is involved.

Variants
  R5        raw world, round-5 weights (60 restored rows + 96 noisy days at 0.2)
  S_F60ND   world S, same weights
  S_F60     world S, only the 60 restored rows at 0.2 (rough days at full weight)
Folds as eval_v6: diagnostic (non-overlapping) + extrapolation 8/10/12 C.
Scores: all held-out rows; clean rows (not rough, not within 3 h of a flag).

Run:  cd research && PYTHONPATH="" <python> -u temp_denoise_v1.py
"""
import env  # noqa: F401
import numpy as np

import common
import harness
from common import rmse, TARGET_FARMS
from harness import views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF

ORIG = common.load_raw
NOISY = ["in_temp", "in_hum", "in_co2"]
H = 1.0


def smoothed_loader(rough_set):
    def ld():
        tX, ty, sX = ORIG()
        tX = tX.copy()
        both = np.concatenate([tX[["farm", "t"] + NOISY].assign(src=0, i=np.arange(len(tX))).values,
                               sX[["farm", "t"] + NOISY].assign(src=1, i=np.arange(len(sX))).values])
        import pandas as pd
        B = pd.DataFrame(both, columns=["farm", "t"] + NOISY + ["src", "i"])
        B[NOISY + ["t"]] = B[NOISY + ["t"]].astype(float)
        B = B.sort_values(["farm", "t"])
        for c in NOISY:
            B[c + "_s"] = B.groupby("farm")[c].transform(lambda s: s.ewm(halflife=H, ignore_na=True).mean())
        B = B[B.src == 0].sort_values("i")
        m = np.array([(f, d) in rough_set for f, d in zip(tX.farm.values, tX.day.values)])
        for c in NOISY:
            tX.loc[m, c] = B[c + "_s"].values[m]
        print("  world S: %d training rows cleaned" % m.sum(), flush=True)
        return tX, ty, sX
    return ld


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


def main():
    nd = TF.noisy_days()
    rough_set = set(map(tuple, nd[nd.noise_score >= nd.noise_score.quantile(0.75)][["farm", "day"]].values))

    labR, ct, phc = build_world()
    W60ND = TF.row_weights(labR, 0.2, w_noisy=0.2)
    W60 = TF.row_weights(labR, 0.2)
    rough = np.array([(f, d) in rough_set for f, d in zip(labR.farm, labR.day)])
    clean = (TF.row_weights(labR, 0.0, radius=3) >= 1) & ~rough
    dmin = labR.groupby(["farm", "day"]).ph_in_temp_3.min()

    common.load_raw = smoothed_loader(rough_set)
    try:
        labS, ctS, phcS = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    assert (labS.row_id.values == labR.row_id.values).all() and ctS == ct and phcS == phc
    y = labR.sub_temp.values
    assert np.allclose(labS.sub_temp.values, y)
    print("features: %d | rough rows %d | clean scoring rows %d" % (len(ct), rough.sum(), clean.sum()))

    sets = [("DIAG10", diag_folds(labR))]
    for th in (8.0, 10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    V = {"R5": (labR, W60ND), "S_F60ND": (labS, W60ND), "S_F60": (labS, W60)}
    oof = {}
    for vn, (lab, w) in V.items():
        for s, fds in sets:
            M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, w[m]))
            oof[(vn, s)] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        print("  %s done" % vn, flush=True)
    np.savez(env.LOCAL + "/temp_denoise_v1_oof.npz", row_id=labR.row_id.values,
             **{"%s__%s" % k: val for k, val in oof.items()})

    print("\n%-8s" % "" + "".join(" | %-8s %-8s" % (s + ":all", "clean") for s, _ in sets))
    for vn in V:
        line = "%-8s" % vn
        for s, _ in sets:
            o = oof[(vn, s)]
            g = ~np.isnan(o)
            line += " | %8.4f %8.4f" % (rmse(o[g], y[g]), rmse(o[g & clean], y[g & clean]))
        print(line)
    print("\nincrements vs R5 (paired block bootstrap, 95% CI, P(worse)):")
    for b in ("S_F60ND", "S_F60"):
        for s, _ in sets:
            for nm, msk in (("all", np.ones(len(y), bool)), ("clean", clean)):
                oa, ob = oof[("R5", s)], oof[(b, s)]
                g = ~np.isnan(oa) & msk
                pr, lo, hi, pw = boot(labR[g].reset_index(drop=True), "sub_temp", oa[g], ob[g])
                print("  R5 -> %-8s %-6s %-5s %+.4f [%+.4f, %+.4f] P(worse)=%.3f" % (b, s, nm, pr, lo, hi, pw))


if __name__ == "__main__":
    main()
