# -*- coding: utf-8 -*-
"""Round-5 temperature candidates, evaluated the way the audits asked
(research/감사결과_2026-09-26.md):

  * non-overlapping folds: the diagnostic round-robin folds (every labelled
    day held out exactly once, 5-day chunks) -- geometry A/B overlap (58% of
    held-out days twice) and keep only the last prediction;
  * several extrapolation thresholds (8, 10, 12 C), never one;
  * score ALL held-out rows, and clean rows (new flags +-3 h removed) as a
    second view; no "test-like" day filtering;
  * report effects as increments with paired block-bootstrap CIs.

Variants (training-row weights from train_flags_v6.py, training data only):
  plain   round-3 configuration
  F60     60 audited restored rows (V1, backward V5, non-midnight V4) at 0.2
  F60ND   F60 + 96 noisy training days (CO2-difference autocorrelation
          collapse, cold days exempt) at 0.2

Run:  cd research && PYTHONPATH="" <python> -u eval_v6.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse, TARGET_FARMS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    y = lab.sub_temp.values

    W = {"plain": None, "F60": TF.row_weights(lab, 0.2), "F60ND": TF.row_weights(lab, 0.2, w_noisy=0.2)}
    clean = TF.row_weights(lab, 0.0, radius=3) >= 1
    print("down-weighted rows: F60 %d | F60ND %d | clean-scoring rows %d of %d"
          % (int((W["F60"] < 1).sum()), int((W["F60ND"] < 1).sum()), int(clean.sum()), len(lab)))

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (8.0, 10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    oof = {}
    for vn, w in W.items():
        for s, fds in sets:
            M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, None if w is None else w[m]))
            oof[(vn, s)] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        print("  %s done" % vn, flush=True)
    np.savez(env.LOCAL + "/eval_v6_oof.npz", row_id=lab.row_id.values,
             **{"%s__%s" % k: val for k, val in oof.items()})

    print("\n%-7s" % "" + "".join(" | %-8s %-8s" % (s + ":all", "clean") for s, _ in sets))
    for vn in W:
        line = "%-7s" % vn
        for s, _ in sets:
            o = oof[(vn, s)]
            g = ~np.isnan(o)
            line += " | %8.4f %8.4f" % (rmse(o[g], y[g]), rmse(o[g & clean], y[g & clean]))
        print(line)

    print("\nincrements (paired block bootstrap, 95% CI, P(worse)):")
    for a, b in (("plain", "F60"), ("F60", "F60ND"), ("plain", "F60ND")):
        for s, _ in sets:
            for nm, msk in (("all", np.ones(len(lab), bool)), ("clean", clean)):
                oa, ob = oof[(a, s)], oof[(b, s)]
                g = ~np.isnan(oa) & msk
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", oa[g], ob[g])
                print("  %-5s -> %-6s %-6s %-5s %+.4f [%+.4f, %+.4f] P(worse)=%.3f"
                      % (a, b, s, nm, pr, lo, hi, pw))


if __name__ == "__main__":
    main()
