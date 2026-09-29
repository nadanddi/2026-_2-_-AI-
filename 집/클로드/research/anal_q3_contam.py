# -*- coding: utf-8 -*-
"""Analysis Q3: why does down-weighting contaminated rows help the cold
extrapolation but "hurt" the warm geometry folds?

cleanw_v6.py: weighting only the 167 input-flagged training rows at 0.2
(eda_forensic_11 V1,V3-V7) improves the calibrated EXTRAP fold 0.9090 ->
0.8815 (paired CI [-0.047,-0.010]) but geometry A/B go 0.7850 -> 0.7907 and
0.7434 -> 0.7489.

Hypothesis: the geometry folds SCORE contaminated rows too (1.74% of training
rows are flagged vs 0.56% of test rows).  A model that stopped fitting
corrupted inputs will miss corrupted validation rows, a penalty the test
barely contains.  Test: re-score both models on rows away from any flag
(flagged rows +-3 h removed from scoring only).

Also profiles the flagged rows (which rules, which days/temperatures, what
the label does there) to understand the restoration.

Saves the out-of-fold predictions of the round-3 temperature configuration
(plain) and the down-weighted variant for later analyses:
    local/oof_temp_r3.npz   keys: row_id, EXT10, geomA, geomB (plain) and
                            *_w02 (down-weighted), each (9600,) with NaN where
                            the row was not held out in that fold set.

Run:  cd research && PYTHONPATH="" <python> -u anal_q3_contam.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import THRESH
from screen_v6 import temp_members, collect, boot
from cleanw_v6 import weights


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    y = lab.sub_temp.values

    W02 = weights(lab, 0, 0.2)
    near = weights(lab, 3, 0.0) < 1          # flagged rows +-3 h (scoring mask)
    flags = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv")
    fl = lab[["row_id"]].merge(flags, on="row_id", how="left")

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    sets = [("EXT10", ext), ("geomA", folds("A")), ("geomB", folds("B"))]

    save = {"row_id": lab.row_id.values}
    for sname, fds in sets:
        for tag, W in (("", None), ("_w02", W02)):
            M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, None if W is None else W[m]))
            save[sname + tag] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        print("  %s done" % sname, flush=True)
    np.savez(env.LOCAL + "/oof_temp_r3.npz", **save)
    print("saved local/oof_temp_r3.npz")

    print("\n== re-scoring away from contaminated rows ==")
    print("%-6s | %-26s | %-26s | %s" % ("", "all scored rows", "rows away from flags (+-3h)", "flag-near rows only"))
    for sname, _ in sets:
        a, b = save[sname], save[sname + "_w02"]
        g = ~np.isnan(a)
        parts = []
        for m in (g, g & ~near, g & near):
            parts.append("%.4f -> %.4f (%+5.1f%%) n=%d" % (rmse(a[m], y[m]), rmse(b[m], y[m]),
                         100 * (rmse(b[m], y[m]) / rmse(a[m], y[m]) - 1), int(m.sum())))
        print("%-6s | %s" % (sname, " | ".join(parts)))
    for sname, _ in sets[1:]:
        a, b = save[sname], save[sname + "_w02"]
        g = ~np.isnan(a) & ~near
        sub = lab[g].reset_index(drop=True)
        from feat_lib import paired_block_boot
        pr, lo, hi, pw = paired_block_boot(sub, "sub_temp", a[g], b[g], n_boot=2000, seed=0, level="row")
        print("  paired %s, clean rows only: %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f" % (sname, pr, lo, hi, pw))

    print("\n== profile of the 167 flagged training rows ==")
    fr = fl.Vany.fillna(False).astype(bool).values
    print("by rule:", {k: int(fl[k].fillna(False).astype(bool).sum()) for k in ["V1", "V3", "V4", "V5", "V6", "V7"]})
    print("by farm:", lab[fr].farm.value_counts().to_dict())
    days = lab[fr].groupby(["farm", "day"]).size().sort_values(ascending=False)
    print("days with most flagged rows:", days.head(8).to_dict())
    d = lab.assign(gap=lab.sub_temp - lab.ph_in_temp_3)
    print("slab - ewm3(air): flagged %.2f vs others %.2f" % (d.gap[fr].mean(), d.gap[~fr].mean()))
    print("in_temp mean: flagged %.2f vs others %.2f | ewm3 mean: flagged %.2f vs others %.2f"
          % (lab.in_temp[fr].mean(), lab.in_temp[~fr].mean(), lab.ph_in_temp_3[fr].mean(), lab.ph_in_temp_3[~fr].mean()))
    print("share of flagged rows with ewm3 < 10 C: %.1f%% (all rows %.1f%%)"
          % (100 * float((lab.ph_in_temp_3[fr] < 10).mean()), 100 * float((lab.ph_in_temp_3 < 10).mean())))


if __name__ == "__main__":
    main()
