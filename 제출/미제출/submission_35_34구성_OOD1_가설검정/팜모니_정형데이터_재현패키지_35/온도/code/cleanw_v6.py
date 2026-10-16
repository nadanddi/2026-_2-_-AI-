# -*- coding: utf-8 -*-
"""sub_temp: milder versions of the contaminated-row down-weighting.

screen_v6.py: down-weighting the 167 input-flagged training rows AND +-3 h
around them (642 rows, 6.7%) to 0.2 improved the calibrated EXTRAP fold by
2.4% (0.9090 -> 0.8870 with 0.45/0.45/0.10 weights) but hurt the warm
geometry folds (A +1.9%, B +0.5%), so it failed the do-no-harm rule.  The
neighbourhood may simply have been too wide.  Here, at the round-3 member
weights (0.65/0.25/0.10) and at 0.45/0.45/0.10:

  r0_w02   flagged rows only, weight 0.2
  r0_w05   flagged rows only, weight 0.5
  r1_w05   flagged rows +-1 h, weight 0.5
  r3_w05   flagged rows +-3 h, weight 0.5

Run:  cd research && PYTHONPATH="" <python> -u cleanw_v6.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import rmse, USABLE, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import THRESH
from screen_v6 import temp_members, collect, boot


def weights(lab, radius, w):
    f = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv")
    bad = set(f.loc[(f.set == "train") & f.Vany.astype(bool), "row_id"])
    key = lab[["row_id", "farm", "t"]]
    out = np.ones(len(lab))
    for farm, g in key[key.row_id.isin(bad)].groupby("farm"):
        m = (key.farm.values == farm) & (np.abs(key.t.values[:, None] - g.t.values[None, :]) <= radius).any(1)
        out[m] = w
    return out


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

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    sets = [("EXT10", ext), ("geomA", folds("A")), ("geomB", folds("B"))]
    variants = [("plain", None), ("r0_w02", (0, 0.2)), ("r0_w05", (0, 0.5)),
                ("r1_w05", (1, 0.5)), ("r3_w05", (3, 0.5))]

    M = {}
    for nm, spec in variants:
        W = None if spec is None else weights(lab, *spec)
        if W is not None:
            print("%s: %d rows down-weighted" % (nm, int((W < 1).sum())), flush=True)
        for sname, fds in sets:
            M[(nm, sname)] = collect(lab, fds, lambda tr, va, m: temp_members(
                tr, va, ct, phc, None if W is None else W[m]))
        print("  %s done" % nm, flush=True)

    for wr, wg, wn in ((0.65, 0.25, 0.10), (0.45, 0.45, 0.10)):
        print("\n== member weights %.2f/%.2f/%.2f ==" % (wr, wg, wn))
        print("%-8s %9s %9s %9s" % ("", "EXT10", "geomA", "geomB"))
        for nm, _ in variants:
            s = []
            for sname, _ in sets:
                Mm = M[(nm, sname)]
                p = wr * Mm["res"] + wg * Mm["ridge"] + wn * Mm["nys"]
                g = ~np.isnan(p)
                s.append(rmse(p[g], y[g]))
            print("%-8s %9.4f %9.4f %9.4f" % (nm, *s))

    ref = M[("plain", "EXT10")]
    a = 0.65 * ref["res"] + 0.25 * ref["ridge"] + 0.10 * ref["nys"]
    for nm, _ in variants[1:]:
        Mm = M[(nm, "EXT10")]
        b = 0.65 * Mm["res"] + 0.25 * Mm["ridge"] + 0.10 * Mm["nys"]
        pr, lo, hi, pw = boot(lab, "sub_temp", a, b)
        print("paired EXT10 %-7s (0.65/0.25/0.10): %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f" % (nm, pr, lo, hi, pw))


if __name__ == "__main__":
    main()
