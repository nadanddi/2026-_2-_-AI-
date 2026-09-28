# -*- coding: utf-8 -*-
"""sub_temp: give the GBM day-restarted filters next to the continuous ones.

Finding (struct_resid.py / segment diagnostics):
  * the label jumps at midnight (|d sub_temp| 23->0 h is 1.40 / 1.68 vs
    ~0.56 at other hours) while the air jumps only 1.2x, and the slab jump
    tracks the air jump (r 0.79 / 0.75): the series looks stitched per day,
    with the inputs carrying the stitch.
  * hour 0 is the worst hour (RMSE 1.14 / 1.06 vs ~0.75 elsewhere).
  * a first-order filter restarted every midnight helps hour 0 but hurts the
    rest, so days are not fully independent.

So offer BOTH memories and let the trees choose.  Every added column uses
only same-greenhouse inputs at or before the row's own hour: an EWM restarted
at 00 of the row's own day is causal.

Run:  cd research && PYTHONPATH="" <python> -u struct_segment.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb

import common
from common import TARGET_FARMS, split_mask, rmse
from harness import load, views, score, folds
import feat_temp74 as T74
import feat_new
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)


def LGBH(s):
    return lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)


def seg_features():
    tX, ty, sX = common.load_raw()
    cols = ["row_id", "farm", "t", "in_temp", "out_temp", "out_rad",
            "act_shade", "act_thermal", "act_heating"]
    a = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    a = a[a.farm.isin(TARGET_FARMS)]
    out = []
    for f, g in a.groupby("farm"):
        g = g.sort_values("t").set_index("t")
        g = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        day = g.index // 24
        rad = g.out_rad * (g.act_shade.fillna(100) / 100) * (g.act_thermal.fillna(100) / 100)
        src = {"in_temp": g.in_temp, "rad": rad, "heat": g.act_heating,
               "out_temp": g.out_temp}
        d = pd.DataFrame({"row_id": g.row_id.values}, index=g.index)
        for c, s in src.items():
            for hl in (1, 2, 4, 8):
                d["seg_%s_%d" % (c, hl)] = s.groupby(day).transform(
                    lambda x, hl=hl: x.ewm(halflife=hl, ignore_na=True).mean()).values
        # first observation of the day: the stitch anchor
        d["seg_in_temp_h0"] = g.in_temp.groupby(day).transform("first").values
        d["seg_in_temp_dev_h0"] = g.in_temp.values - d["seg_in_temp_h0"].values
        out.append(d.dropna(subset=["row_id"]))
    return pd.concat(out, ignore_index=True)


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg = seg_features()
    lab = lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
    base = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    segc = [c for c in sg.columns if c != "row_id"]
    only_in = [c for c in segc if c.startswith("seg_in_temp")]
    cands = [("93f base", base),
             ("93f + seg in_temp (6)", base + only_in),
             ("93f + seg all (18)", base + segc)]
    y = lab.sub_temp.values
    h0 = lab.hour.values == 0

    oof = {}
    print("%-26s %8s %8s | %8s %8s" % ("", "A", "A 0시", "B", "B 0시"))
    for nm, cols in cands:
        row = []
        for kind in ("A", "B"):
            (r, sd, per), o = score(lab, "sub_temp", cols, LGBH, kind=kind,
                                    seeds=(7,), return_oof=True)
            oof[(nm, kind)] = o
            g = ~np.isnan(o)
            row += [r, rmse(o[g & h0], y[g & h0])]
        print("%-26s %8.4f %8.4f | %8.4f %8.4f" % (nm, *row))

    for nm, _ in cands[1:]:
        print("\n== paired: 93f base vs %s ==" % nm)
        for kind in ("A", "B"):
            a, b = oof[("93f base", kind)], oof[(nm, kind)]
            g = ~np.isnan(a) & ~np.isnan(b)
            idx = [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]
            d = [rmse(b[i], y[i]) - rmse(a[i], y[i]) for i in idx]
            sub = lab[g].reset_index(drop=True)
            pr, lo, hi, pw = paired_block_boot(sub, "sub_temp", a[g], b[g],
                                               n_boot=2000, seed=0, level="row")
            print("  %s | per-fold %s (%d/5) | %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (kind, " ".join("%+.3f" % x for x in d),
                     sum(x < 0 for x in d), pr, lo, hi, pw))


if __name__ == "__main__":
    main()
