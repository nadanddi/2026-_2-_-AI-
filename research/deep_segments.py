# -*- coding: utf-8 -*-
"""Where did our three scored submissions actually move, and where is the test
set risky?  Uses only our own predictions and the test INPUTS.

Scores: round 1 0.7450/0.2442, round 2 0.6666/0.2287, round 3 0.5624/0.2055.
Nothing here back-solves test labels from the scores (rule 5 forbids using
evaluation labels); segments are described by inputs, and the score changes
are only read qualitatively next to where the predictions moved.

Run:  cd research && PYTHONPATH="" <python> deep_segments.py
"""
import os

import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from common import TARGET_FARMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tX, ty, sX = common.load_raw()
    subs = {"r1": pd.read_csv(os.path.join(ROOT, "Claude", "submissions", "submission_01_scored.csv")),
            "r2": pd.read_csv(os.path.join(ROOT, "research", "submissions", "submission_03.csv")),
            "r3": pd.read_csv(os.path.join(ROOT, "research", "submissions", "submission_04.csv"))}
    a = pd.concat([tX, sX]).query("farm in @TARGET_FARMS").sort_values(["farm", "t"])
    a["ewm3"] = a.groupby("farm").in_temp.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    te = sX[["row_id", "farm", "day", "hour"]].merge(a[["row_id", "in_temp", "ewm3", "act_heating",
                                                       "act_thermal", "out_temp"]], on="row_id")
    for k, s in subs.items():
        te = te.merge(s.rename(columns={"sub_temp": "t_" + k, "sub_ec": "e_" + k}), on="row_id")

    # test blocks = contiguous runs of test days per farm
    te["block"] = ""
    for f in TARGET_FARMS:
        days = sorted(te[te.farm == f].day.unique())
        b, prev = 0, None
        for d in days:
            if prev is not None and d != prev + 1:
                b += 1
            te.loc[(te.farm == f) & (te.day == d), "block"] = "%s-B%d(%d~)" % (f, b + 1, d)
            prev = d
    lab = ty.merge(tX[["row_id", "in_temp"]], on="row_id").query("farm in @TARGET_FARMS")
    lo1 = float(np.percentile(a.loc[a.row_id.isin(lab.row_id), "ewm3"].dropna(), 1))
    te["cold"] = te.ewm3 < 9
    te["night"] = (te.hour >= 20) | (te.hour <= 6)
    te["anom"] = ((te.farm == "F13") & te.day.isin([204, 207, 208])) | ((te.farm == "F47") & (te.day == 223))

    def seg_table(title, key):
        print("\n=== %s ===" % title)
        print("%-18s %5s | %9s %9s | %8s %8s | %8s %8s | %6s" %
              ("segment", "n", "ewm3 avg", "heat100%", "dT r1>r2", "dT r2>r3", "dE r1>r2", "dE r2>r3", "SSq%"))
        tot = float(((te.t_r3 - te.t_r2) ** 2).sum())
        for k, g in te.groupby(key):
            print("%-18s %5d | %9.2f %9.2f | %+8.3f %+8.3f | %+8.3f %+8.3f | %5.1f%%" %
                  (str(k)[:18], len(g), g.ewm3.mean(), float((g.act_heating >= 99).mean()),
                   (g.t_r2 - g.t_r1).mean(), (g.t_r3 - g.t_r2).mean(),
                   (g.e_r2 - g.e_r1).mean(), (g.e_r3 - g.e_r2).mean(),
                   100 * float(((g.t_r3 - g.t_r2) ** 2).sum()) / tot))

    print("labelled-row ewm3 1st percentile: %.2f | test rows below it: %.1f%%"
          % (lo1, 100 * float((te.ewm3 < lo1).mean())))
    seg_table("test block", "block")
    te["tband"] = pd.cut(te.ewm3, [-5, 6, 8, 10, 12, 15, 40], labels=["<6", "6-8", "8-10", "10-12", "12-15", ">=15"])
    seg_table("ewm3 temperature band", "tband")
    te["period"] = np.where(te.night, "night 20-06", "day 07-19")
    seg_table("day / night", "period")
    seg_table("anomalous test days (F13 204/207/208, F47 223)", "anom")

    print("\n=== temperature: where round 3 still sits vs the training-label cold relation ===")
    print("training slab - ewm3 by band: >=10 -0.6..-0.8 | 8-10 -0.52 | 6-8 -1.17 | <6 -1.72")
    for k, g in te.groupby("tband", observed=True):
        print("  %-6s n=%4d  r2 pred-ewm3 %+.2f | r3 pred-ewm3 %+.2f" %
              (k, len(g), (g.t_r2 - g.ewm3).mean(), (g.t_r3 - g.ewm3).mean()))


if __name__ == "__main__":
    main()
