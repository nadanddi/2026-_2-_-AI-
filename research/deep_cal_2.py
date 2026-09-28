# -*- coding: utf-8 -*-
"""Map each (farm, day) to a date id and print sequences."""
import env  # noqa
import numpy as np, pandas as pd
k = pd.read_csv(env.LOCAL + "/deep_cal_1_keys.csv")
k = k[k.farm.isin(["F13", "F47"])].copy()
# date id ordered by first appearance (min day over both farms)
first = k.groupby("wk").day.min().sort_values()
did = {w: i for i, w in enumerate(first.index)}
k["date"] = k.wk.map(did)
k.to_csv(env.LOCAL + "/deep_cal_2_days.csv", index=False)
for f in ["F13", "F47"]:
    s = k[k.farm == f].sort_values("day")
    print(f)
    line = []
    for d, dt, te in zip(s.day, s.date, s.is_test):
        line.append("%d:%d%s" % (d, dt, "*" if te else ""))
    for i in range(0, len(line), 16):
        print("  " + " ".join(line[i:i+16]))
g = k.groupby("date").apply(lambda x: "".join(sorted(x.farm.str[1:].tolist())))
print(g.value_counts())
