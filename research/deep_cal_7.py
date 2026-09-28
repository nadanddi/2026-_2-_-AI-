# -*- coding: utf-8 -*-
"""Relabel fuzzy date groups in first-appearance order and print both records."""
import env  # noqa
import numpy as np, pandas as pd
k = pd.read_csv(env.LOCAL + "/deep_cal_6_days.csv")
# order by median over members of (day) restricted to days<=178 if any
def okey(x):
    fp = x[x.day <= 178]
    return fp.day.mean() if len(fp) else 1000 + x.day.mean()
order = k.groupby("g").apply(okey).sort_values()
rid = {g: i for i, g in enumerate(order.index)}
k["date"] = k.g.map(rid)
k.to_csv(env.LOCAL + "/deep_cal_7_days.csv", index=False)
for f in ["F13", "F47"]:
    s = k[k.farm == f].sort_values("day")
    line = ["%d:%d%s" % (d, dt, "*" if te else "") for d, dt, te in zip(s.day, s.date, s.is_test)]
    print(f)
    for i in range(0, len(line), 16):
        print("  " + " ".join(line[i:i + 16]))
c = k.groupby(["date", "farm"]).size().unstack(fill_value=0)
print(c.value_counts())
print("dates with any test day:", k[k.is_test].date.nunique(), " dates only in days>178:", (k.groupby("date").day.min() > 178).sum())
