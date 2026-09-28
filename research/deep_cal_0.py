# -*- coding: utf-8 -*-
import env  # noqa
import numpy as np, pandas as pd
import common
tX, ty, sX = common.load_raw()
for n, d in [("tX", tX), ("ty", ty), ("sX", sX)]:
    print(n, d.shape, list(d.columns)[:25])
for f in ["F13", "F47"]:
    a = tX[tX.farm == f]; b = sX[sX.farm == f]
    print(f, "train days", a.day.min(), a.day.max(), a.day.nunique(), "test days", sorted(b.day.unique()))
    lab = ty[(ty.farm == f)]
    print("  labelled days", lab.day.nunique(), lab.day.min(), lab.day.max())
    miss = sorted(set(range(a.day.min(), max(a.day.max(), b.day.max())+1)) - set(a.day) - set(b.day))
    print("  missing days", miss[:50], len(miss))
