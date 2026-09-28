# -*- coding: utf-8 -*-
"""How does substrate - air behave when the air gets very cold?  Per
greenhouse (all 51 with labels), mean of (sub_temp - in_temp) by in_temp
band, CENTRED on the greenhouse's own 10-15 C offset, so only the SHAPE is
compared (levels differ by heating system, catalog 6.45).  Also the same for
the 24-h mean air (slow) vs current air.  Analysis only.
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import common

tX, ty, sX = common.load_raw()
a = tX.merge(ty[["row_id", "sub_temp"]], on="row_id").dropna(subset=["sub_temp", "in_temp"])
a = a.sort_values(["farm", "t"])
a["air24"] = a.groupby("farm").in_temp.transform(lambda s: s.rolling(24, min_periods=12).mean())
a["off"] = a.sub_temp - a.in_temp
bands = [-99, 4, 6, 8, 10, 15, 99]
a["band"] = pd.cut(a.in_temp, bands)
rows = []
for f, g in a.groupby("farm"):
    ref = g.loc[(g.in_temp > 10) & (g.in_temp <= 15), "off"].mean()
    m = g.groupby("band", observed=False).off.agg(["mean", "size"])
    r = {"farm": f, "n<6": int((g.in_temp <= 6).sum())}
    for b, v in m.iterrows():
        r[str(b)] = (v["mean"] - ref) if v["size"] >= 30 else np.nan
    rows.append(r)
R = pd.DataFrame(rows).set_index("farm")
cols = [c for c in R.columns if c.startswith("(")]
print("shape of (substrate - air) relative to its 10-15C level, farms with >=30 rows per band")
print(R[["n<6"] + cols].round(2).to_string())
print("\nmedian over farms:", R[cols].median().round(2).to_dict())
print("share of farms where offset at <=6C is ABOVE the 10-15C level: %.2f"
      % (R["(4.0, 6.0]"].dropna() > 0).mean())
print("\nF13/F47 test rows <=6C: %d" % (sX.in_temp <= 6).sum())
