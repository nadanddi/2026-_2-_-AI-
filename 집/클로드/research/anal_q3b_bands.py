# -*- coding: utf-8 -*-
"""Q3b: where, by temperature band, does down-weighting the 167 contaminated
rows change the EXTRAP-fold predictions?  Round 4 lowered only the coldest
test rows and got worse; if this change also just lowers the coldest rows it
would repeat that.  Uses local/oof_temp_r3.npz (anal_q3_contam.py).

Run:  cd research && PYTHONPATH="" <python> anal_q3b_bands.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from harness import load
import features_v4 as F4
from cleanw_v6 import weights
from common import rmse

panel, lab0, _ = load()
lab = lab0.merge(F4.phys_features(), on="row_id", how="left")
z = np.load(env.LOCAL + "/oof_temp_r3.npz", allow_pickle=True)
assert (z["row_id"] == lab.row_id.values).all()
y = lab.sub_temp.values
clean = weights(lab, 3, 0.0) >= 1
for s in ("EXT10", "geomA"):
    a, b = z[s], z[s + "_w02"]
    g = ~np.isnan(a) & clean
    d = pd.DataFrame({"x": lab.ph_in_temp_3.values[g], "y": y[g], "a": a[g], "b": b[g]})
    d["band"] = pd.cut(d.x, [-5, 6, 8, 10, 12, 15, 40], labels=["<6", "6-8", "8-10", "10-12", "12-15", ">=15"])
    print("\n== %s, clean rows ==" % s)
    print("%-6s %5s | %8s %8s | %8s %8s | %9s" % ("band", "n", "rmse r3", "rmse w02", "bias r3", "bias w02", "mean move"))
    for k, gg in d.groupby("band", observed=True):
        print("%-6s %5d | %8.3f %8.3f | %+8.3f %+8.3f | %+9.3f"
              % (k, len(gg), rmse(gg.a, gg.y), rmse(gg.b, gg.y), (gg.a - gg.y).mean(), (gg.b - gg.y).mean(), (gg.b - gg.a).mean()))
