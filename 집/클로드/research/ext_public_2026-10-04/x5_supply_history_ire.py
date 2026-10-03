# -*- coding: utf-8 -*-
"""X5 (diagnostic; 2026-10-04 집 클로드).  User hypothesis: high EC follows a period
of insufficient supply or fast consumption.  Public 이레 strawberry (18 farms) has
irrigation (공급량) and drain (배액량) with drain EC.  Within-farm day deviations;
Spearman of today's drain-EC deviation with the PREVIOUS k days' (k=1,3,7) irrigation
sum, drain sum, drain ratio, mean inside temp, and with today's minus previous-day
irrigation; plus: on the farm's top-10% EC days, mean of prior-3-day irrigation
deviation (in farm SD units)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
I = pd.read_csv(os.path.join(OUT, "x0_ire_hourly.csv"), parse_dates=["t"]).rename(columns={"(양액)공급량": "irr", "(양액)배액EC": "dec", "(양액)배액량": "drn", "내부온도": "ti"})
I.loc[I.dec <= 0, "dec"] = np.nan; I["d"] = I.t.dt.normalize()
D = I.groupby(["farm", "d"]).agg(dec=("dec", "mean"), irr=("irr", "sum"), drn=("drn", "sum"), ti=("ti", "mean"), n=("dec", "count")).reset_index()
D = D[D.n >= 6].sort_values(["farm", "d"]).copy()
D["ratio"] = D.drn / D.irr.replace(0, np.nan)
# detrend with a farm-wise 15-day centred rolling median to remove the season trend
for c in ("dec", "irr", "drn", "ratio", "ti"):
    D[c + "_dv"] = D[c] - D.groupby("farm")[c].transform(lambda s: s.rolling(15, center=True, min_periods=5).median())
g = D.groupby("farm")
print("[이레] farm-days %d (season detrended by 15-day rolling median)" % len(D))
for k in (1, 3, 7):
    row = []
    for c in ("irr", "drn", "ratio", "ti"):
        prev = g[c + "_dv"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())
        row.append("%s %+.2f" % (c, spearmanr(D.dec_dv, prev, nan_policy="omit").correlation))
    print("  drain-EC dev vs previous %d-day mean of: %s" % (k, " | ".join(row)))
dirr = g.irr_dv.diff()
print("  drain-EC dev vs today's irrigation change from yesterday: %+.2f" % spearmanr(D.dec_dv, dirr, nan_policy="omit").correlation)
print("  drain-EC dev vs same-day irrigation dev: %+.2f" % spearmanr(D.dec_dv, D.irr_dv, nan_policy="omit").correlation)
D["top"] = D.groupby("farm").dec_dv.transform(lambda s: s >= s.quantile(.9))
for c in ("irr", "drn", "ratio"):
    z = D[c + "_dv"] / D.groupby("farm")[c + "_dv"].transform("std")
    p3 = z.groupby(D.farm).transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    print("  prior-3-day %s (farm SD units): top-10%% EC days %+.2f vs others %+.2f" % (c, p3[D.top].mean(), p3[~D.top].mean()))
