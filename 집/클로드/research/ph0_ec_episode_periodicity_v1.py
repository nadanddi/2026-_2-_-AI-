# -*- coding: utf-8 -*-
"""PH0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
User material (domain_2026-10-03): drainage EC rises in the reproductive stage
(Ca/K uptake, H+ release) and falls in the vegetative stage.  Strawberry trusses
come in cycles, so high-EC episodes per 동 might RECUR with a period.
Data: pass 1 only (record day < 179; one calendar loop), objective calendar date
index = record order with pair seconds sharing their date (C6.205).  동: pair role,
singletons by the input classifier (local/st_dong_assign_v1.csv).  EC label day
means, lock days excluded.  Per farm x 동: series over date index (linear
interpolation over missing dates), detrended by a centred 31-date rolling median.
Reports: high (>= 1.0) episode start dates and gaps between starts; ACF of the
detrended series at lags 5..60.
Reading (fixed): periodic clue if, in BOTH farms for 동 B, the detrended ACF has a
local maximum >= .30 at some lag in 15..60 dates and the two farms' peak lags
differ by <= 7 dates."""
import env  # noqa: F401
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
lock = {(z["farm"], int(z["day"])) for z in json.loads(open(dc5.p3.LOCK, encoding="utf-8").read())["selected"]}
D = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
EC = Y.groupby(["farm", "day"]).sub_ec.mean()
peaks = {}
for f in ("F13", "F47"):
    G = D[(D.farm == f) & (D.day < 179)].sort_values("day").copy()
    G["date"] = np.cumsum(G.role != "second") - 1
    G["ec"] = [EC.get((f, d), np.nan) if (f, d) not in lock else np.nan for d in G.day]
    print("\n%s pass 1: records %d, dates %d, labelled %d" % (f, len(G), G.date.nunique(), G.ec.notna().sum()))
    for dg in ("A", "B"):
        H = G[G.dong == dg].groupby("date").ec.mean()
        s = H.reindex(range(G.date.min(), G.date.max() + 1)).interpolate(limit_area="inside")
        hi = (s >= 1.0)
        starts = [int(t) for t in s.index[1:] if hi.get(t, False) and not hi.get(t - 1, False)]
        z = s - s.rolling(31, center=True, min_periods=10).median()
        z = z.dropna()
        acf = {L: z.autocorr(L) for L in range(5, 61)}
        loc = [L for L in range(16, 60) if acf[L] >= acf[L - 1] and acf[L] >= acf[L + 1]]
        best = max(loc, key=lambda L: acf[L]) if loc else None
        print("  %s: obs %d, high starts at dates %s, gaps %s" % (dg, H.notna().sum(), starts, list(np.diff(starts))))
        print("     detrended ACF: " + " ".join("L%d %.2f" % (L, acf[L]) for L in range(5, 61, 5)))
        print("     best local max in 15..60: %s" % (("L%d %.2f" % (best, acf[best])) if best else "none"))
        if dg == "B":
            peaks[f] = (best, acf[best] if best else np.nan)
ok = all(p[0] and p[1] >= .30 for p in peaks.values()) and abs(peaks["F13"][0] - peaks["F47"][0]) <= 7 if all(p[0] for p in peaks.values()) else False
print("\nPH0 periodic clue:", ok, peaks)
