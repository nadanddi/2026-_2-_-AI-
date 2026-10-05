# -*- coding: utf-8 -*-
"""SD1 (diagnostic, fixed before running; 2026-10-05 집 클로드).
Use the DOMAIN score (FZ1 6.321: signs from the user's material, no fitted weights)
    S_low = z(in_hum) + z(shade) + z(thermal) + z(fog) - z(in_temp) - z(in_co2)
to correct the LEVEL of sealed rows on top of SG2.  Sealed days are compressed toward ~1.2
(ND0 / SD0: normal sealed days over-predicted, high ones under-predicted); among sealed days
S_low correlates with EC (Spearman -.33).
Hour-causal: S_low at hour h from the record's hours 0..h means, z per farm per h from
LABELLED training records' 0..h means; sealed row = share of act_vent == 0 over hours 0..h >= .8.
Correction (one form, fixed): on sealed rows, pred = SG2 + (b0 + b1 * S_low_h), where (b0, b1)
= OLS of the SG2 residual (label - SG2) on S_low_h over sealed rows of the OTHER DIAG10 folds
(leave-fold-out; the residuals there are out-of-fold), pooled over farms.
Data: DIAG10 R3S seed-mean OOF rows with SG2 anchors (local/hk0_rows_v1.csv).
Clue (fixed): beats SG2 on all rows AND pass-2 rows, normal days (label day mean < 1) not
worse by > 1 %, sealed rows better in BOTH farms, farm x 5-day cluster bootstrap P(worse) on
all rows < .0125.  Also reported: b1 per fold (sign stability), pass-2 high / normal."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X.sort_values(["farm", "day", "hour"])
DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]; SG = np.array([1, 1, 1, 1, -1, -1.])
g = X.groupby(["farm", "day"])
for v in DV:
    X["cm_" + v] = g[v].transform(lambda z: z.expanding().mean())
X["seal"] = g.act_vent.transform(lambda z: z.eq(0).astype(float).expanding().mean())
for f in ("F13", "F47"):
    for h in range(24):
        m = (X.farm == f) & (X.hour == h)
        Z = (X.loc[m, ["cm_" + v for v in DV]] - X.loc[m, ["cm_" + v for v in DV]].mean()) / X.loc[m, ["cm_" + v for v in DV]].std().replace(0, 1)
        X.loc[m, "S_low"] = (Z.fillna(0).values * SG).sum(axis=1)
O = O.merge(X[["row_id", "S_low", "seal"]], on="row_id", how="left")
d = O.a1 - O.pm
O["sg"] = np.where(O.a1.notna() & (d.abs() <= .30), O.p + .5 * d, O.p)
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
O["res"] = O.sub_ec - O.sg
S = O.seal >= .8
O["new"] = O.sg
bs = []
for k in sorted(O.validation_fold.unique()):
    tr = S & (O.validation_fold != k); te = S & (O.validation_fold == k)
    A = np.c_[np.ones(tr.sum()), O.S_low[tr]]
    b = np.linalg.lstsq(A, O.res[tr].values, rcond=None)[0]; bs.append(b)
    O.loc[te, "new"] = O.sg[te] + b[0] + b[1] * O.S_low[te]
bs = np.array(bs)
print("sealed rows %d (%.0f%%); b0 range %.3f..%.3f, b1 range %.3f..%.3f (negative = lower EC when S_low high)" % (
    S.sum(), 100 * S.mean(), bs[:, 0].min(), bs[:, 0].max(), bs[:, 1].min(), bs[:, 1].max()))
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = {}
print("\n%-16s %6s %8s %8s %8s" % ("segment", "rows", "SG2", "SD1", "change"))
for nm, m in (("all", O.dm.notna()), ("pass-2", O.day >= 179), ("high", O.dm >= 1), ("normal", O.dm < 1),
              ("pass-2 high", (O.day >= 179) & (O.dm >= 1)), ("pass-2 normal", (O.day >= 179) & (O.dm < 1)),
              ("sealed F13", S & (O.farm == "F13")), ("sealed F47", S & (O.farm == "F47"))):
    a, b = r(O.sg[m] - O.sub_ec[m]), r(O.new[m] - O.sub_ec[m]); res[nm] = (a, b)
    print("%-16s %6d %8.4f %8.4f %+7.1f%%" % (nm, m.sum(), a, b, 100 * (b / a - 1)))
O["cl"] = O.farm + "_" + (O.day // 5).astype(str)
dd = ((O.new - O.sub_ec) ** 2 - (O.sg - O.sub_ec) ** 2).groupby(O.cl).agg(["sum", "count"])
sm, n = dd["sum"].values, dd["count"].values
idx = np.random.default_rng(20261005).integers(0, len(sm), (20000, len(sm)))
p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
print("cluster bootstrap P(worse) all rows: %.4f" % p)
clue = (res["all"][1] < res["all"][0] and res["pass-2"][1] < res["pass-2"][0] and res["normal"][1] <= 1.01 * res["normal"][0]
        and res["sealed F13"][1] < res["sealed F13"][0] and res["sealed F47"][1] < res["sealed F47"][0] and p < .0125)
print("\nSD1 clue:", clue)
