# -*- coding: utf-8 -*-
"""HG3ALL (supporting evidence, rule fixed = HG3 6.329, fixed before running; 2026-10-05 집 클로드).
HG3 failed only on power (2nd-pass high days = 5).  The mechanism is not pass-specific, so apply the
SAME rule (pm_h >= .9, best two SG2 anchors >= 1.0, hour-causal S_low <= -1.0, step .5; SG2 elsewhere)
to ALL 360 DIAG10 days (31 high days) on stored R3S seed-mean OOF (hk0_rows_v1.csv; anchors = SG2
search with reference outside the fold).  Not a new formal judgement (stored seeds, rows seen in
HK0); reported: change vs SG2 overall / high / normal / pass-1 / pass-2, per farm, and farm x 5-day
cluster bootstrap P(worse) over all rows.  Reading: supports HG3 if all rows improve with P < .025,
normal days unchanged (<= +0.5 %), and both farms improve."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X.sort_values(["farm", "day", "hour"])
DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]; SG = np.array([1, 1, 1, 1, -1, -1.])
g = X.groupby(["farm", "day"])
for v in DV:
    X["cm_" + v] = g[v].transform(lambda z: z.expanding().mean())
for f in ("F13", "F47"):
    for h in range(24):
        m = (X.farm == f) & (X.hour == h); C = X.loc[m, ["cm_" + v for v in DV]]
        X.loc[m, "S_low"] = (((C - C.mean()) / C.std().replace(0, 1)).fillna(0).values * SG).sum(axis=1)
O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv")).merge(X[["row_id", "S_low"]], on="row_id")
d = O.a1 - O.pm
O["sg"] = np.where(O.a1.notna() & (d.abs() <= .3), O.p + .5 * d, O.p)
gate = O.a1.notna() & O.a2.notna() & (O.a1 >= 1) & (O.a2 >= 1) & (O.pm >= .9) & (O.S_low <= -1.0)
O["hg3"] = np.where(gate, O.p + .5 * ((O.a1 + O.a2) / 2 - O.pm), O.sg)
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("gated rows %d (days %d: high %d, normal %d)" % (gate.sum(), O[gate].groupby(["farm", "day"]).ngroups,
      O[gate & (O.dm >= 1)].groupby(["farm", "day"]).ngroups, O[gate & (O.dm < 1)].groupby(["farm", "day"]).ngroups))
res = {}
for nm, m in (("all", O.dm.notna()), ("high", O.dm >= 1), ("normal", O.dm < 1), ("pass-1", O.day < 179), ("pass-2", O.day >= 179),
              ("F13", O.farm == "F13"), ("F47", O.farm == "F47")):
    a, b = r(O.sg[m] - O.sub_ec[m]), r(O.hg3[m] - O.sub_ec[m]); res[nm] = 100 * (b / a - 1)
    print("  %-7s SG2 %.4f -> HG3 %.4f (%+.2f%%)" % (nm, a, b, res[nm]))
O["cl"] = O.farm + "_" + (O.day // 5).astype(str)
dd = ((O.hg3 - O.sub_ec) ** 2 - (O.sg - O.sub_ec) ** 2).groupby(O.cl).agg(["sum", "count"])
sm, n = dd["sum"].values, dd["count"].values
idx = np.random.default_rng(20261005).integers(0, len(sm), (20000, len(sm)))
p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
print("cluster bootstrap P(worse) all rows: %.4f" % p)
print("normal gated days:", O[gate & (O.dm < 1)].groupby(["farm", "day"]).dm.first().round(2).to_dict())
print("HG3ALL supports HG3:", res["all"] < 0 and p < .025 and res["normal"] <= .5 and res["F13"] < 0 and res["F47"] < 0)
