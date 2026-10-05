# -*- coding: utf-8 -*-
"""HG frontier (POST-HOC design exploration, 2026-10-05 집 클로드; not a judgement).
Trade-off of the HG2-type high-EC correction on top of SG2 as the two knobs change:
domain-guard threshold T (apply only if hour-causal S_low <= T) and step w (pred = p + w (mean(a1,a2) - pm_h)).
Gate: pm_h >= .9 and best two anchors >= 1.0.  Data: DIAG10 (hk0_rows_v1.csv, R3S seed mean, SG2
anchors) and EL1 (ec3_AF0_all.csv anchors af_a1/af_a2, seeds 7/101/2024 averaged).
S_low as in SD1 (0..h means, z per farm per h over labelled training rows)."""
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
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv")).merge(X[["row_id", "S_low"]], on="row_id")
dd = H.a1 - H.pm; H["sg"] = np.where(H.a1.notna() & (dd.abs() <= .3), H.p + .5 * dd, H.p)
H = H[H.day >= 179].copy(); H["validator"] = "DIAG10"
A = pd.read_csv(os.path.join(env.LOCAL, "ec3_AF0_all.csv")); A = A[A.validator == "EL1"].sort_values(["farm", "day", "hour"]).copy()
A["p"] = A[["base_7", "base_101", "base_2024"]].mean(axis=1); A["sg"] = A[["sg_7", "sg_101", "sg_2024"]].mean(axis=1)
A["pm"] = A.groupby(["farm", "day"]).p.transform(lambda z: z.expanding().mean()); A["a1"], A["a2"] = A.af_a1, A.af_a2
A = A.merge(X[["row_id", "S_low"]], on="row_id")
print("pass-2 rows, change vs SG2:  all / high / normal   (DIAG10 | EL1)")
for T in (None, 0.0, -0.5, -1.0):
    for w in (0.5, 0.35, 0.2):
        out = []
        for D in (H, A):
            D = D.copy(); D["dm"] = D.groupby(["farm", "day"]).sub_ec.transform("mean")
            gate = D.a1.notna() & D.a2.notna() & (D.a1 >= 1) & (D.a2 >= 1) & (D.pm >= .9)
            if T is not None:
                gate &= D.S_low <= T
            new = np.where(gate, D.p + w * ((D.a1 + D.a2) / 2 - D.pm), D.sg)
            c = [100 * (r(new[m] - D.sub_ec[m]) / r(D.sg[m] - D.sub_ec[m]) - 1) for m in (D.dm.notna(), D.dm >= 1, D.dm < 1)]
            out.append("%+5.1f / %+5.1f / %+5.1f" % tuple(c))
        print("  guard %-5s step %.2f :  %s  |  %s" % ("none" if T is None else T, w, out[0], out[1]))
