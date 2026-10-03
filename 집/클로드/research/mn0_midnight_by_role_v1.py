# -*- coding: utf-8 -*-
"""MN0 (diagnostic; 2026-10-03 집 클로드).  Temperature error at hours 0-2 is
outsized (6b.13).  C6.205: for the SECOND record of a same-date pair the previous
record is the other 동's SAME date (its 23h is that date's end), not the previous
evening.  Any cross-midnight feature would be mis-read there.
DIAG10 W40G (rebuilt) and members: RMSE / bias by hour block (0-2, 3-9, 10-23) x
pair role (first / second / single), per farm.  Also the jump between the previous
record's 23h and this record's 0h for in_temp, by role."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

R = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
mask = (T.mask_base_7 + T.mask_base_101) / 2
T["w40"] = 0.4 * mask + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
T = T.merge(R, on=["farm", "day"], how="left")
T["hb"] = pd.cut(T.hour, [-1, 2, 9, 23], labels=["h0-2", "h3-9", "h10-23"])
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
out = []
for (rl, hb), G in T.groupby(["role", "hb"], observed=True):
    out.append(dict(role=rl, hours=hb, n=len(G), w40=r(G.w40 - G.sub_temp), bias=(G.w40 - G.sub_temp).mean(),
                    mask=r(mask[G.index] - G.sub_temp), codex=r(G.codex_base - G.sub_temp), pfn=r(G.pfn - G.sub_temp)))
print(pd.DataFrame(out).round(3).to_string(index=False))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
P = X.pivot_table(index=["farm", "day"], columns="hour", values="in_temp")
J = []
for (f, d) in P.index:
    if (f, d - 1) in P.index:
        J.append(dict(farm=f, day=d, jump=abs(P.loc[(f, d), 0] - P.loc[(f, d - 1), 23])))
J = pd.DataFrame(J).merge(R, on=["farm", "day"])
print("\n|in_temp(0h) - previous record in_temp(23h)| by role:")
print(J.groupby("role").jump.describe()[["count", "50%", "mean"]].round(2).to_string())
