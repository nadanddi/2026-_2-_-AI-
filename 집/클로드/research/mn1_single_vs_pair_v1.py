# -*- coding: utf-8 -*-
"""MN1 (diagnostic; 2026-10-03 집 클로드).  MN0: single records (dates with one
record) have larger temperature error at all hours.  Is it the pass-2 share or a
property of singles?  DIAG10, pass 1 only and by calendar thirds of pass 1, RMSE of
W40G (temp) and R3S (EC) by role; also how singles are distributed along pass 1
(runs of singles)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

R = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["p"] = 0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
T["y"] = T.sub_temp
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); E = E[E.validator == "DIAG10"].copy()
E["p"] = E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1); E["y"] = E.sub_ec
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for name, F in (("temp W40G", T), ("EC R3S", E)):
    F = F.merge(R, on=["farm", "day"])
    F["seg"] = np.where(F.day >= 179, "pass2", np.where(F.day < 60, "p1a(<60)", np.where(F.day < 120, "p1b(60-119)", "p1c(120-178)")))
    print("\n[%s] RMSE (n days) by segment x role" % name)
    tab = F.groupby(["seg", "role"]).apply(lambda G: "%.3f (%d)" % (r(G.p - G.y), G.day.nunique()), include_groups=False).unstack()
    print(tab.to_string())
    D = F.groupby(["farm", "day", "role", "seg"]).apply(lambda G: (G.p - G.y).mean(), include_groups=False).rename("bias").reset_index()
    print(" day-level |bias| mean by seg x role:")
    print(D.assign(ab=D.bias.abs()).groupby(["seg", "role"]).ab.mean().unstack().round(3).to_string())
seq = R[R.day < 179].sort_values(["farm", "day"])
for f, G in seq.groupby("farm"):
    s = "".join({"first": "[", "second": "]", "single": "."}[x] for x in G.role)
    print("\n%s pass1 role sequence: %s" % (f, s))
