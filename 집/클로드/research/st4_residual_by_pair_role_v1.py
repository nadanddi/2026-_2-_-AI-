# -*- coding: utf-8 -*-
"""ST4 (diagnostic; 2026-10-03 집 클로드).  Do current models mis-read the pair
role (ST2/ST3: identical-weather neighbour records = one date, order ~ fixed 동)?
Day-mean residual (label - prediction) by role: first-of-pair, second-of-pair,
single; per farm x pass.  Temperature W40G (rebuilt from stored DIAG10 members),
EC R3S seed mean (DIAG10).  Roles from train_X+test_X weather (structure only).
Reading: a role effect >= .15 C (temp) or >= .05 (EC) with Wilcoxon/MWU p < .01
(first vs second) = the model does not read the 동 -> follow-up feature test."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
A = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + V) for f in ("train_X.csv", "test_X.csv")])
A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
A["farm"], A["day"], A["hour"] = A.row_id.str[:3], A.row_id.str[4:7].astype(int), A.row_id.str[8:10].astype(int)
role = {}
for f in ("F13", "F47"):
    W = A[A.farm == f].pivot_table(index="day", columns="hour", values=V)
    days = sorted(W.index); k = 0
    while k < len(days):
        d = days[k]
        if k + 1 < len(days) and days[k + 1] == d + 1:
            a, b = W.loc[d].values, W.loc[d + 1].values; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9:
                role[(f, d)], role[(f, d + 1)] = "first", "second"; k += 2; continue
        role[(f, d)] = "single"; k += 1
pd.Series(role).rename("role").rename_axis(["farm", "day"]).reset_index().to_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"), index=False)
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["p"] = 0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
T["e"] = T.sub_temp - T.p
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); E = E[E.validator == "DIAG10"].copy()
E["e"] = E.sub_ec - E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
for name, F in (("temp W40G", T), ("EC R3S", E)):
    D = F.groupby(["farm", "day"]).e.mean().reset_index()
    D["role"] = [role[(f, d)] for f, d in zip(D.farm, D.day)]
    D["pas"] = np.where(D.day >= 179, "late", "early")
    print("\n[%s] day-mean residual (label - pred)" % name)
    print(D.groupby(["farm", "pas", "role"]).e.agg(["count", "mean", "std"]).round(3).to_string())
    for (f, p), G in D.groupby(["farm", "pas"]):
        a, b = G.e[G.role == "first"], G.e[G.role == "second"]
        if len(a) >= 5 and len(b) >= 5:
            print("  %s %s second-first %+.3f  MWU p %.4f" % (f, p, b.mean() - a.mean(), mannwhitneyu(a, b).pvalue))
