# -*- coding: utf-8 -*-
"""HC3 (descriptive; 2026-10-04 집 클로드).  High-EC days (label day mean >= 1, DIAG10,
R3S seed mean) split by (a) above/below the high-day mean (1.549) and (b) well
predicted (|pred - true| <= 0.20, fixed before looking) or not.  List days and
compare group means of key inputs."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
P = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
F = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
H = P[P.y >= 1].merge(F, on=["farm", "day"], how="left")
m = H.y.mean()
H["level"] = np.where(H.y > m, "평균보다 높음", "평균보다 낮음")
H["err"] = H.p - H.y
H["fit"] = np.where(H.err.abs() <= .20, "잘 맞춤", np.where(H.err < 0, "못 맞춤(과소)", "못 맞춤(과대)"))
H["grp"] = H.level + " / " + H.fit
print("high-day mean %.3f, days %d" % (m, len(H)))
print(H.groupby(["level", "fit"]).size().to_string())
K = ["farm", "day", "dong", "role", "y", "p", "err", "act_vent_zero", "out_temp_mean", "in_out_diff", "act_heating_mean", "act_co2_day",
     "act_shade_day", "act_thermal_day", "act_fog_mean", "act_circfan_mean", "in_hum_mean", "late"]
for gname, G in H.sort_values(["grp", "y"]).groupby("grp", sort=False):
    print("\n==== %s (%d days)" % (gname, len(G)))
    print(G[K].round(2).to_string(index=False))
num = ["y", "p", "err", "act_vent_zero", "out_temp_mean", "in_out_diff", "act_heating_mean", "act_co2_day", "act_shade_day", "act_thermal_day",
       "act_fog_mean", "act_circfan_mean", "in_hum_mean", "late"]
S = H.groupby("grp")[num].mean()
S["is_B"] = H.groupby("grp").dong.apply(lambda s: (s == "B").mean())
S["is_second"] = H.groupby("grp").role.apply(lambda s: (s == "second").mean())
S["F47"] = H.groupby("grp").farm.apply(lambda s: (s == "F47").mean())
print("\nGROUP MEANS"); print(S.round(2).T.to_string())
H.to_csv(os.path.join(env.LOCAL, "hc3_high_day_groups.csv"), index=False)
