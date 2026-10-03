# -*- coding: utf-8 -*-
"""MN2 (diagnostic; 2026-10-03 집 클로드).  MN1: pass-2 pair SECOND records have
huge errors (temp RMSE 1.345, EC .488 on 7 DIAG10 days).  Profile by pass x role:
label day means (EC, substrate temp, substrate - indoor gap), key inputs, and the
day-level bias of W40G / R3S; list every pass-2 pair (first, second) side by side.
Test layout: roles of the 60 evaluation days (structure only)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

R = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
C = ["in_temp", "in_hum", "act_heating", "act_vent", "act_thermal", "act_circfan", "act_fog"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + C)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X = X.merge(Y, on="row_id", how="left")
X["farm"], X["day"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int)
X["gap"] = X.sub_temp - X.in_temp
DM = X.groupby(["farm", "day"])[C + ["sub_temp", "gap", "sub_ec"]].mean().reset_index().merge(R, on=["farm", "day"])
DM["pass"] = np.where(DM.day >= 179, 2, 1)
print("label/input day means by pass x role (mean over days):")
print(DM.groupby(["pass", "role"])[["sub_ec", "sub_temp", "gap", "in_temp", "act_heating", "act_vent", "act_thermal"]].mean().round(2).to_string())
print("n days:", DM.groupby(["pass", "role"]).size().to_dict())
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["e"] = (0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn) - T.sub_temp
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); E = E[E.validator == "DIAG10"].copy()
E["e"] = E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1) - E.sub_ec
bt = T.groupby(["farm", "day"]).e.mean().rename("tbias"); be = E.groupby(["farm", "day"]).e.mean().rename("ebias")
DM = DM.merge(bt, on=["farm", "day"], how="left").merge(be, on=["farm", "day"], how="left")
P2 = DM[(DM["pass"] == 2) & DM.role.isin(["first", "second"])].sort_values(["farm", "day"])
print("\npass-2 pair records (labelled):")
print(P2[["farm", "day", "role", "sub_ec", "sub_temp", "gap", "in_temp", "act_heating", "act_vent", "tbias", "ebias"]].round(2).to_string(index=False))
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"]); te = te[te.row_id.str[:3].isin(["F13", "F47"])]
td = pd.DataFrame({"farm": te.row_id.str[:3], "day": te.row_id.str[4:7].astype(int)}).drop_duplicates().merge(R, on=["farm", "day"])
print("\nTEST days pairs:", td[td.role != "single"].sort_values(["farm", "day"]).to_string(index=False))
