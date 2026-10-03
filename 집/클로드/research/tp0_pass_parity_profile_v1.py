# -*- coding: utf-8 -*-
"""TP0 (diagnostic; 2026-10-03 집 클로드).  Does the source (동) <-> record-day
parity mapping or the source character change between pass 1 (<179) and pass 2
(>=179)?  F47 late temperature error (W30G day RMSE 1.03) is twice F13 late.
Per farm x pass x parity (DIAG10 public OOF, labels of training data):
  temperature: mean substrate-indoor gap, mean W30G residual (seeds 7/101) and RMSE;
  EC: mean label, share of high-EC days (day mean >= 1);
  inputs (train_X, all labelled+unlabelled days): mean in_temp, act_heating, act_vent.
Descriptive; no model selection."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
T["w40"] = np.nan
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
m = (T.mask_base_7 + T.mask_base_101) / 2
T["w40"] = 0.4 * m + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
T["res"] = T.sub_temp - T.w40
T["gap"] = T.sub_temp - T.in_temp
T["pas"], T["par"] = (T.day >= 179).astype(int), T.day % 2
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
E = E[E.validator == "DIAG10"].copy()
E["pas"], E["par"] = (E.day >= 179).astype(int), E.day % 2
ed = E.groupby(["farm", "pas", "par", "day"]).sub_ec.mean().reset_index()
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "act_heating", "act_vent", "out_temp"])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int)
X["pas"], X["par"] = (X.day >= 179).astype(int), X.day % 2
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("W40G (from stored members, PFN mean) DIAG10 RMSE %.4f" % r(T.res))
out = []
for k, G in T.groupby(["farm", "pas", "par"]):
    e = ed[(ed.farm == k[0]) & (ed.pas == k[1]) & (ed.par == k[2])]
    x = X[(X.farm == k[0]) & (X.pas == k[1]) & (X.par == k[2])]
    out.append(dict(farm=k[0], pas=k[1], par=k[2], days=G.day.nunique(), gap=G.gap.mean(), res=G.res.mean(),
                    rmse=r(G.res), ec=e.sub_ec.mean(), hiEC=(e.sub_ec >= 1).mean(), xdays=x.day.nunique(),
                    in_temp=x.in_temp.mean(), heat=x.act_heating.mean(), vent=x.act_vent.mean(), out_temp=x.out_temp.mean()))
print(pd.DataFrame(out).round(3).to_string(index=False))
