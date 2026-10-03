# -*- coding: utf-8 -*-
"""NX0 (diagnostic; 2026-10-04 집 클로드).
(1) Substrate-temperature link: on high-EC days (31) and sealed days (85), Spearman of
    the EC day mean and of the R3S residual with the TRUE substrate temp day mean, the
    substrate-indoor gap, and the W40G-predicted gap (DIAG10 temperature OOF rebuilt;
    legal at evaluation since it comes from inputs).
(2) Pass-2 high days: their season index as used in DIAG10 (interpolated) vs the season
    their own weather twin points to (SE0 table)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy(); O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
E = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index(); E["res"] = E.y - E.p
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["w40"] = 0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
TD = T.groupby(["farm", "day"]).agg(st=("sub_temp", "mean"), ti=("in_temp", "mean"), w40=("w40", "mean")).reset_index()
TD["gap"] = TD.st - TD.ti; TD["gap_pred"] = TD.w40 - TD.ti
F = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))[["farm", "day", "act_vent_zero"]]
D = E.merge(TD, on=["farm", "day"]).merge(F, on=["farm", "day"])
for nm, S in (("HIGH-EC (y>=1)", D[D.y >= 1]), ("SEALED", D[D.act_vent_zero >= .8]), ("ALL", D)):
    out = []
    for c in ("st", "gap", "w40", "gap_pred"):
        out.append("%s: EC %+.2f resid %+.2f" % (c, spearmanr(S[c], S.y).correlation, spearmanr(S[c], S.res).correlation))
    print("%-15s n %3d | %s" % (nm, len(S), " | ".join(out)))
S0 = pd.read_csv(os.path.join(env.LOCAL, "se0_diag_season_gap.csv"))
H = D[(D.y >= 1) & (D.day >= 179)].merge(S0[["farm", "day", "s_interp", "s_twin", "exact"]], on=["farm", "day"], how="left")
print("\npass-2 high days: season used (interp) vs own-weather twin")
print(H[["farm", "day", "y", "p", "res", "s_interp", "s_twin", "exact"]].round(2).to_string(index=False))
