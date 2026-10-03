# -*- coding: utf-8 -*-
"""MN3 (diagnostic, fixed reading; 2026-10-04 집 클로드).  Do inputs right after midnight
carry a trace of a high state that started earlier, beyond what R3S already uses
(it has the 0 h values *_h0)?  Features (legal: same greenhouse current/previous
inputs): hours 0-2 mean of 10 indoor/actuator channels + out_temp, change 0 h -> 2 h,
and the PREVIOUS record's 18-23 h mean of the same channels.  Targets: R3S day residual
(y - pred, DIAG10 seed mean) and the 0-5 h true EC, on (a) high-EC days (31),
(b) days R3S predicts >= .9 (42), (c) sealed days (85).
Reading (fixed): a trace exists if some feature has |Spearman| >= .45 with the residual
in (a) AND the same sign with |rho| >= .25 in (b) and (c)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
C = ["in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_fog", "act_co2", "out_temp"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + C)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
e = X[X.hour <= 2].groupby(["farm", "day"])[C].mean().add_suffix("_e")
h0 = X[X.hour == 0].set_index(["farm", "day"])[C]; h2 = X[X.hour == 2].set_index(["farm", "day"])[C]
dlt = (h2 - h0).add_suffix("_d02")
ev = X[X.hour >= 18].groupby(["farm", "day"])[C].mean()
pv = ev.copy(); pv.index = pd.MultiIndex.from_arrays([pv.index.get_level_values(0), pv.index.get_level_values(1) + 1]); pv = pv.add_suffix("_prevEve")
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy(); O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
D = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).join(O[O.hour <= 5].groupby(["farm", "day"]).sub_ec.mean().rename("y_early"))
F = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv")).set_index(["farm", "day"])[["act_vent_zero"]]
D = D.join([e, dlt, pv, F]); D["res"] = D.y - D.p
feats = [c for c in D.columns if c.endswith(("_e", "_d02", "_prevEve"))]
sets = {"a_high": D[D.y >= 1], "b_pred_hi": D[D.p >= .9], "c_sealed": D[D.act_vent_zero >= .8]}
R = pd.DataFrame({k: {f: spearmanr(S[f], S.res, nan_policy="omit").correlation for f in feats} for k, S in sets.items()})
R["early_EC_high"] = {f: spearmanr(sets["a_high"][f], sets["a_high"].y_early, nan_policy="omit").correlation for f in feats}
R["absA"] = R.a_high.abs()
print({k: len(v) for k, v in sets.items()})
print(R.sort_values("absA", ascending=False).drop(columns="absA").head(15).round(2).to_string())
ok = [f for f in feats if abs(R.loc[f, "a_high"]) >= .45 and np.sign(R.loc[f, "b_pred_hi"]) == np.sign(R.loc[f, "a_high"]) == np.sign(R.loc[f, "c_sealed"])
      and abs(R.loc[f, "b_pred_hi"]) >= .25 and abs(R.loc[f, "c_sealed"]) >= .25]
print("\nMN3 trace features meeting the fixed reading:", ok if ok else "none")
