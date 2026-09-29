# -*- coding: utf-8 -*-
"""F32 pump effect net of air temperature; F37/F32 non-integer label segments (rounding mode)."""
import env  # noqa
import numpy as np, pandas as pd
P = pd.read_pickle(env.LOCAL + "/eda_farm_panels.pkl")
p = P["F32"].copy()
d = p.groupby("day").agg(sub=("sub_temp", "mean"), air=("in_temp", "mean"), pump=("act_pump", "mean"),
                         valve=("act_valve", "mean"), cool=("act_cool", "mean"), side=("act_side", "mean")).dropna()
A = np.c_[np.ones(len(d)), d[["air", "pump", "valve", "cool", "side"]].values]
c, *_ = np.linalg.lstsq(A, d["sub"].values, rcond=None)
print("F32 daily sub ~ air+pump+valve+cool+side (pump etc in %%):", np.round(c, 4), "n=", len(d))
A0 = A[:, :2]; c0, *_ = np.linalg.lstsq(A0, d["sub"].values, rcond=None)
r0 = d["sub"].values - A0 @ c0; r1 = d["sub"].values - A @ c
print("daily RMSE air-only %.3f vs +actuators %.3f" % (r0.std(), r1.std()))
# hourly: sub ~ ew(air) + pump_ewm
p["pump_e"] = p.act_pump.ewm(halflife=3).mean()
L = p[p.sub_temp.notna()].dropna(subset=["ew2", "ew8", "ew24", "pump_e"])
for cols in (["ew2", "ew8", "ew24"], ["ew2", "ew8", "ew24", "pump_e"]):
    A = np.c_[np.ones(len(L)), L[cols].values]; c, *_ = np.linalg.lstsq(A, L.sub_temp.values, rcond=None)
    print(cols, "hourly RMSE %.3f coef %s" % ((L.sub_temp.values - A @ c).std(), np.round(c, 3)))
# pump vs other F32 inputs available in F13/F47 (proxy?)
for c in ["out_rad", "in_temp", "in_hum", "act_vent", "hour"]:
    print("corr pump,", c, round(p.act_pump.corr(p[c]), 3))
print("pump on-share by hour:", p.groupby("hour").act_pump.mean().round(0).values)
for f in ["F32", "F37"]:
    q = P[f]; L = q[q.sub_temp.notna()].copy()
    L["isint"] = np.isclose(L.sub_temp, np.round(L.sub_temp))
    s = L.groupby("day").isint.mean()
    print(f, "days all-int %d, none-int %d, mixed %d; non-int day range %s" % ((s == 1).sum(), (s == 0).sum(), ((s > 0) & (s < 1)).sum(),
          (s[s < 1].index.min(), s[s < 1].index.max())))
    ni = L[~L.isint]
    print(f, "non-int labels sample", ni.sub_temp.head(8).round(3).tolist())
