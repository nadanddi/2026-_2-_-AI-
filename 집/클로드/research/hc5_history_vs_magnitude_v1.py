# -*- coding: utf-8 -*-
"""HC5 (descriptive; 2026-10-04 집 클로드).  User hypothesis in our data (no supply
records): is the SIZE of a high-EC day (n=31) and the EC of sealed days (n=85)
related to (a) the same 동's previous EC trajectory (labels of earlier records within
7 dates: mean, min, max, last, change) and (b) cumulative transpiration demand of the
same 동 over the previous 3/7 dates (inputs: sum of out_rad, mean in-temp, mean VPD
from in_temp/in_hum)?  Spearman; descriptive (small n)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
D = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"])
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "out_rad"])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["farm"], X["day"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int)
es = 0.6108 * np.exp(17.27 * X.in_temp / (X.in_temp + 237.3)); X["vpd"] = es * (1 - X.in_hum / 100)
XI = X.groupby(["farm", "day"]).agg(rad=("out_rad", "sum"), vpd=("vpd", "mean"), ti=("in_temp", "mean")).reset_index()
A = R.merge(XI, on=["farm", "day"], how="left").merge(D[["farm", "day", "ec", "hi", "act_vent_zero"]], on=["farm", "day"], how="left")
feat = []
for _, r in A.iterrows():
    S = A[(A.farm == r.farm) & (A.dong == r.dong) & (A.date < r.date)]
    S7 = S[S.date >= r.date - 7]; S3 = S[S.date >= r.date - 3]
    lab = S7.dropna(subset=["ec"])
    f = dict(farm=r.farm, day=r.day,
             prev_ec_mean=lab.ec.mean() if len(lab) else np.nan, prev_ec_min=lab.ec.min() if len(lab) else np.nan,
             prev_ec_max=lab.ec.max() if len(lab) else np.nan, prev_ec_last=lab.sort_values("date").ec.iloc[-1] if len(lab) else np.nan,
             prev_ec_slope=(lab.sort_values("date").ec.iloc[-1] - lab.sort_values("date").ec.iloc[0]) if len(lab) >= 2 else np.nan,
             rad3=S3.rad.mean(), rad7=S7.rad.mean(), vpd3=S3.vpd.mean(), vpd7=S7.vpd.mean(), ti3=S3.ti.mean())
    feat.append(f)
A = A.merge(pd.DataFrame(feat), on=["farm", "day"])
cols = ["prev_ec_mean", "prev_ec_min", "prev_ec_max", "prev_ec_last", "prev_ec_slope", "rad3", "rad7", "vpd3", "vpd7", "ti3"]
for name, S in (("HIGH-EC days", A[A.hi == 1]), ("SEALED days", A[A.act_vent_zero >= .8]), ("ALL labelled days", A[A.ec.notna()])):
    out = []
    for c in cols:
        z = S[[c, "ec"]].dropna()
        out.append("%s %+.2f(n%d)" % (c, spearmanr(z[c], z.ec).correlation, len(z)))
    print("%s: n %d" % (name, len(S))); print("   " + "\n   ".join(out))
