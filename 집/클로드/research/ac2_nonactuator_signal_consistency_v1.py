# -*- coding: utf-8 -*-
"""AC2 (exploration, 2026-10-07 집 클로드).  Same as AC1 for the NON-actuator inputs: indoor climate (in_temp,
in_hum, in_co2: mean / night / day, daily range, in-out difference), outdoor weather (out_temp, out_hum, out_rad,
out_wspd: mean / night / day) and the record period (pass, record day).  High-EC days (31) vs normal (329)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
H["hi"] = (H.ec >= 1).astype(int); H["recday"] = H.day
cols = [v + s for v in ("in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd") for s in ("_mean", "_night", "_day")] + ["in_temp_range", "in_out_diff", "recday"]
rows = []
for c in cols:
    x = H[c]; auc = roc_auc_score(H.hi, x.fillna(x.median()))
    med_n = x[H.hi == 0].median(); side = 1 if x[H.hi == 1].median() >= med_n else -1
    rows.append((c, x[H.hi == 1].median(), med_n, auc, ((x[H.hi == 1] - med_n) * side > 0).mean(), ((x[H.hi == 0] - med_n) * side > 0).mean()))
T = pd.DataFrame(rows, columns=["var", "median_high", "median_normal", "AUC", "high_on_side", "normal_on_side"])
T["sep"] = (T.AUC - .5).abs()
print(T.sort_values("sep", ascending=False).drop(columns="sep").round(2).to_string(index=False))
print("\nhigh days by period (record day): <120 %d, 120-178 %d, >=179 %d | all days by period: %d / %d / %d" % (
    ((H.hi == 1) & (H.day < 120)).sum(), ((H.hi == 1) & H.day.between(120, 178)).sum(), ((H.hi == 1) & (H.day >= 179)).sum(),
    (H.day < 120).sum(), H.day.between(120, 178).sum(), (H.day >= 179).sum()))
S = H[H.act_vent_zero >= .8]
print("\nwithin SEALED days only (n %d, high %d): AUC of indoor/outdoor variables" % (len(S), S.hi.sum()))
rows = [(c, roc_auc_score(S.hi, S[c].fillna(S[c].median()))) for c in cols]
print("; ".join("%s %.2f" % (c, a) for c, a in sorted(rows, key=lambda z: -abs(z[1] - .5))[:10]))
