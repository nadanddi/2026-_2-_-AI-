# -*- coding: utf-8 -*-
"""SF1b (POST-HOC, descriptive only): among the 53 sealed OOF days, how well do
single day-level quantities separate high-EC (16) from normal (37), and does
anything add to v2's own daily mean prediction (pm)?  Uses local/sf1_daytable.csv.
sub_temp is a LABEL (not an input) - shown only as a mechanism clue.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u sf1b_sealed_auc_posthoc.py"""
import env  # noqa
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
D = pd.read_csv(os.path.join(env.LOCAL, "sf1_daytable.csv"))
S = D[D.oof & D.sealed].copy()
S["high"] = (S.ec >= 1.2).astype(int)
print("single-feature AUC (high vs normal, sealed days; 0.5 = none; <0.5 means lower in high)")
for c in ("pm", "ec_m2", "ec_m4", "ec_m1", "shade", "co2sup", "tin", "tin_min", "fan", "fog", "tout", "day", "heat", "tsub"):
    m = S[c].notna()
    print("  %-7s n=%2d AUC %.3f" % (c, m.sum(), roc_auc_score(S.high[m], S[c][m])))
# incremental over pm: leave-one-out logistic on [pm] vs [pm + x]
def loo(cols):
    Z = S[cols].values
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    p = np.zeros(len(S))
    for tr, te in LeaveOneOut().split(Z):
        p[te] = LogisticRegression(C=1.0).fit(Z[tr], S.high.values[tr]).predict_proba(Z[te])[:, 1]
    return roc_auc_score(S.high, p)
print("\nLOO logistic AUC: pm %.3f" % loo(["pm"]))
for x in (["shade"], ["co2sup"], ["tin"], ["shade", "co2sup"], ["shade", "co2sup", "tin", "fan", "tout"], ["day"]):
    print("  pm + %-32s %.3f" % ("+".join(x), loo(["pm"] + x)))
print("  inputs only shade+co2sup+tin+fan+tout      %.3f" % loo(["shade", "co2sup", "tin", "fan", "tout"]))
# day-index structure of high sealed days
print("\nhigh sealed days by day index:", sorted(zip(S.farm[S.high == 1], S.day[S.high == 1])))
print("sealed days per 20-day window (high/all):")
S["w"] = (S.day // 20) * 20
print(S.groupby("w").high.agg(["sum", "count"]).T.to_string())
