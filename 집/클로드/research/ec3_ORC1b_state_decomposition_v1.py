# -*- coding: utf-8 -*-
"""EC ORC1b (diagnostic, label post-hoc; 2026-10-03 집 클로드).
Replaces ORC1/ISO0, which were run as inline code whose source was not saved
(Codex review 6.199).  Explicit definition here: high-EC day = DIAG10 label day
mean >= 1.0 (same as Codex 6.183, 31/360 days).  Model = R3S seed mean (DC5 OOF).
Reports: AUC(pred day mean -> high), mean label/pred on high vs normal days,
RMSE if the state were known (per-state mean residual shift, estimated on the
OTHER DIAG10 folds = out-of-fold), and RMSE with a perfect daily level.
Label-based oracle: NOT an evaluation feature; diagnostic only.
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

o = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
d = o[o.validator == "DIAG10"].copy()
d["p"] = d[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
d["ym"] = d.groupby(["farm", "day"]).sub_ec.transform("mean")
d["pm"] = d.groupby(["farm", "day"]).p.transform("mean")
d["hi"] = d.ym >= 1.0
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
day = d.groupby(["farm", "day"]).agg(hi=("hi", "first"), ym=("ym", "first"), pm=("pm", "first"))
print("DIAG10 days %d, high (label day mean >= 1) %d" % (len(day), day.hi.sum()))
print("AUC(pred day mean -> high) %.3f" % roc_auc_score(day.hi, day.pm))
print("high days: mean label %.3f pred %.3f | normal: label %.3f pred %.3f" % (
    day.ym[day.hi].mean(), day.pm[day.hi].mean(), day.ym[~day.hi].mean(), day.pm[~day.hi].mean()))
sh = np.zeros(len(d))
for f in d.validation_fold.unique():
    tr, te = d.validation_fold != f, d.validation_fold == f
    for s in (True, False):
        m = (d.sub_ec - d.p)[tr & (d.hi == s)].mean()
        sh[(te & (d.hi == s)).values] = m
print("R3S RMSE %.4f | + known state (OOF shift) %.4f | perfect daily level %.4f" % (
    r(d.p - d.sub_ec), r(d.p + sh - d.sub_ec), r(d.p - d.pm + d.ym - d.sub_ec)))
print("share of R3S SSE on high days %.1f%%" % (100 * ((d.p - d.sub_ec) ** 2)[d.hi].sum() / ((d.p - d.sub_ec) ** 2).sum()))
