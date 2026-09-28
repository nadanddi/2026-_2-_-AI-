# -*- coding: utf-8 -*-
"""DIAGNOSTIC ONLY (breaks rule 5 on purpose, never used for a submission).
Temperature counterpart of anal_ec_noncausal*.py.  The rival shows 0.49 vs our
0.5456 (-10%).  Our LB/DIAG10 ratio is ~0.79, so 0.49 ~ 0.62 on DIAG10.
How much would FUTURE inputs buy?

Codex member (ridge physics base + LightGBM residual, round-5 weights),
DIAG10, seeds 726/727:
  CAUSAL     Codex features (89)
  NONCAUSAL  + full-day mean/min/max/std of in_temp, in_hum, in_co2, out_temp,
             out_rad, act_heating, act_thermal + leads +1/+2/+3/+6 h of
             in_temp and out_temp (same greenhouse)

Run:  cd research && PYTHONPATH="" <python> -u anal_temp_noncausal.py
"""
import os
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from common import split_mask, rmse
from harness import load
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
from temp_transfer_v2 import codex_fit_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

AGG = ["in_temp", "in_hum", "in_co2", "out_temp", "out_rad", "act_heating", "act_thermal"]


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    a = pd.concat([tX, sX], ignore_index=True)
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"])
    agg = a.groupby(["farm", "day"])[AGG].agg(["mean", "min", "max", "std"])
    agg.columns = ["fd_%s_%s" % c for c in agg.columns]
    lead = pd.DataFrame({"row_id": a.row_id.values})
    for c in ("in_temp", "out_temp"):
        for L in (1, 2, 3, 6):
            lead["%s_lead%d" % (c, L)] = a.groupby("farm")[c].shift(-L).values
    lab = lab.join(agg, on=["farm", "day"]).merge(lead, on="row_id", how="left")
    extra = list(agg.columns) + [c for c in lead.columns if c != "row_id"]
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    fds = diag_folds(lab)
    for sd in (726, 727):
        o = {k: np.full(len(lab), np.nan) for k in ("causal", "noncausal")}
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            o["causal"][vam] = codex_fit_predict(lab[trm], lab[vam], w[trm], sd, FEATURE_COLUMNS)
            o["noncausal"][vam] = codex_fit_predict(lab[trm], lab[vam], w[trm], sd, FEATURE_COLUMNS + extra)
        g = ~np.isnan(o["causal"])
        a1, b1 = rmse(o["causal"][g], y[g]), rmse(o["noncausal"][g], y[g])
        print("seed %d | Codex CAUSAL %.4f | NONCAUSAL %.4f (%+.1f%%) | rival-equivalent on DIAG10 ~0.62"
              % (sd, a1, b1, 100 * (b1 / a1 - 1)), flush=True)


if __name__ == "__main__":
    main()
