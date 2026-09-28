# -*- coding: utf-8 -*-
"""DIAGNOSTIC ONLY - never used for a submission (it breaks rule 5 on purpose).

A rival team shows EC 0.1279 vs our 0.2134.  EC is a day-level quantity
(95% of its variance is between days).  We only let a row see hours 0..h of
its own day (rule 5).  Question: how much of the gap would the REST of the
day (future hours) explain?

Same ExtraTrees member as round 3 (ec_v6.et), DIAG10, seeds 7/8:
  CAUSAL     f14 + causal fingerprint (as round 3)
  NONCAUSAL  CAUSAL + full-day mean/min/max/std of every input, broadcast to
             all 24 hours of the day (uses future hours -> not allowed)
  FLOOR      true day mean of EC as the prediction (what a perfect day-level
             model would score; the remainder is within-day variation)
If NONCAUSAL lands near 0.13 the rival's number is most likely built on
information we may not use; if it stays near 0.2 the gap is elsewhere.

Run:  cd research && PYTHONPATH="" <python> -u anal_ec_noncausal.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    ins = [c for c in USABLE if c not in OUT_COLS and lab[c].notna().any()]
    agg = lab.groupby(["farm", "day"])[ins].agg(["mean", "min", "max", "std"])
    agg.columns = ["fd_%s_%s" % c for c in agg.columns]
    lab = lab.join(agg, on=["farm", "day"])
    c_nc = c_et + list(agg.columns)
    y = lab.sub_ec.values
    daymean = lab.groupby(["farm", "day"]).sub_ec.transform("mean").values
    fds = diag_folds(lab)
    print("inputs aggregated: %s" % ins, flush=True)
    for sd in (7, 8):
        ec_v6.SEED = sd
        o = {k: np.full(len(lab), np.nan) for k in ("causal", "noncausal")}
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            tr, va = lab[trm], lab[vam].reset_index(drop=True)
            for k, cols in (("causal", c_et), ("noncausal", c_nc)):
                p = ec_v6.et().fit(tr[cols], tr.sub_ec.values).predict(va[cols])
                o[k][vam] = np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46) if k == "causal" else p
        g = ~np.isnan(o["causal"])
        dm_err = lambda p: rmse(pd.Series(p[g]).groupby([lab.farm[g].values, lab.day[g].values]).transform("mean").values,
                                daymean[g])
        print("seed %d | CAUSAL %.4f (day-level err %.4f) | NONCAUSAL %.4f (day-level err %.4f) | FLOOR %.4f"
              % (sd, rmse(o["causal"][g], y[g]), dm_err(o["causal"]), rmse(o["noncausal"][g], y[g]),
                 dm_err(o["noncausal"]), rmse(daymean[g], y[g])), flush=True)


if __name__ == "__main__":
    main()
