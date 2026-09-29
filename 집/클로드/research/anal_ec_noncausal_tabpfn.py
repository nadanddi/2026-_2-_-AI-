# -*- coding: utf-8 -*-
"""DIAGNOSTIC ONLY (breaks rule 5 on purpose, never used for a submission).
anal_ec_noncausal.py with a stronger learner: does TabPFN, given the FULL
day's inputs, get anywhere near EC 0.128?  If even this stays far above,
the rival's EC cannot be reached from F13/F47 inputs at all.

TabPFN v2 (GPU, float32), 2000 x 4 context bag (seeds 1-4), DIAG10:
  CAUSAL     f14 + causal fingerprint (38 features, as the candidate member)
  NONCAUSAL  + full-day mean/min/max/std of the 10 inputs (40 more)
  DAYMEAN    NONCAUSAL prediction averaged over the day (a day-level model)

Run:  cd research && PYTHONPATH="" <python> -u anal_ec_noncausal_tabpfn.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds

assert torch.cuda.is_available()


def tp(Xtr, ytr, Xva, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xva)


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
    y = lab.sub_ec.values
    Xc = lab[c_et].values.astype(np.float32)
    Xn = lab[c_et + list(agg.columns)].values.astype(np.float32)
    o = {k: np.full(len(lab), np.nan) for k in ("causal", "noncausal")}
    for i, fd in enumerate(diag_folds(lab)):
        trm, vam = split_mask(lab, fd)
        for k, X in (("causal", Xc), ("noncausal", Xn)):
            o[k][vam] = np.mean([tp(X[trm], y[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)
        print("  fold %d done" % i, flush=True)
    g = ~np.isnan(o["causal"])
    key = [lab.farm.values, lab.day.values]
    dm = pd.Series(o["noncausal"]).groupby(key).transform("mean").values
    truth_dm = pd.Series(y).groupby(key).transform("mean").values

    def dl(p):
        return rmse(pd.Series(p).groupby(key).transform("mean").values[g], truth_dm[g])
    print("TabPFN DIAG10 | CAUSAL %.4f (day-level %.4f) | NONCAUSAL %.4f (day-level %.4f) | DAYMEAN %.4f"
          % (rmse(o["causal"][g], y[g]), dl(o["causal"]), rmse(o["noncausal"][g], y[g]), dl(o["noncausal"]),
             rmse(dm[g], y[g])), flush=True)


if __name__ == "__main__":
    main()
