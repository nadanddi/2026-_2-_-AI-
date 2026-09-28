# -*- coding: utf-8 -*-
"""Screening only (not an adoption test): is TabICL (BSD-3, tabicl 2.2.0,
checkpoint tabicl-regressor-v2) competitive for EC?  Two DIAG10 folds, CPU,
full training fold as context.  Compared on the same folds with the round-3
raw model and TabPFN v2 (2000 x 4 bag).  If TabICL alone is close to or better
than TabPFN, a proper pre-registered test follows.

Run:  cd research && PYTHONPATH="" <python> -u web_tabicl_screen_v1.py
"""
import time

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
from tabicl import TabICLRegressor

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from web_tabpfn_v1 import tabpfn_fit_predict


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    ones = np.ones(len(lab))
    ec_v6.SEED = 7
    for i, fd in enumerate(diag_folds(lab)[:2]):
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        yt = tr.sub_ec.values
        t0 = time.time()
        raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
               + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
               + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
        t1 = time.time()
        pf = np.mean([tabpfn_fit_predict(X[trm], y[trm], ones[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)
        t2 = time.time()
        m = TabICLRegressor(random_state=0, device="cpu", n_jobs=6)
        m.fit(X[trm], y[trm])
        ic = m.predict(X[vam])
        t3 = time.time()
        yv = y[vam]
        print("fold %d (%d val rows) | round-3 %.4f (%.0fs) | TabPFN %.4f (%.0fs) | TabICL %.4f (%.0fs) | "
              "0.5 PFN+0.5 ICL %.4f | corr(PFN,ICL) err %.2f"
              % (i, len(yv), rmse(raw, yv), t1 - t0, rmse(pf, yv), t2 - t1, rmse(ic, yv), t3 - t2,
                 rmse(0.5 * pf + 0.5 * ic, yv), np.corrcoef(pf - yv, ic - yv)[0, 1]), flush=True)


if __name__ == "__main__":
    main()
