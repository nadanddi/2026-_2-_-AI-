# -*- coding: utf-8 -*-
"""Where does TabPFN help EC - sealed days or ordinary days?  (analysis)

DIAG10, round-3 raw (seed 7) and TabPFN bag (2000 x 4, seeds 1-4, GPU).
For blend weights 0..1 report RMSE on sealed / non-sealed days (full-day
definition of catalog 6.11, analysis only).  OOFs saved to
local/anal_ec_regime_tabpfn.npz for later regime-aware designs.

Run:  cd research && PYTHONPATH="" <python> -u anal_ec_regime_tabpfn.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from anal_ec_regime import sealed_days
from anal_ec_noncausal_tabpfn import tp


def main():
    tX, _, _ = common.load_raw()
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    f13 = [c for c in f14 if c != "day"]
    c_sub = f13 + F4.names(fp)          # round-3 ET view as submitted
    c_pfn = f14 + F4.names(fp)          # TabPFN member view (candidate)
    y = lab.sub_ec.values
    X = lab[c_pfn].values.astype(np.float32)
    s_tr = sealed_days(tX[tX.farm.isin(TARGET_FARMS)])
    is_s = np.array([bool(s_tr.get(k, False)) for k in zip(lab.farm, lab.day)])
    ec_v6.SEED = 7
    raw = np.full(len(lab), np.nan)
    pfn = np.full(len(lab), np.nan)
    for i, fd in enumerate(diag_folds(lab)):
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        yt = tr.sub_ec.values
        raw[vam] = (0.6 * ec_v6.et().fit(tr[c_sub], yt).predict(va[c_sub])
                    + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
                    + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
        pfn[vam] = np.mean([tp(X[trm], y[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)
        print("  fold %d done" % i, flush=True)
    np.savez(env.LOCAL + "/anal_ec_regime_tabpfn.npz", row_id=lab.row_id.values, raw=raw, pfn=pfn, sealed=is_s)
    fds = diag_folds(lab)
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p = np.full(len(lab), np.nan)
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            p[vam] = np.clip(causal_shrink((1 - w) * raw[vam] + w * pfn[vam], lab[vam].reset_index(drop=True), 0.5),
                             0.062, 3.46)
        print("w %.1f | all %.4f | sealed %.4f | non-sealed %.4f | sealed bias %+.3f"
              % (w, rmse(p, y), rmse(p[is_s], y[is_s]), rmse(p[~is_s], y[~is_s]), (p - y)[is_s].mean()), flush=True)


if __name__ == "__main__":
    main()
