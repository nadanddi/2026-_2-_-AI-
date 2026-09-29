# -*- coding: utf-8 -*-
"""DIAGNOSTIC (uses the true sealed-day label as gate; never for adoption).
Is a sealed-day-context TabPFN better than round-3 / the v2 blend ON sealed
days at all?  DIAG10, round-3 seed 7, samples 1-4 / 21-24.  Reads the
anal_ec_regime_tabpfn.npz OOFs for round-3 and the full-context bag.
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import common
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from anal_ec_noncausal_tabpfn import tp

_, _, lab0 = load()
fp = F4.fp_features()
lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
X = lab[f14 + F4.names(fp)].values.astype(np.float32)
y = lab.sub_ec.values
z = np.load(env.LOCAL + "/anal_ec_regime_tabpfn.npz", allow_pickle=True)
assert (z["row_id"] == lab.row_id.values).all()
raw, pfn, s = z["raw"], z["pfn"], z["sealed"]
spec = np.full(len(lab), np.nan)
for fd in diag_folds(lab):
    trm, vam = split_mask(lab, fd)
    m = vam & s
    if m.any():
        spec[m] = np.mean([tp(X[trm & s], y[trm & s], X[m], 21 + i) for i in range(4)], axis=0)
g = s & ~np.isnan(spec)
print("sealed days only (raw, no shrink): round-3 %.4f | full-context TabPFN %.4f | sealed-context TabPFN %.4f | "
      "0.5 r3 + 0.5 sealed %.4f | 0.8 r3 + 0.2 full %.4f"
      % (rmse(raw[g], y[g]), rmse(pfn[g], y[g]), rmse(spec[g], y[g]), rmse(0.5 * raw[g] + 0.5 * spec[g], y[g]),
         rmse(0.8 * raw[g] + 0.2 * pfn[g], y[g])))
print("bias: round-3 %+.3f | full %+.3f | sealed-ctx %+.3f" % ((raw - y)[g].mean(), (pfn - y)[g].mean(), (spec - y)[g].mean()))
