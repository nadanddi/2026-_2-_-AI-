# -*- coding: utf-8 -*-
"""Confirmation of temp_pfnweight_v1.py (W20 passed on the SAME bag OOFs the
member was first judged on).  Fresh context samples: bags {9..12} and
{13..16}, never used before; everything else as web_tabpfn_v2_gpu.py.

Same pre-set rule as temp_pfnweight_v1.py for W20 only:
  0.6*base + 0.2*Codex + 0.2*bag  vs  0.7*base + 0.2*Codex + 0.1*bag
better in all 12 cells (DIAG10/EXT10/EXT12 x base 7/101 x 2 fresh bags),
DIAG10 p_worse < 0.025 in all 4 DIAG10 cells (single arm now).
Also reported: W20 vs the pre-TabPFN reference 0.8*base + 0.2*Codex.
OOFs saved as local/web_tabpfn_v5_temp_<set>.npy.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v5_gpu.py
"""
import os
import sys

import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF
import web_tabpfn_v2_gpu  # noqa: F401  (patches the TabPFN call to CUDA float32)
import web_tabpfn_v2 as V2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

BAGS = {1: (9, 10, 11, 12), 2: (13, 14, 15, 16)}


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    X = lab[FEATURE_COLUMNS].values.astype(np.float32)
    ph = F4.phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ok = True
    for s, fds in sets:
        mem = {}
        for b, seeds in BAGS.items():
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[vam] = V2.bag_predict(X[trm], y[trm], w[trm], X[vam], seeds)[0]
            mem[b] = o
        np.save(env.LOCAL + "/web_tabpfn_v5_temp_%s.npy" % s, np.vstack([mem[1], mem[2]]))
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            for b in BAGS:
                g = ~np.isnan(base) & ~np.isnan(mem[b])
                old = 0.8 * base + 0.2 * cx
                w10 = 0.7 * base + 0.2 * cx + 0.1 * mem[b]
                w20 = 0.6 * base + 0.2 * cx + 0.2 * mem[b]
                d = rmse(w20[g], y[g]) / rmse(w10[g], y[g]) - 1
                pw = boot(lab[g].reset_index(drop=True), "sub_temp", w10[g], w20[g])[3] if s == "DIAG10" else np.nan
                ok = ok and d < 0 and (s != "DIAG10" or pw < 0.025)
                print("%-6s base %3d bag %d | member %.5f | W10 %.5f W20 %.5f (%+.2f%%, p_worse %.4f) | W20 vs no-TabPFN %+.2f%%"
                      % (s, bs, b, rmse(mem[b][g], y[g]), rmse(w10[g], y[g]), rmse(w20[g], y[g]), 100 * d, pw,
                         100 * (rmse(w20[g], y[g]) / rmse(old[g], y[g]) - 1)), flush=True)
    print("W20 CONFIRMATION VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
