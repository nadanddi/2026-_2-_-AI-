# -*- coding: utf-8 -*-
"""Second confirmation after temp_pfnbag8_v1.py.  W20 with an 8-sample bag
passed on samples 1-8 and 9-16, but that design was chosen after seeing
web_tabpfn_v5_gpu.py.  Fresh samples: series 1 = 17..24, series 2 = 25..32.

Rule (fixed): W20_8 = 0.6*base + 0.2*Codex + 0.2*mean(8 samples) must beat
the candidate W10 = 0.7*base + 0.2*Codex + 0.1*mean(first 4 samples of the
same series) in all 12 cells (DIAG10/EXT10/EXT12 x base 7/101 x 2 series),
DIAG10 p_worse < 0.025.  Per-sample OOFs saved (local/web_tabpfn_v6_temp_<set>.npy,
16 rows) for later bag-size studies.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v6_gpu.py
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

SERIES = {1: tuple(range(17, 25)), 2: tuple(range(25, 33))}


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
        per = {}
        for k, seeds in SERIES.items():
            for sd in seeds:
                o = np.full(len(lab), np.nan)
                for fd in fds:
                    trm, vam = split_mask(lab, fd)
                    o[vam] = V2.tabpfn_fit_predict(X[trm], y[trm], w[trm], X[vam], sd)
                per[sd] = o
            print("  %s series %d done" % (s, k), flush=True)
        allseeds = [sd for k in SERIES for sd in SERIES[k]]
        np.save(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s, np.vstack([per[sd] for sd in allseeds]))
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            for k, seeds in SERIES.items():
                m8 = np.mean([per[sd] for sd in seeds], axis=0)
                m4 = np.mean([per[sd] for sd in seeds[:4]], axis=0)
                g = ~np.isnan(base) & ~np.isnan(m8)
                w10 = 0.7 * base + 0.2 * cx + 0.1 * m4
                w20 = 0.6 * base + 0.2 * cx + 0.2 * m8
                d = rmse(w20[g], y[g]) / rmse(w10[g], y[g]) - 1
                pw = boot(lab[g].reset_index(drop=True), "sub_temp", w10[g], w20[g])[3] if s == "DIAG10" else np.nan
                ok = ok and d < 0 and (s != "DIAG10" or pw < 0.025)
                print("%-6s base %3d series %d | member8 %.5f member4 %.5f | W10 %.5f W20_8 %.5f (%+.2f%%, p_worse %.4f)"
                      % (s, bs, k, rmse(m8[g], y[g]), rmse(m4[g], y[g]), rmse(w10[g], y[g]), rmse(w20[g], y[g]),
                         100 * d, pw), flush=True)
    print("W20_8 CONFIRMATION VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
