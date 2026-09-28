# -*- coding: utf-8 -*-
"""A second temperature TabPFN member on a DIFFERENT view: the MASK-world
features of our own base members (ct + phys, temp_mask_v1.build_world),
instead of the Codex view.  Diversity test on top of the candidate G_C2
(catalog 6.70).

Member CT8: 8-sample bag (samples 101..108 / 111..118, weighted context as
all temperature members), GPU float32.  Codex-view member: saved 8-sample
OOFs (samples 1-8 from v2, 17-24 from v6) paired with the two series.
Arm (fixed): G_C2_2 = G_C2 with the TabPFN share 0.2g split into
  0.1g Codex-view + 0.1g ct-view.
Rule: better than G_C2 in all 12 cells (DIAG10/EXT10/EXT12 x seed pairs
7/726, 101/727 x 2 series), DIAG10 p_worse < 0.025.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_ct_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np

import common
import harness
from common import split_mask, rmse, TARGET_FARMS
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader, build_world, ORIG
import web_tabpfn_v2_gpu  # noqa: F401  (CUDA float32 TabPFN call)
import web_tabpfn_v2 as V2

SERIES = {"A": (tuple(range(101, 109)), "v2"), "B": (tuple(range(111, 119)), "v6")}


def main():
    common.load_raw = masked_loader
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    cols = list(dict.fromkeys(ct + phc))
    print("ct view: %d features" % len(cols), flush=True)
    X = lab[cols].values.astype(np.float32)
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ok = True
    for s, fds in sets:
        cxv = {"v2": np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
               "v6": np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0)}
        ctm = {}
        for k, (seeds, _) in SERIES.items():
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[vam] = V2.bag_predict(X[trm], y[trm], w[trm], X[vam], seeds)[0]
            ctm[k] = o
            print("  %s series %s done" % (s, k), flush=True)
        np.save(env.LOCAL + "/web_tabpfn_ct_temp_%s.npy" % s, np.vstack([ctm["A"], ctm["B"]]))
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for k, (_, src) in SERIES.items():
                m, mc = cxv[src], ctm[k]
                g = ~np.isnan(base) & ~np.isnan(m) & ~np.isnan(mc)
                wb, wc = 0.6 - 0.2 * (1 - g8), 0.2 + 0.4 * (1 - g8)
                ref = wb * base + wc * cx + 0.2 * g8 * m
                cand = wb * base + wc * cx + 0.1 * g8 * m + 0.1 * g8 * mc
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])[3] if s == "DIAG10" else np.nan
                ok = ok and d < 0 and (s != "DIAG10" or pw < 0.025)
                print("%-6s seeds %3d/%d series %s | ct member %.5f codex-view member %.5f corr(err) %.2f | G_C2 %.5f G_C2_2 %.5f (%+.2f%%, p_worse %.4f)"
                      % (s, bs, cs, k, rmse(mc[g], y[g]), rmse(m[g], y[g]),
                         np.corrcoef((mc - y)[g], (m - y)[g])[0, 1], rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, pw),
                      flush=True)
    print("G_C2_2 PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
